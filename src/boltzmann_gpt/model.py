"""AttributeModel: the public inference API.

Pipeline: binary attribute vector ``v`` -> DBM mean-field beliefs ``H`` ->
adapter -> ``K`` soft-prompt embeddings -> frozen LLM. Only the DBM and the
adapter are shipped here; the generator is loaded from the Hugging Face Hub and
is never modified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import torch

from .adapter import BeliefAdapter
from .dbm import DeepBoltzmannMachine
from .features import FeatureSpec, ValueSpec

CONFIG_FILE = "config.json"
FEATURE_SPEC_FILE = "feature_spec.json"
DBM_FILE = "dbm.safetensors"
ADAPTER_FILE = "adapter.safetensors"

DEFAULT_LLM_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def select_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """cuda -> mps -> cpu, unless a device is given explicitly."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve(repo_id_or_path: Union[str, Path]) -> Path:
    """Return a local checkpoint directory, downloading from the Hub if needed."""
    local = Path(repo_id_or_path).expanduser()
    if local.is_dir():
        return local

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"{repo_id_or_path!r} is not a local directory and huggingface_hub "
            "is not installed; install it to fetch checkpoints from the Hub."
        ) from exc

    return Path(
        snapshot_download(
            repo_id=str(repo_id_or_path),
            allow_patterns=[CONFIG_FILE, FEATURE_SPEC_FILE, DBM_FILE, ADAPTER_FILE],
        )
    )


class AttributeModel:
    """Energy-based attribute model with a frozen-LLM rendering head."""

    def __init__(
        self,
        dbm: DeepBoltzmannMachine,
        adapter: BeliefAdapter,
        feature_spec: FeatureSpec,
        config: Mapping[str, Any],
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.device = select_device(device)
        self.config = dict(config)
        self.features = feature_spec

        self.dbm = dbm.to(self.device).eval()
        self.adapter = adapter.to(self.device).eval()
        for p in self.dbm.parameters():
            p.requires_grad_(False)
        for p in self.adapter.parameters():
            p.requires_grad_(False)

        bm_cfg = self.config.get("bm", {})
        adapter_cfg = self.config.get("adapter", {})
        self.mean_field_iters = int(bm_cfg.get("mean_field_iters", 10))
        self.use_top_layer = bool(adapter_cfg.get("use_top_layer", False))
        self.include_visible = bool(adapter_cfg.get("include_visible", False))

        self._llm = None
        self._tokenizer = None

    # -- construction -------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ) -> "AttributeModel":
        """Load a checkpoint directory, or a Hugging Face Hub repo id."""
        from safetensors.torch import load_file

        root = _resolve(repo_id_or_path)
        for name in (CONFIG_FILE, FEATURE_SPEC_FILE, DBM_FILE, ADAPTER_FILE):
            if not (root / name).exists():
                raise FileNotFoundError(f"{name} is missing from {root}")

        config = json.loads((root / CONFIG_FILE).read_text())
        feature_spec = FeatureSpec(json.loads((root / FEATURE_SPEC_FILE).read_text()))

        bm_cfg = config["bm"]
        dbm = DeepBoltzmannMachine(bm_cfg["layer_sizes"])
        dbm.load_state_dict(load_file(str(root / DBM_FILE)))

        if dbm.n_visible != feature_spec.n_visible:
            raise ValueError(
                f"DBM visible dim {dbm.n_visible} does not match feature_spec "
                f"n_visible {feature_spec.n_visible}"
            )

        a_cfg = config["adapter"]
        adapter = BeliefAdapter(
            input_dim=int(a_cfg["input_dim"]),
            hidden_layers=[int(h) for h in a_cfg["hidden_layers"]],
            num_soft_tokens=int(a_cfg["num_soft_tokens"]),
            embedding_dim=int(a_cfg["embedding_dim"]),
        )
        adapter.load_state_dict(load_file(str(root / ADAPTER_FILE)))

        return cls(dbm, adapter, feature_spec, config, device=device)

    # -- attribute schema ---------------------------------------------------

    def attributes(self) -> Dict[str, List[str]]:
        """Attribute group name -> allowed values."""
        return self.features.attributes()

    def encode(
        self, spec: Optional[Mapping[str, ValueSpec]] = None, **kwargs: ValueSpec
    ) -> torch.Tensor:
        """Build a binary visible vector from named attribute values.

        With no arguments this returns the default configuration: the modal
        value of every attribute group in the training data. Groups that are
        not named fall back to that modal value.
        """
        assignment: Dict[str, ValueSpec] = dict(spec or {})
        assignment.update(kwargs)
        return self.features.encode(assignment).to(self.device)

    def clamp(
        self,
        v: torch.Tensor,
        spec: Optional[Mapping[str, ValueSpec]] = None,
        **kwargs: ValueSpec,
    ) -> torch.Tensor:
        """Overwrite the named attribute groups of ``v`` and return a new vector.

        Only the visible units change here. The beliefs are re-equilibrated by
        mean-field inference the next time ``energy``, ``beliefs`` or
        ``generate`` is called on the returned vector, which is how the clamp
        propagates through the joint distribution.
        """
        assignment: Dict[str, ValueSpec] = dict(spec or {})
        assignment.update(kwargs)
        return self.features.clamp(v, assignment).to(self.device)

    def decode(self, v: torch.Tensor) -> Dict[str, Union[str, List[str], None]]:
        """Read a visible vector back as an attribute assignment."""
        return self.features.decode(v.detach().cpu())

    # -- energy and beliefs -------------------------------------------------

    def _as_batch(self, v: torch.Tensor) -> torch.Tensor:
        v = torch.as_tensor(v, dtype=torch.float32).to(self.device)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        if v.shape[-1] != self.dbm.n_visible:
            raise ValueError(
                f"Visible vector has {v.shape[-1]} units, expected {self.dbm.n_visible}"
            )
        return v

    def energy(self, v: torch.Tensor) -> torch.Tensor:
        """Mean-field energy score F~(v); lower means more coherent.

        Returns a scalar tensor for a single vector, or shape ``(batch,)``.
        """
        batched = v.dim() > 1 if isinstance(v, torch.Tensor) else True
        out = self.dbm.energy(self._as_batch(v), n_iter=self.mean_field_iters)
        return out if batched else out.squeeze(0)

    def beliefs(self, v: torch.Tensor) -> torch.Tensor:
        """Converged mean-field belief vector H."""
        batched = v.dim() > 1 if isinstance(v, torch.Tensor) else True
        out = self.dbm.beliefs(
            self._as_batch(v),
            n_iter=self.mean_field_iters,
            use_top_layer=self.use_top_layer,
        )
        return out if batched else out.squeeze(0)

    def soft_prompts(self, v: torch.Tensor) -> torch.Tensor:
        """Soft-prompt embeddings ``(batch, K, D)`` for a visible vector."""
        v_batch = self._as_batch(v)
        h = self.dbm.beliefs(
            v_batch, n_iter=self.mean_field_iters, use_top_layer=self.use_top_layer
        )
        if self.include_visible:
            h = torch.cat([v_batch, h], dim=1)
        with torch.no_grad():
            return self.adapter(h)

    # -- generation ---------------------------------------------------------

    @property
    def llm_id(self) -> str:
        return self.config.get("llm", {}).get("model_id", DEFAULT_LLM_ID)

    def _load_llm(self):
        """Load the frozen generator from the Hub on first use."""
        if self._llm is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.llm_id)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            llm = AutoModelForCausalLM.from_pretrained(self.llm_id, dtype=torch.float32)
            llm = llm.to(self.device).eval()
            for p in llm.parameters():
                p.requires_grad_(False)
            self._llm = llm
        return self._llm, self._tokenizer

    def _embed_layer(self):
        llm, _ = self._load_llm()
        if hasattr(llm, "transformer") and hasattr(llm.transformer, "wte"):
            return llm.transformer.wte
        if hasattr(llm, "model") and hasattr(llm.model, "embed_tokens"):
            return llm.model.embed_tokens
        raise ValueError(f"Cannot find the embedding layer of {type(llm).__name__}")

    def default_prompt(self) -> str:
        """The text prompt used when ``generate`` is called without one."""
        prompt_cfg = self.config.get("prompt", {})
        template = prompt_cfg.get("template")
        if template is None:
            raise ValueError(
                "This checkpoint has no prompt template; pass prompt=... to generate()."
            )
        return template.format(domain=prompt_cfg.get("domain", self.features.domain))

    def generate(
        self,
        v: torch.Tensor,
        prompt: Optional[str] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        seed: Optional[int] = None,
    ) -> str:
        """Render a visible attribute configuration as review text.

        The beliefs are re-equilibrated from ``v`` here, so a clamped vector
        takes effect at this point. The generator's weights are frozen.
        """
        v_batch = self._as_batch(v)
        if v_batch.shape[0] != 1:
            raise ValueError("generate() takes a single visible vector")

        llm, tokenizer = self._load_llm()
        if seed is not None:
            torch.manual_seed(seed)

        soft = self.soft_prompts(v_batch)
        embed = self._embed_layer()

        text = self.default_prompt() if prompt is None else prompt
        current_ids = tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        generated: List[int] = []
        eos_id = tokenizer.eos_token_id
        with torch.no_grad():
            for _ in range(max_new_tokens):
                text_embeds = embed(current_ids)
                inputs_embeds = torch.cat(
                    [soft.to(dtype=text_embeds.dtype), text_embeds], dim=1
                )
                logits = llm(inputs_embeds=inputs_embeds).logits
                next_logits = logits[:, -1, :].float() / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                if eos_id is not None and next_token.item() == eos_id:
                    break
                current_ids = torch.cat([current_ids, next_token], dim=1)
                generated.append(int(next_token.item()))

        return tokenizer.decode(generated, skip_special_tokens=True)
