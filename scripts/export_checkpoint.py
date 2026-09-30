#!/usr/bin/env python3
"""Convert a pickled training artifact into the public checkpoint format.

Outputs ``dbm.safetensors``, ``adapter.safetensors``, ``config.json`` and
``feature_spec.json``. The published package never touches pickle; this script
does, because it reads the author's own training output, and it warns about it.

The training artifacts are:

  * the DBM pickle, a pickled ``DeepBoltzmannMachine`` module with
    ``layer_sizes``, ``weights`` (ParameterList) and ``biases`` (ParameterList);
  * the adapter pickle, either a pickled ``BeliefMachine`` module or a
    checkpoint dict ``{'model': BeliefMachine, 'optimizer_state', ...}``. The
    relevant parts are ``adapter`` (nn.Sequential) and ``adapter_norm``
    (nn.LayerNorm), plus ``num_soft_tokens``, ``embed_dim``, ``use_top_layer``
    and ``include_visible``.

Unpickling needs the original classes importable, so pass ``--gbm-repo`` with
the path to the research repository.

Usage:
    uv run python3 scripts/export_checkpoint.py \\
        --dbm  /path/to/<run>_FT.pkl \\
        --adapter /path/to/<run>_adapter_val-best.pkl \\
        --parquet /path/to/<dataset>.parquet \\
        --domain smartphone \\
        --gbm-repo /path/to/GBM \\
        --out ./checkpoints/smartphone
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import torch


def fail(message: str) -> "NoReturn":  # type: ignore[valid-type]
    """Stop with an explicit message rather than guessing."""
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Attribute groups
# ---------------------------------------------------------------------------
# Tag column prefix -> (group name, one-hot or multi-label). The prefixes come
# from the preprocessing described in the paper's appendix. Order matters:
# longer prefixes are matched first.
GROUP_RULES: List[tuple] = [
    ("Tag_Brand_", "brand", "one_hot"),
    ("Tag_Price_", "price", "one_hot"),
    ("Tag_Rating_", "rating", "one_hot"),
    ("Tag_Topic_", "topic", "multi_label"),
    ("Tag_Purchase_Freq_", "purchase_freq", "one_hot"),
    ("Tag_PriceRange_", "price_range", "one_hot"),
    ("Tag_Purchase_", "purchase_flags", "multi_label"),
    ("Tag_Auto_", "auto_terms", "multi_label"),
]


def assign_group(column: str):
    for prefix, name, kind in GROUP_RULES:
        if column.startswith(prefix):
            return name, kind, column[len(prefix):]
    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_pickle(path: Path) -> Any:
    warnings.warn(
        f"Unpickling {path}: pickle executes arbitrary code. Only run this on "
        "your own training artifacts.",
        stacklevel=2,
    )
    import pickle

    with path.open("rb") as fp:
        return pickle.load(fp)


def extract_dbm(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict) and "model" in obj:
        obj = obj["model"]
    if not hasattr(obj, "layer_sizes"):
        fail(
            f"The DBM pickle holds a {type(obj).__name__} with no 'layer_sizes'. "
            "Expected a DeepBoltzmannMachine. Cannot determine the architecture."
        )
    if not (hasattr(obj, "weights") and hasattr(obj, "biases")):
        fail(
            f"{type(obj).__name__} has no 'weights'/'biases' ParameterList; the "
            "checkpoint layout cannot be determined."
        )

    layer_sizes = [int(s) for s in obj.layer_sizes]
    tensors = OrderedDict()
    for i, w in enumerate(obj.weights):
        tensors[f"weights.{i}"] = w.detach().cpu().contiguous()
    for i, b in enumerate(obj.biases):
        tensors[f"biases.{i}"] = b.detach().cpu().contiguous()

    expected_w = len(layer_sizes) - 1
    if len(obj.weights) != expected_w or len(obj.biases) != len(layer_sizes):
        fail(
            f"layer_sizes {layer_sizes} implies {expected_w} weight matrices and "
            f"{len(layer_sizes)} bias vectors, but the pickle has "
            f"{len(obj.weights)} and {len(obj.biases)}."
        )
    for i in range(expected_w):
        shape = tuple(tensors[f"weights.{i}"].shape)
        if shape != (layer_sizes[i], layer_sizes[i + 1]):
            fail(
                f"weights.{i} has shape {shape}, expected "
                f"{(layer_sizes[i], layer_sizes[i + 1])}."
            )

    return {"layer_sizes": layer_sizes, "tensors": tensors}


def extract_adapter(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        if "model" not in obj:
            fail(
                "The adapter pickle is a dict without a 'model' key "
                f"(keys: {sorted(obj)}); cannot locate the BeliefMachine."
            )
        obj = obj["model"]

    if not hasattr(obj, "adapter"):
        fail(
            f"The adapter pickle holds a {type(obj).__name__} with no 'adapter' "
            "attribute. Expected a BeliefMachine."
        )

    mlp = obj.adapter
    linears = [m for m in mlp if isinstance(m, torch.nn.Linear)]
    if not linears:
        fail("The adapter MLP contains no Linear layers; layout undetermined.")

    tensors = OrderedDict()
    for i, module in enumerate(mlp):
        if isinstance(module, torch.nn.Linear):
            tensors[f"mlp.{i}.weight"] = module.weight.detach().cpu().contiguous()
            tensors[f"mlp.{i}.bias"] = module.bias.detach().cpu().contiguous()

    if not hasattr(obj, "adapter_norm"):
        fail(
            "The BeliefMachine has no 'adapter_norm'; this exporter only supports "
            "checkpoints whose soft prompts pass through a LayerNorm."
        )
    tensors["norm.weight"] = obj.adapter_norm.weight.detach().cpu().contiguous()
    tensors["norm.bias"] = obj.adapter_norm.bias.detach().cpu().contiguous()

    num_soft_tokens = getattr(obj, "num_soft_tokens", None)
    embed_dim = getattr(obj, "embed_dim", None)
    if num_soft_tokens is None or embed_dim is None:
        fail("The BeliefMachine lacks 'num_soft_tokens' and/or 'embed_dim'.")
    if linears[-1].out_features != num_soft_tokens * embed_dim:
        fail(
            f"Final adapter Linear has {linears[-1].out_features} outputs but "
            f"num_soft_tokens * embed_dim = {num_soft_tokens * embed_dim}."
        )

    return {
        "input_dim": int(linears[0].in_features),
        "hidden_layers": [int(layer.out_features) for layer in linears[:-1]],
        "num_soft_tokens": int(num_soft_tokens),
        "embedding_dim": int(embed_dim),
        "use_top_layer": bool(getattr(obj, "use_top_layer", False)),
        "include_visible": bool(getattr(obj, "include_visible", False)),
        "tensors": tensors,
    }


# ---------------------------------------------------------------------------
# Feature spec
# ---------------------------------------------------------------------------
def build_feature_spec(
    feature_names: List[str], domain: str, modal: Dict[str, Any] | None
) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, Any]] = OrderedDict()
    for index, column in enumerate(feature_names):
        assigned = assign_group(column)
        if assigned is None:
            fail(
                f"Column {column!r} matches no known Tag_ prefix. Add a rule to "
                "GROUP_RULES rather than letting it be dropped silently."
            )
        name, kind, value = assigned
        group = groups.setdefault(
            name, {"type": kind, "indices": [], "values": [], "default": None}
        )
        group["indices"].append(index)
        group["values"].append(value)

    # TODO: the modal value of each group must be computed from the training
    # parquet (the same table the DBM was trained on): for a one-hot group the
    # value whose column has the highest mean, for a multi-label group every
    # value whose column mean exceeds 0.5. Pass --parquet to fill this in;
    # without it the defaults stay null and encode() with no arguments fails.
    if modal is not None:
        for name, group in groups.items():
            if name not in modal:
                fail(f"No modal value computed for attribute {name!r}.")
            group["default"] = modal[name]

    return {
        "domain": domain,
        "n_visible": len(feature_names),
        "feature_names": feature_names,
        "groups": groups,
    }


def compute_modal(parquet: Path, feature_names: List[str]) -> Dict[str, Any]:
    """Modal value per attribute group, from the training table."""
    import pandas as pd

    df = pd.read_parquet(parquet, columns=feature_names)
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        fail(f"{parquet} is missing feature columns: {missing[:10]}")

    means = df.mean()
    grouped: Dict[str, List[tuple]] = OrderedDict()
    for column in feature_names:
        assigned = assign_group(column)
        name, kind, value = assigned
        grouped.setdefault(name, []).append((value, float(means[column]), kind))

    modal: Dict[str, Any] = {}
    for name, entries in grouped.items():
        kind = entries[0][2]
        if kind == "one_hot":
            modal[name] = max(entries, key=lambda e: e[1])[0]
        else:
            modal[name] = [value for value, mean, _ in entries if mean > 0.5]
    return modal


def read_feature_names(parquet: Path) -> List[str]:
    import pandas as pd

    columns = list(pd.read_parquet(parquet).columns)
    names = [c for c in columns if c.startswith("Tag_")]
    if not names:
        fail(f"{parquet} has no Tag_ columns.")
    return names


# ---------------------------------------------------------------------------
# Default prompt
# ---------------------------------------------------------------------------
# The one-shot prompt of UnifiedDataset.create_prompt (use_example=True) in the
# research repo, with the example and task fields as placeholders. The layout
# is byte-identical: SYSTEM, a blank line, EXAMPLE, a blank line, TASK.
PROMPT_TEMPLATE = (
    "## Instruction\n"
    "You are a helpful assistant to generate online review about {domain} in Amazon.\n"
    "Consider product name, price in dollars, and average rating.\n"
    "\n"
    "## Example:\n"
    "Product name: {example_product_name}\n"
    "Price: ${example_price}\n"
    "Average rating: {example_average_rating}\n"
    "Review: {example_review}\n"
    "\n"
    "## Task:\n"
    "Product name: {product_name}\n"
    "Price: ${price}\n"
    "Average rating: {average_rating}\n"
    "Review:"
)

# The domain string the prompts were trained with.
PROMPT_DOMAIN = {"smartphone": "smartphone", "beauty": "beauty products"}

# Author-written synthetic example reviews and generic default products; no
# real product or review is shipped. The average rating is 3.0 because the
# training table had no average-rating column, so every training prompt showed
# create_prompt's fallback of 3.0 for both example and target.
PROMPT_FIELDS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "smartphone": {
        "example": {
            "product_name": "Unlocked Android Smartphone, 6.5-inch Display, 128GB",
            "price": 199.99,
            "average_rating": 3.0,
            "review": (
                "Switched to this phone from an older model and it has been solid "
                "so far. Battery easily lasts a full day, the screen is bright, and "
                "the camera does well in daylight. It gets a little warm when gaming "
                "and the speaker is only average, but for the price I am happy with it."
            ),
        },
        "product": {
            "product_name": "Unlocked Android Smartphone",
            "price": 299.99,
            "average_rating": 3.0,
        },
    },
    "beauty": {
        "example": {
            "product_name": "Daily Hydrating Face Moisturizer for Sensitive Skin, 1.7 oz",
            "price": 14.99,
            "average_rating": 3.0,
            "review": (
                "I have dry, sensitive skin and this moisturizer absorbs quickly "
                "without feeling greasy. No irritation after two weeks of daily use, "
                "and a little goes a long way. The scent is light and fades fast. Not "
                "a miracle product, but a good everyday cream for the price."
            ),
        },
        "product": {
            "product_name": "Moisturizing Face Cream",
            "price": 19.99,
            "average_rating": 3.0,
        },
    },
}


def build_prompt_config(domain: str, prompt_domain: str) -> Dict[str, Any]:
    """The ``prompt`` section of config.json."""
    if domain not in PROMPT_FIELDS:
        fail(
            f"No default prompt fields for domain {domain!r}; add an entry to "
            "PROMPT_FIELDS."
        )
    fields = PROMPT_FIELDS[domain]
    return {
        "domain": prompt_domain,
        "template": PROMPT_TEMPLATE,
        "example": dict(fields["example"]),
        "product": dict(fields["product"]),
    }


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dbm", required=True, type=Path, help="pickled DBM (…_FT.pkl)")
    parser.add_argument("--adapter", required=True, type=Path, help="pickled BeliefMachine checkpoint")
    parser.add_argument("--out", required=True, type=Path, help="output checkpoint directory")
    parser.add_argument("--domain", required=True, help="e.g. smartphone, beauty")
    parser.add_argument(
        "--prompt-domain",
        help="domain string in the prompt (default: 'beauty products' for beauty, else --domain)",
    )
    parser.add_argument("--parquet", type=Path, help="training table, for feature names and modal defaults")
    parser.add_argument("--feature-names", type=Path, help="JSON list of Tag_ columns, if no parquet is available")
    parser.add_argument("--gbm-repo", type=Path, help="path to the research repo, added to sys.path for unpickling")
    parser.add_argument("--llm-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--mean-field-iters", type=int, default=10)
    args = parser.parse_args()

    if args.gbm_repo:
        sys.path.insert(0, str(args.gbm_repo.expanduser().resolve()))

    from safetensors.torch import save_file

    dbm = extract_dbm(load_pickle(args.dbm))
    adapter = extract_adapter(load_pickle(args.adapter))

    if args.parquet:
        feature_names = read_feature_names(args.parquet)
        modal = compute_modal(args.parquet, feature_names)
    elif args.feature_names:
        feature_names = json.loads(args.feature_names.read_text())
        modal = None
        print(
            "WARNING: no --parquet given, so the modal defaults are null and "
            "encode() with no arguments will fail.",
            file=sys.stderr,
        )
    else:
        fail("Pass --parquet (preferred) or --feature-names; the visible layout "
             "cannot be recovered from the pickles alone.")

    if len(feature_names) != dbm["layer_sizes"][0]:
        fail(
            f"{len(feature_names)} Tag_ columns but the DBM's visible layer has "
            f"{dbm['layer_sizes'][0]} units."
        )

    expected_input = sum(dbm["layer_sizes"][1:])
    if adapter["use_top_layer"]:
        expected_input = dbm["layer_sizes"][-1]
    if adapter["include_visible"]:
        expected_input += dbm["layer_sizes"][0]
    if adapter["input_dim"] != expected_input:
        fail(
            f"The adapter's first Linear takes {adapter['input_dim']} inputs but "
            f"the DBM belief vector is {expected_input}-dimensional "
            f"(use_top_layer={adapter['use_top_layer']}, "
            f"include_visible={adapter['include_visible']})."
        )

    spec = build_feature_spec(feature_names, args.domain, modal)
    prompt_domain = args.prompt_domain or PROMPT_DOMAIN.get(args.domain, args.domain)

    config = {
        "architecture": "boltzmann-gpt",
        "bm": {
            "type": "dbm",
            "layer_sizes": dbm["layer_sizes"],
            "mean_field_iters": args.mean_field_iters,
        },
        "adapter": {
            "input_dim": adapter["input_dim"],
            "hidden_layers": adapter["hidden_layers"],
            "num_soft_tokens": adapter["num_soft_tokens"],
            "embedding_dim": adapter["embedding_dim"],
            "use_top_layer": adapter["use_top_layer"],
            "include_visible": adapter["include_visible"],
        },
        "llm": {"model_id": args.llm_id},
        "prompt": build_prompt_config(args.domain, prompt_domain),
    }

    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    save_file(dbm["tensors"], str(out / "dbm.safetensors"))
    save_file(adapter["tensors"], str(out / "adapter.safetensors"))
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "feature_spec.json").write_text(json.dumps(spec, indent=2) + "\n")

    print(f"Wrote checkpoint to {out}")
    print(f"  DBM layers      : {dbm['layer_sizes']}")
    print(f"  adapter         : {adapter['input_dim']} -> {adapter['hidden_layers']} -> "
          f"{adapter['num_soft_tokens']}x{adapter['embedding_dim']}")
    print(f"  visible units   : {len(feature_names)} in {len(spec['groups'])} groups")


if __name__ == "__main__":
    main()
