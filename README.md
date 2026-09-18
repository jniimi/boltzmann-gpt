# boltzmann-gpt

An inference-only reference implementation of the architecture in *Energy-Based Attribute Models for Controllable Review Generation with Frozen LLMs*. A Deep Boltzmann Machine (DBM) is trained on binary one-hot attribute features of product reviews and captures the domain's attribute co-occurrence structure; its converged mean-field beliefs are projected by a small MLP adapter into 16 soft-prompt embeddings, which are prepended to a text prompt and rendered as review text by a frozen Qwen2.5-0.5B-Instruct. Because the attribute model is external to the language model and has an energy function, the same checkpoint supports three things a prompt alone does not: scoring the coherence of an arbitrary attribute configuration, clamping individual attributes and re-equilibrating the beliefs before decoding, and doing both without touching the generator's weights or activations.

- Paper: Junichiro Niimi (Meijo University), TMLR.
- OpenReview: <https://openreview.net/forum?id=pOIFHY4dOJ>
- The preliminary version of this work circulated as the "Boltzmann GPT" preprint, which is where the package name comes from.

## Install

```bash
uv add boltzmann-gpt          # in an existing project
# or, from a clone:
uv sync
uv run python -c "import boltzmann_gpt; print(boltzmann_gpt.__version__)"
```

## Quickstart

```python
from boltzmann_gpt import AttributeModel

model = AttributeModel.from_pretrained("path/to/checkpoint")  # or a Hub repo id

# The attribute schema: group name -> allowed values.
model.attributes()
# {'brand': ['apple', 'samsung', ...], 'rating': ['1', ..., '5'], 'price': [...], ...}

# 1. Encode an attribute configuration. Groups you leave out take their modal
#    training value, so encode() with no arguments is the default configuration.
v = model.encode(brand="apple", rating="5", price="Premium")

# 2. Coherence. energy() is the mean-field energy score; lower is more coherent.
print(model.energy(v))
print(model.energy(model.clamp(v, price="Entry")))   # a premium brand at entry price

# 3. Clamp and generate. clamp() only rewrites the visible units; the beliefs
#    are re-equilibrated by mean-field inference inside generate().
print(model.generate(v, max_new_tokens=100, temperature=0.7, seed=0))
print(model.generate(model.clamp(v, rating="1"), seed=0))
```

`model.beliefs(v)` returns the converged belief vector `H` (the concatenated mean-field means) if you want to inspect or reuse it directly.

## Attribute schema

The DBM's visible layer is a flat binary vector, but you never address it by index. `feature_spec.json` records, for every attribute group, which visible units it owns, the label of each unit, whether the group is one-hot or multi-label, and its modal value in the training data.

- **one-hot** groups (brand, price tier, rating, purchase-frequency bin, price-range bin) take exactly one value. Setting one clears the rest of the group.
- **multi-label** groups (review topics, purchase-history flags, TF-IDF terms) take a list of values, possibly empty.

`encode()` and `clamp()` both accept a dict as the first argument or keyword arguments. An unknown group name or value raises a `ValueError` listing the valid options, so a typo is never silently encoded as "no attribute set". `decode(v)` reads a visible vector back as an assignment.

Attribute construction (brand vocabularies, price bins, topic lexicons, purchase-history statistics, TF-IDF terms) is described in the paper's preprocessing appendix.

## Checkpoints

A checkpoint directory holds `config.json`, `feature_spec.json`, `dbm.safetensors` and `adapter.safetensors`. Loading uses safetensors only — this package never reads or writes pickles. If the argument to `from_pretrained` is not an existing directory it is treated as a Hugging Face Hub repo id and fetched with `huggingface_hub`; the frozen generator named in `config.json` is downloaded from the Hub on the first call to `generate()`.

`scripts/export_checkpoint.py` converts the original pickled training artifacts into this format. It is for the author's own files, it warns when it unpickles, and it fails with an explicit message rather than guessing whenever the artifact's structure does not match what it expects.

## Scope

This repository is a reference implementation of inference only.

- **Not included:** training code (layer-wise pretraining, PCD joint fine-tuning, adapter training), the baselines and ablations, the evaluation harness, and per-sample experiment outputs.
- **No data.** The underlying Amazon Reviews 2023 corpus is publicly available from its original source, and the filtering and feature construction are described in the paper's appendix in enough detail to reconstruct the tagged table.
- **No numbers are restated here.** The paper is the source for every empirical claim about the model.

The DBM is a model of how attributes co-occur in the training domain. It is not a world model and it does not represent causal structure: clamping fixes visible units in the learnt distribution and re-runs mean-field inference, so the resulting shifts reflect model-internal distributional consistency, not identified real-world effects.

## Responsible use

Everything this package generates is synthetic review text produced by a language model from an attribute configuration. It is not a real customer's opinion and it describes no real purchase. Posting such text as a genuine review is deceptive and is against the terms of every major review platform. If you publish or redistribute generations, disclose that they are model output.

## License

MIT. See [LICENSE](LICENSE).
