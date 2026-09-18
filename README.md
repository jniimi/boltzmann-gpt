# boltzmann-gpt

An inference-only reference implementation of the architecture in *Energy-Based Attribute Models for Controllable Review Generation with Frozen LLMs*. A Deep Boltzmann Machine (DBM) is trained on binary one-hot attribute features of product reviews and captures the domain's attribute co-occurrence structure; its converged mean-field beliefs are projected by a small MLP adapter into 30 soft-prompt embeddings, which are prepended to a text prompt and rendered as review text by a frozen Qwen2.5-0.5B-Instruct. Because the attribute model is external to the language model and has an energy function, the same checkpoint supports three things a prompt alone does not: scoring the coherence of an arbitrary attribute configuration, clamping individual attributes and re-equilibrating the beliefs before decoding, and doing both without touching the generator's weights or activations.

- Paper: Junichiro Niimi (Meijo University), TMLR.
- OpenReview: <https://openreview.net/forum?id=pOIFHY4dOJ>
- The preliminary version of this work circulated as the "Boltzmann GPT" preprint, which is where the package name comes from.

> **Note.** This is a reference implementation released alongside the paper, not the code that produced the paper's results. The paper is the authoritative description of the method; see [Relation to the paper](#relation-to-the-paper).

### Citation:
If you use this package, please cite the following:
```bibtex
@article{niimi2026energybased,
    title = {Energy-Based Attribute Models for Controllable Review Generation with Frozen {LLM}s},
    author = {Junichiro Niimi},
    journal = {Transactions on Machine Learning Research},
    issn = {2835-8856},
    year = {2026},
    url = {https://openreview.net/forum?id=pOIFHY4dOJ}
}
```

## Install

```bash
uv add git+https://github.com/jniimi/boltzmann-gpt   # in an existing project
# or, from a clone:
uv sync
uv run python -c "import boltzmann_gpt; print(boltzmann_gpt.__version__)"
```

## Quickstart

```python
from boltzmann_gpt import AttributeModel

model = AttributeModel.from_pretrained("jniimi/boltzmann-gpt-smartphone")  # or a local checkpoint directory

# The attribute schema: group name -> allowed values.
model.attributes()
# {'price': ['Entry', 'Mid', 'High', 'Premium'], 'brand': ['apple', 'asus', ..., 'samsung', ...],
#  'rating': ['1', '2', '3', '4', '5'], 'topic': ['Battery', 'Screen', ...], ...}

# 1. Encode an attribute configuration. Groups you leave out take their modal
#    training value, so encode() with no arguments is the default configuration.
v = model.encode(brand="samsung", rating="5", price="Premium", topic=["Camera", "Battery"])

# 2. Coherence. energy() is the mean-field energy score; lower is more coherent.
print(model.energy(v))
print(model.energy(model.clamp(v, price="Entry")))

# 3. Clamp and generate. clamp() only rewrites the visible units; the beliefs
#    are re-equilibrated by mean-field inference inside generate().
print(model.generate(v, max_new_tokens=100, temperature=0.7, seed=0))
print(model.generate(model.clamp(v, rating="1"), seed=0))
```

`model.beliefs(v)` returns the converged belief vector `H` (the concatenated mean-field means) if you want to inspect or reuse it directly.

### The text prompt

Without `prompt=`, `generate()` uses the one-shot prompt layout from the paper: an instruction naming the domain, one example review, and the task's product name, price and average rating. The released checkpoints ship a synthetic, author-written example review and a generic product (both in `config.json`, under `prompt`), not real Amazon data. You can set the task fields and replace any part of the example:

```python
model.generate(v, product_name="Unlocked Android Smartphone, 256GB", price=449.99, seed=0)
model.generate(v, example={"review": "Your own example review."}, seed=0)  # merged over the default example
print(model.default_prompt(price=449.99))   # the prompt generate() would use
model.generate(v, prompt="...your full prompt...")  # used verbatim
```

The average rating stays at 3.0 by default: the training table had no average-rating field, so every training prompt showed 3.0. Numeric prices are shown as Python floats (`$25.0`, `$199.99`), as during training.

## Attribute schema

The DBM's visible layer is a flat binary vector, but you never address it by index. `feature_spec.json` records, for every attribute group, which visible units it owns, the label of each unit, whether the group is one-hot or multi-label, and its modal value in the training data.

- **one-hot** groups (brand, price tier, rating, purchase-frequency bin, price-range bin) take exactly one value. Setting one clears the rest of the group.
- **multi-label** groups (review topics, purchase-history flags, TF-IDF terms) take a list of values, possibly empty.

`encode()` and `clamp()` both accept a dict as the first argument or keyword arguments. An unknown group name or value raises a `ValueError` listing the valid options, so a typo is never silently encoded as "no attribute set". `decode(v)` reads a visible vector back as an assignment.

Attribute construction (brand vocabularies, price bins, topic lexicons, purchase-history statistics, TF-IDF terms) is described in the paper's preprocessing appendix.

## Checkpoints

Two checkpoints are released on the Hugging Face Hub, both from the seed-0 run reported in the paper:

| Hub repo | Domain | Visible units |
|---|---|---|
| [`jniimi/boltzmann-gpt-smartphone`](https://huggingface.co/jniimi/boltzmann-gpt-smartphone) | Smartphone | 160 |
| [`jniimi/boltzmann-gpt-beauty`](https://huggingface.co/jniimi/boltzmann-gpt-beauty) | Beauty | 170 |

A checkpoint directory holds `config.json`, `feature_spec.json`, `dbm.safetensors` and `adapter.safetensors`. Loading uses safetensors only — this package never reads or writes pickles. If the argument to `from_pretrained` is not an existing directory it is treated as a Hugging Face Hub repo id and fetched with `huggingface_hub`; the frozen generator named in `config.json` is downloaded from the Hub on the first call to `generate()`.

`scripts/export_checkpoint.py` converts the original pickled training artifacts into this format. It is for the author's own files, it warns when it unpickles, and it fails with an explicit message rather than guessing whenever the artifact's structure does not match what it expects. `scripts/write_model_card.py` renders the Hub model card from an exported checkpoint's `config.json` and `feature_spec.json`.

## Scope

This repository is a reference implementation of inference only.

- **Not included:** training code (layer-wise pretraining, PCD joint fine-tuning, adapter training), the baselines and ablations, the evaluation harness, and per-sample experiment outputs.
- **No data.** The underlying Amazon Reviews 2023 corpus is publicly available from its original source, and the filtering and feature construction are described in the paper's appendix in enough detail to reconstruct the tagged table.
- **No numbers are restated here.** The paper is the source for every empirical claim about the model.

### Relation to the paper

The package was reorganized from the original research code for release: the inference path was rewritten around a documented API and a safetensors checkpoint format, and the released weights were exported from the seed-0 training artifacts. The experiments reported in the paper were run with the original research code, not with this package.

As a consequence, small differences between this implementation and the description in the paper, as well as bugs, are possible. Where the two disagree, the paper is the specification and the discrepancy is a defect of this package. Generated text is also not expected to match the samples in the paper token for token, since sampling depends on library versions and hardware. If you find a discrepancy or a bug, please [open an issue](https://github.com/jniimi/boltzmann-gpt/issues).

## Responsible use

Everything this package generates is synthetic review text produced by a language model from an attribute configuration. It is not a real customer's opinion and it describes no real purchase. Posting such text as a genuine review is deceptive and is against the terms of every major review platform. If you publish or redistribute generations, disclose that they are model output.

## License

MIT. See [LICENSE](LICENSE).
