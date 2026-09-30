#!/usr/bin/env python3
"""Write the Hugging Face model card (README.md) into an exported checkpoint.

The card is rendered from the checkpoint's own config.json and
feature_spec.json, so the architecture and attribute schema it lists are the
ones actually shipped.

Usage:
    uv run python scripts/write_model_card.py checkpoints/smartphone
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_URL = "https://github.com/jniimi/boltzmann-gpt"
OPENREVIEW_URL = "https://openreview.net/forum?id=pOIFHY4dOJ"
TITLE = "Energy-Based Attribute Models for Controllable Review Generation with Frozen LLMs"

SOURCE = {
    "smartphone": "the *Cell Phones and Accessories* category (smartphones only; accessories excluded)",
    "beauty": "the *All Beauty* category",
}

EXAMPLE = {
    "smartphone": ('brand="samsung", rating="5", price="Premium"', 'price="Entry"'),
    "beauty": ('brand="neutrogena", rating="5", topic=["Skincare"]', 'topic=["Skincare", "Allergy"]'),
}

CITATION = """@article{niimi2026energybased,
    title = {Energy-Based Attribute Models for Controllable Review Generation with Frozen {LLM}s},
    author = {Junichiro Niimi},
    journal = {Transactions on Machine Learning Research},
    issn = {2835-8856},
    year = {2026},
    url = {https://openreview.net/forum?id=pOIFHY4dOJ}
}"""


def schema_table(spec: dict) -> str:
    rows = ["| Group | Type | Units | Values | Default (modal) |", "|---|---|---|---|---|"]
    for name, g in spec["groups"].items():
        values = g["values"]
        if len(values) > 40:
            shown = f"{len(values)} TF-IDF terms (see `feature_spec.json`)"
        else:
            shown = ", ".join(f"`{v}`" for v in values)
        default = g["default"]
        if isinstance(default, list):
            default = ", ".join(f"`{v}`" for v in default) if default else "(none)"
        else:
            default = f"`{default}`"
        kind = "one-hot" if g["type"] == "one_hot" else "multi-label"
        rows.append(f"| `{name}` | {kind} | {len(values)} | {shown} | {default} |")
    return "\n".join(rows)


def render(root: Path) -> str:
    config = json.loads((root / "config.json").read_text())
    spec = json.loads((root / "feature_spec.json").read_text())
    domain = spec["domain"]
    layers = config["bm"]["layer_sizes"]
    a = config["adapter"]
    llm = config["llm"]["model_id"]
    prompt = config["prompt"]
    enc, clamp = EXAMPLE[domain]
    repo_id = f"jniimi/boltzmann-gpt-{domain}"
    label = domain.capitalize()

    return f"""---
license: mit
library_name: boltzmann-gpt
base_model: {llm}
pipeline_tag: text-generation
language:
- en
tags:
- energy-based-model
- deep-boltzmann-machine
- soft-prompt
- controllable-generation
- synthetic-reviews
---

# boltzmann-gpt-{domain}

The {label} checkpoint of *{TITLE}* (Junichiro Niimi, TMLR 2026). A Deep Boltzmann Machine over binary review attributes, plus the MLP adapter that turns its mean-field beliefs into soft prompts for a frozen `{llm}`. Load it with the [`boltzmann-gpt`]({REPO_URL}) package.

- Paper: [OpenReview]({OPENREVIEW_URL})
- Code: [{REPO_URL}]({REPO_URL})

**Everything this model generates is synthetic review text.** It is not a real customer's opinion and it describes no real purchase. Do not post generations as genuine reviews; if you publish or redistribute them, state clearly that they are model output.

## Usage

```bash
uv add git+{REPO_URL}
```

```python
from boltzmann_gpt import AttributeModel

model = AttributeModel.from_pretrained("{repo_id}")
model.attributes()                         # group -> allowed values

v = model.encode({enc})
print(model.energy(v))                     # mean-field energy score, lower = more coherent
print(model.energy(model.clamp(v, {clamp})))
print(model.generate(v, seed=0))           # renders via the frozen generator
print(model.generate(model.clamp(v, rating="1"), seed=0))
```

`generate()` downloads `{llm}` from the Hub on first use. Without a `prompt=` argument it uses the paper's one-shot prompt layout (paper, Appendix B): an instruction naming the domain ("{prompt["domain"]}"), one example review, and the task fields. The example review is synthetic and author-written, and the product is generic ("{prompt["product"]["product_name"]}", ${prompt["product"]["price"]}); both live in `config.json`. Override them with `generate(v, product_name=..., price=..., example={{"product_name": ..., "price": ..., "review": ...}})`, where a partial `example` dict is merged over the default. The average rating is 3.0 for both example and product, because it was constant at 3.0 in every training prompt. Pass `prompt=` to supply the full text prompt yourself; `model.default_prompt(...)` returns the prompt `generate()` would use.

## What is in this repository

| File | Contents |
|---|---|
| `dbm.safetensors` | DBM weights and biases, layers `{layers}` |
| `adapter.safetensors` | adapter MLP `{a["input_dim"]} → {" → ".join(str(h) for h in a["hidden_layers"])} → {a["num_soft_tokens"]}×{a["embedding_dim"]}` (ReLU) and the LayerNorm over the soft-prompt embeddings |
| `config.json` | architecture, mean-field iterations ({config["bm"]["mean_field_iters"]}), generator id, default prompt |
| `feature_spec.json` | the {spec["n_visible"]} visible units: column names, attribute groups, modal defaults |

The adapter reads the concatenation of all converged hidden-layer means ({a["input_dim"]} dimensions) and emits {a["num_soft_tokens"]} soft-prompt embeddings, which are prepended to the text prompt. The generator's weights are never modified. This is the seed-0 run reported in the paper; the other seeds are not released.

## Attribute schema

{schema_table(spec)}

Groups left out of `encode()` take their modal training value. The modal defaults are marginal modes taken group by group, so the default configuration as a whole is not a typical review (the topic and TF-IDF groups are empty, for instance) and its energy is higher than that of a typical training vector. For meaningful comparisons, start from a fully specified configuration.

## Training data

Built from Amazon Reviews 2023, {SOURCE[domain]}: verified purchases, English only, one review per user; 52,952 / 1,024 / 1,024 train / validation / test reviews. Attribute construction is described in the paper's preprocessing appendix. No review text or user data is included here.

## Limitations

The DBM models how attributes co-occur in this one training domain. It is not a causal model: clamping an attribute fixes visible units in the learnt distribution and re-runs mean-field inference, so the resulting changes reflect model-internal consistency, not real-world effects. Generations inherit the biases of the review corpus and of the generator, and a small generator often produces repetitive or ungrammatical text.

## License

The DBM and adapter weights in this repository are released under the MIT License. The frozen generator, `{llm}`, is not included and is distributed by its authors under the Apache License 2.0.

## Citation

```bibtex
{CITATION}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    out = args.checkpoint / "README.md"
    out.write_text(render(args.checkpoint))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
