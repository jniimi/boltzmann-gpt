"""Tests against a locally generated fake checkpoint.

No network access and no real weights: the checkpoint is random. Anything that
needs the frozen LLM is out of scope here.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from boltzmann_gpt import AttributeModel

N_VISIBLE = 12
LAYER_SIZES = [N_VISIBLE, 8, 6]
NUM_SOFT_TOKENS = 4
EMBED_DIM = 10
HIDDEN_LAYERS = [16]

FEATURE_SPEC = {
    "domain": "test",
    "n_visible": N_VISIBLE,
    "feature_names": (
        ["Tag_Brand_apple", "Tag_Brand_samsung", "Tag_Brand_others"]
        + [f"Tag_Rating_{i}" for i in range(1, 6)]
        + ["Tag_Price_Entry", "Tag_Price_Premium"]
        + ["Tag_Topic_battery", "Tag_Topic_screen"]
    ),
    "groups": {
        "brand": {
            "type": "one_hot",
            "indices": [0, 1, 2],
            "values": ["apple", "samsung", "others"],
            "default": "others",
        },
        "rating": {
            "type": "one_hot",
            "indices": [3, 4, 5, 6, 7],
            "values": ["1", "2", "3", "4", "5"],
            "default": "5",
        },
        "price": {
            "type": "one_hot",
            "indices": [8, 9],
            "values": ["Entry", "Premium"],
            "default": "Entry",
        },
        "topic": {
            "type": "multi_label",
            "indices": [10, 11],
            "values": ["battery", "screen"],
            "default": ["battery"],
        },
    },
}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("fake-checkpoint")
    torch.manual_seed(0)

    dbm_tensors = {}
    for i in range(len(LAYER_SIZES) - 1):
        dbm_tensors[f"weights.{i}"] = torch.randn(LAYER_SIZES[i], LAYER_SIZES[i + 1]) * 0.1
    for i, size in enumerate(LAYER_SIZES):
        dbm_tensors[f"biases.{i}"] = torch.randn(size) * 0.1
    save_file(dbm_tensors, str(root / "dbm.safetensors"))

    input_dim = sum(LAYER_SIZES[1:])
    dims = [input_dim] + HIDDEN_LAYERS + [NUM_SOFT_TOKENS * EMBED_DIM]
    adapter_tensors = {}
    # Linear modules sit at even positions of the Sequential (Linear, ReLU, ...)
    for i in range(len(dims) - 1):
        adapter_tensors[f"mlp.{2 * i}.weight"] = torch.randn(dims[i + 1], dims[i]) * 0.1
        adapter_tensors[f"mlp.{2 * i}.bias"] = torch.zeros(dims[i + 1])
    adapter_tensors["norm.weight"] = torch.ones(EMBED_DIM)
    adapter_tensors["norm.bias"] = torch.zeros(EMBED_DIM)
    save_file(adapter_tensors, str(root / "adapter.safetensors"))

    config = {
        "architecture": "boltzmann-gpt",
        "bm": {"type": "dbm", "layer_sizes": LAYER_SIZES, "mean_field_iters": 5},
        "adapter": {
            "input_dim": input_dim,
            "hidden_layers": HIDDEN_LAYERS,
            "num_soft_tokens": NUM_SOFT_TOKENS,
            "embedding_dim": EMBED_DIM,
            "use_top_layer": False,
            "include_visible": False,
        },
        "llm": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct"},
        "prompt": {"domain": "test", "template": "## Instruction\nAbout {domain}.\nReview:"},
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "feature_spec.json").write_text(json.dumps(FEATURE_SPEC))
    return root


@pytest.fixture(scope="module")
def model(checkpoint):
    return AttributeModel.from_pretrained(checkpoint, device="cpu")


def test_attributes_schema(model):
    attrs = model.attributes()
    assert set(attrs) == {"brand", "rating", "price", "topic"}
    assert attrs["rating"] == ["1", "2", "3", "4", "5"]


def test_default_encode_uses_modal_values(model):
    v = model.encode()
    assert v.shape == (N_VISIBLE,)
    assert set(v.unique().tolist()) <= {0.0, 1.0}
    assert model.decode(v) == {
        "brand": "others",
        "rating": "5",
        "price": "Entry",
        "topic": ["battery"],
    }


def test_encode_overrides_and_falls_back(model):
    v = model.encode(brand="apple", rating="1")
    decoded = model.decode(v)
    assert decoded["brand"] == "apple"
    assert decoded["rating"] == "1"
    # unspecified groups keep their modal value
    assert decoded["price"] == "Entry"
    assert decoded["topic"] == ["battery"]


def test_encode_accepts_positional_dict(model):
    assert torch.equal(model.encode({"brand": "apple"}), model.encode(brand="apple"))


def test_encode_multi_label_group(model):
    v = model.encode(topic=["battery", "screen"])
    assert model.decode(v)["topic"] == ["battery", "screen"]
    assert model.decode(model.encode(topic=[]))["topic"] == []


def test_clamp_resets_the_group_one_hot(model):
    v = model.encode(rating="5")
    clamped = model.clamp(v, rating="1")
    assert model.decode(clamped)["rating"] == "1"
    # the original vector is untouched
    assert model.decode(v)["rating"] == "5"
    # other groups are untouched
    assert model.decode(clamped)["brand"] == model.decode(v)["brand"]


def test_energy_shapes(model):
    v = model.encode()
    assert model.energy(v).shape == ()
    batch = torch.stack([model.encode(rating="1"), model.encode(rating="5")])
    assert model.energy(batch).shape == (2,)


def test_energy_is_deterministic(model):
    v = model.encode()
    assert torch.allclose(model.energy(v), model.energy(v))


def test_beliefs_shape_and_range(model):
    h = model.beliefs(model.encode())
    assert h.shape == (sum(LAYER_SIZES[1:]),)
    assert float(h.min()) >= 0.0 and float(h.max()) <= 1.0


def test_soft_prompt_shape(model):
    soft = model.soft_prompts(model.encode())
    assert soft.shape == (1, NUM_SOFT_TOKENS, EMBED_DIM)


def test_unknown_group_raises(model):
    with pytest.raises(ValueError, match="Unknown attribute"):
        model.encode(colour="red")


def test_unknown_value_raises(model):
    with pytest.raises(ValueError, match="Unknown value"):
        model.encode(rating="6")


def test_one_hot_rejects_multiple_values(model):
    with pytest.raises(ValueError, match="one-hot"):
        model.encode(rating=["1", "2"])


def test_wrong_visible_dim_raises(model):
    with pytest.raises(ValueError, match="expected"):
        model.energy(torch.zeros(N_VISIBLE + 3))


def test_default_prompt_substitutes_domain(model):
    assert "About test." in model.default_prompt()


# -- one-shot default prompt ----------------------------------------------

ONE_SHOT_TEMPLATE = (
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
EXAMPLE_FIELDS = {
    "product_name": "Example Phone, 128GB",
    "price": 199.99,
    "average_rating": 3.0,
    "review": "Works fine. Battery lasts a day.",
}
PRODUCT_FIELDS = {"product_name": "Target Phone", "price": 25.0, "average_rating": 3.0}


def reference_create_prompt(domain, example_row, target_row):
    """Reimplementation of the research repo's UnifiedDataset.create_prompt
    with use_example=True."""
    tgt_title = str(target_row.get("product_name", "Unknown Product"))
    tgt_price = target_row.get("price", 0)
    tgt_rating = target_row.get("overall_average", 3.0)
    SYSTEM = "## Instruction\n"
    SYSTEM += f"You are a helpful assistant to generate online review about {domain} in Amazon.\n"
    SYSTEM += "Consider product name, price in dollars, and average rating.\n"
    EXAMPLE = "## Example:\n"
    EXAMPLE += f"Product name: {example_row.get('product_name', 'Unknown Product')}\n"
    EXAMPLE += f"Price: ${example_row.get('price', 0)}\n"
    EXAMPLE += f"Average rating: {example_row.get('overall_average', 3.0)}\n"
    EXAMPLE += f"Review: {example_row.get('text', '')}\n"
    TASK = "## Task:\n"
    TASK += f"Product name: {tgt_title}\n"
    TASK += f"Price: ${tgt_price}\n"
    TASK += f"Average rating: {tgt_rating}\n"
    TASK += "Review:"
    return f"{SYSTEM}\n{EXAMPLE}\n{TASK}"


def _with_prompt(model, prompt_cfg):
    return AttributeModel(
        model.dbm, model.adapter, model.features,
        {**model.config, "prompt": prompt_cfg}, device="cpu",
    )


@pytest.fixture(scope="module")
def one_shot(model):
    return _with_prompt(model, {
        "domain": "beauty products",
        "template": ONE_SHOT_TEMPLATE,
        "example": EXAMPLE_FIELDS,
        "product": PRODUCT_FIELDS,
    })


def test_one_shot_matches_create_prompt(one_shot):
    # training rows had no overall_average column, so the rating fell back to 3.0
    example_row = {"product_name": EXAMPLE_FIELDS["product_name"],
                   "price": EXAMPLE_FIELDS["price"], "text": EXAMPLE_FIELDS["review"]}
    target_row = {"product_name": PRODUCT_FIELDS["product_name"],
                  "price": PRODUCT_FIELDS["price"]}
    expected = reference_create_prompt("beauty products", example_row, target_row)
    assert one_shot.default_prompt() == expected


def test_one_shot_price_formatting(one_shot):
    prompt = one_shot.default_prompt()
    assert "Price: $199.99\n" in prompt
    assert "Price: $25.0\n" in prompt
    # integers are shown as floats, as in the training table
    assert "Price: $25.0\n" in one_shot.default_prompt(price=25)
    assert "Average rating: 4.0\n" in one_shot.default_prompt(average_rating=4)
    assert "Price: $1999.99\n" in one_shot.default_prompt(price=1999.99)


def test_one_shot_overrides(one_shot):
    prompt = one_shot.default_prompt(
        product_name="Other Cream", price=9.5, example={"review": "Nice."}
    )
    expected = reference_create_prompt(
        "beauty products",
        {"product_name": EXAMPLE_FIELDS["product_name"],
         "price": EXAMPLE_FIELDS["price"], "text": "Nice."},
        {"product_name": "Other Cream", "price": 9.5},
    )
    assert prompt == expected
    # the config defaults are not mutated
    assert "Target Phone" in one_shot.default_prompt()
    assert "Works fine." in one_shot.default_prompt()


def test_one_shot_missing_field_raises(model):
    m = _with_prompt(model, {"domain": "x", "template": ONE_SHOT_TEMPLATE,
                             "example": EXAMPLE_FIELDS})
    with pytest.raises(ValueError, match="product_name"):
        m.default_prompt()
    assert "Product name: Given\n" in m.default_prompt(
        product_name="Given", price=1.0, average_rating=3.0
    )


def test_zero_shot_config_ignores_fields(model):
    # old configs: the template has only {domain}
    assert model.default_prompt(product_name="X", price=1.0) == model.default_prompt()
    assert model.default_prompt() == "## Instruction\nAbout test.\nReview:"
