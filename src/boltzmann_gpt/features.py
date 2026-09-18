"""Attribute schema: named groups of visible units and how to set them.

The DBM's visible layer is a flat binary vector. ``feature_spec.json`` records,
for each named attribute group, which visible indices it owns, the label of
each index, whether the group is one-hot or multi-label, and the modal value
observed in the training data (used as the default configuration).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Union

import torch

ONE_HOT = "one_hot"
MULTI_LABEL = "multi_label"

ValueSpec = Union[str, Sequence[str], None]


@dataclass(frozen=True)
class AttributeGroup:
    """One named attribute, e.g. ``rating`` or ``topic``."""

    name: str
    kind: str  # ONE_HOT or MULTI_LABEL
    indices: List[int]
    values: List[str]
    default: Union[str, List[str], None]

    def index_of(self, value: str) -> int:
        try:
            return self.indices[self.values.index(value)]
        except ValueError:
            raise ValueError(
                f"Unknown value {value!r} for attribute {self.name!r}. "
                f"Valid values: {self.values}"
            ) from None


class FeatureSpec:
    """Attribute schema loaded from ``feature_spec.json``."""

    def __init__(self, spec: Mapping[str, Any]):
        self.domain: str = spec.get("domain", "unknown")
        self.n_visible: int = int(spec["n_visible"])
        self.feature_names: List[str] = list(spec.get("feature_names", []))

        if self.feature_names and len(self.feature_names) != self.n_visible:
            raise ValueError(
                f"feature_names has {len(self.feature_names)} entries but "
                f"n_visible is {self.n_visible}"
            )

        self.groups: Dict[str, AttributeGroup] = {}
        for name, g in spec["groups"].items():
            kind = g.get("type", ONE_HOT)
            if kind not in (ONE_HOT, MULTI_LABEL):
                raise ValueError(
                    f"Attribute {name!r} has unknown type {kind!r}; "
                    f"expected {ONE_HOT!r} or {MULTI_LABEL!r}"
                )
            indices = [int(i) for i in g["indices"]]
            values = [str(v) for v in g["values"]]
            if len(indices) != len(values):
                raise ValueError(
                    f"Attribute {name!r}: {len(indices)} indices but {len(values)} values"
                )
            bad = [i for i in indices if not 0 <= i < self.n_visible]
            if bad:
                raise ValueError(
                    f"Attribute {name!r}: indices {bad} outside [0, {self.n_visible})"
                )
            self.groups[name] = AttributeGroup(
                name=name, kind=kind, indices=indices, values=values,
                default=g.get("default"),
            )

    def attributes(self) -> Dict[str, List[str]]:
        """Group name -> allowed values."""
        return {name: list(g.values) for name, g in self.groups.items()}

    def get(self, name: str) -> AttributeGroup:
        if name not in self.groups:
            raise ValueError(
                f"Unknown attribute {name!r}. Valid attributes: {sorted(self.groups)}"
            )
        return self.groups[name]

    # -- encoding -----------------------------------------------------------

    def default_vector(self) -> torch.Tensor:
        """Visible vector where every group takes its modal training value."""
        v = torch.zeros(self.n_visible, dtype=torch.float32)
        for group in self.groups.values():
            if group.default is None:
                continue
            self._set_group(v, group, group.default)
        return v

    def encode(self, assignment: Mapping[str, ValueSpec]) -> torch.Tensor:
        """Build a visible vector; groups not named take their modal value."""
        v = self.default_vector()
        return self.clamp(v, assignment)

    def clamp(self, v: torch.Tensor, assignment: Mapping[str, ValueSpec]) -> torch.Tensor:
        """Return a copy of ``v`` with the named groups overwritten."""
        if v.shape[-1] != self.n_visible:
            raise ValueError(
                f"Visible vector has {v.shape[-1]} units, expected {self.n_visible}"
            )
        out = v.detach().clone().to(dtype=torch.float32)
        for name, value in assignment.items():
            self._set_group(out, self.get(name), value)
        return out

    def _set_group(self, v: torch.Tensor, group: AttributeGroup, value: ValueSpec) -> None:
        """Zero the group's units, then set the units named by ``value``."""
        idx = torch.tensor(group.indices, dtype=torch.long)
        v[..., idx] = 0.0

        if value is None:
            if group.kind == ONE_HOT:
                raise ValueError(
                    f"Attribute {group.name!r} is one-hot and needs a value; "
                    f"valid values: {group.values}"
                )
            return

        if isinstance(value, str):
            selected: List[str] = [value]
        elif isinstance(value, Iterable):
            selected = [str(x) for x in value]
        else:
            raise ValueError(
                f"Attribute {group.name!r}: expected a value name or a list of "
                f"value names, got {type(value).__name__}"
            )

        if group.kind == ONE_HOT and len(selected) != 1:
            raise ValueError(
                f"Attribute {group.name!r} is one-hot and takes exactly one "
                f"value, got {len(selected)}; valid values: {group.values}"
            )

        for name in selected:
            v[..., group.index_of(name)] = 1.0

    # -- decoding -----------------------------------------------------------

    def decode(self, v: torch.Tensor) -> Dict[str, Union[str, List[str], None]]:
        """Read a visible vector back as an attribute assignment."""
        if v.dim() != 1:
            raise ValueError("decode expects a single visible vector")
        out: Dict[str, Union[str, List[str], None]] = {}
        for name, group in self.groups.items():
            active = [
                value for value, i in zip(group.values, group.indices) if v[i] > 0.5
            ]
            if group.kind == ONE_HOT:
                out[name] = active[0] if len(active) == 1 else (active or None)
            else:
                out[name] = active
        return out
