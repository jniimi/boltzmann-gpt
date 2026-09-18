"""Inference-only reference implementation of the energy-based attribute model
described in the accompanying TMLR paper (see README).
"""

from .adapter import BeliefAdapter
from .dbm import DeepBoltzmannMachine
from .features import AttributeGroup, FeatureSpec
from .model import AttributeModel, select_device

__all__ = [
    "AttributeModel",
    "AttributeGroup",
    "BeliefAdapter",
    "DeepBoltzmannMachine",
    "FeatureSpec",
    "select_device",
]

__version__ = "0.1.0"
