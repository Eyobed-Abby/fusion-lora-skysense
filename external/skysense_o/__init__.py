# from skysense_o.data import build_detection_test_loader, build_detection_train_loader
# from skysense_o import SkySenseODatasetMapper
# from skysense_o.skysense_o_model import SkySenseO

# external/skysense_o/__init__.py

"""
Lightweight init for the vendored SkySense-O package.

We only need the modules to be importable; any environment-specific
paths (HPC, DETECTRON2_DATASETS, etc.) are handled outside.
"""

from . import data
from . import modeling
from .skysense_o_model import SkySenseO
from .data.dataset_mappers.skysense_o_dataset_mapper import SkySenseODatasetMapper

__all__ = ["data", "modeling", "SkySenseO", "SkySenseODatasetMapper"]
