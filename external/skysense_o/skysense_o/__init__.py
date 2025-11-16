# import sys
# import os
# sys.path.append('/gruntdata/rs_nas/workspace/xingsu.zq/detectron2-xyz-main')
# os.environ['DETECTRON2_DATASETS'] = '/gruntdata/rs_nas/workspace/xingsu.zq/SkySense-O/datasets/'
# os.chdir('/gruntdata/rs_nas/workspace/xingsu.zq/SkySense-O')
# from . import data  
# from . import modeling
# from .skysense_o_model import SkySenseO
# from .data.dataset_mappers.skysense_o_dataset_mapper import SkySenseODatasetMapper

import os
from pathlib import Path

# --------------------------------------------------
# Set DETECTRON2_DATASETS to local SkySense-O datasets
# --------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
SKYSENSE_ROOT = CURRENT_DIR.parent       # external/skysense_o/
DATASET_DIR = SKYSENSE_ROOT / "datasets"

os.environ['DETECTRON2_DATASETS'] = str(DATASET_DIR)

# --------------------------------------------------
# Import SkySense-O modules
# --------------------------------------------------
from . import data
from . import modeling
from .skysense_o_model import SkySenseO
from .data.dataset_mappers.skysense_o_dataset_mapper import SkySenseODatasetMapper

