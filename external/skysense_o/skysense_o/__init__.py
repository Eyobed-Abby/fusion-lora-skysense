# import sys
# import os
# sys.path.append('/gruntdata/rs_nas/workspace/xingsu.zq/detectron2-xyz-main')
# os.environ['DETECTRON2_DATASETS'] = '/gruntdata/rs_nas/workspace/xingsu.zq/SkySense-O/datasets/'
# os.chdir('/gruntdata/rs_nas/workspace/xingsu.zq/SkySense-O')
# from . import data  
# from . import modeling
# from .skysense_o_model import SkySenseO
# from .data.dataset_mappers.skysense_o_dataset_mapper import SkySenseODatasetMapper



import sys
import os
from pathlib import Path

# Repo root is the parent of the 'skysense_o' package directory
REPO_ROOT = Path(__file__).resolve().parents[1]

# Ensure the repo root is on sys.path so 'skysense_o' and local modules resolve
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Point Detectron2 to the local datasets folder (only set if not already set)
os.environ.setdefault("DETECTRON2_DATASETS", str(REPO_ROOT / "datasets"))


# sys.path.append('/gruntdata/rs_nas/workspace/xingsu.zq/detectron2-xyz-main')
# os.environ['DETECTRON2_DATASETS'] = '/gruntdata/rs_nas/workspace/xingsu.zq/SkySense-O/datasets/'
# os.chdir('/gruntdata/rs_nas/workspace/xingsu.zq/SkySense-O')
from . import data  
from . import modeling
from .skysense_o_model import SkySenseO
from .data.dataset_mappers.skysense_o_dataset_mapper import SkySenseODatasetMapper