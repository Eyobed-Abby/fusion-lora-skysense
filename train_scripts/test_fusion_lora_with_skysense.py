# train_scripts/test_fusion_lora_with_skysense.py

import sys
from pathlib import Path

import torch

# ------------------------------------------------------------------
# 0. Path setup: make 'fusion_lora' and the SkySense repo importable
# ------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent              # FUSION-LORA-SKYSENSE/
EXTERNAL_DIR = ROOT_DIR / "external"

sys.path.insert(0, str(ROOT_DIR))       # so `import fusion_lora` works

# adjust this line if your SkySense subfolder name is different:
# e.g. EXTERNAL_DIR / "SkySense-O" or / "skysense_o"
sys.path.insert(0, str(EXTERNAL_DIR / "skysense_o"))

import demo  # this is external/SkySense-O/demo/demo.py
from detectron2.checkpoint import DetectionCheckpointer


from fusion_lora import FusionLoRAWrapper   # from fusion_lora/__init__.py


# ------------------------------------------------------------------
# 1. Get the visual encoder you already use in SkySense-O
# ------------------------------------------------------------------
def build_visual_encoder(device):
    """
    Build the SkySense model using the same logic as demo.py,
    then extract the visual encoder submodule (the one that supports dense=True).
    """

    # 1) Create config exactly like demo.main() does
    cfg = demo.init_config()
    cfg.set_new_allowed(True)

    # Path to the SkySense config YAML used in the demo command:
    # python demo/demo.py --config-file configs/skysense_o_demo.yaml ...
    REPO_ROOT = Path(demo.__file__).resolve().parent.parent  # external/SkySense-O/
    CONFIG_PATH = REPO_ROOT / "configs" / "skysense_o_demo.yaml"

    cfg.merge_from_file(str(CONFIG_PATH))
    cfg.set_new_allowed(True)
    # you can also merge_from_list(...) if you want extra opts
    cfg.freeze()

    # 2) Build the full SkySense model using Trainer.build_model
    model = demo.Trainer.build_model(cfg)
    model.to(device)
    model.eval()

    # 3) Load weights using the same mechanism as demo.py
    #    cfg.MODEL.WEIGHTS is defined in the YAML, so we can reuse it
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )

    # 4) Inspect model **once** to find the visual encoder attribute
    print("\n[build_visual_encoder] Model top-level modules:\n")
    for name, module in model.named_children():
        print(f"  {name}: {type(module)}")

    # 👇 IMPORTANT:
    # After you run this once, check what prints here (e.g. 'clip_model', 'backbone', etc.)
    # and then set visual_encoder accordingly.
    #
    # Example possibilities (you will choose the correct one):
    #
    # visual_encoder = model.clip_model.visual
    # visual_encoder = model.backbone.clip_model.visual
    # visual_encoder = model.backbone
    #
    # For now, we raise so that you remember to set it:
    raise RuntimeError(
        "Check the printed modules above and set `visual_encoder = ...` "
        "to the submodule that takes (x, dense=True) and returns a list of feature maps."
    )

    # Once you know the right path, replace the raise with something like:
    #
    # visual_encoder = model.clip_model.visual
    # visual_encoder.to(device)
    # visual_encoder.eval()
    # return visual_encoder

# ------------------------------------------------------------------
# 2. Debug: check stage shapes from visual_encoder(dense=True)
# ------------------------------------------------------------------

def debug_visual_encoder(visual_encoder, device):
    rgb_dummy = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        feats = visual_encoder(rgb_dummy, dense=True)

    print("Number of stages:", len(feats))
    for i, f in enumerate(feats):
        print(f"Stage {i}: shape = {tuple(f.shape)}")


# ------------------------------------------------------------------
# 3. Test FusionLoRAWrapper with dummy RGB + 6-band spectral inputs
# ------------------------------------------------------------------

def test_fusion_wrapper(visual_encoder, device):
    fusion_model = FusionLoRAWrapper(visual_encoder).to(device)
    fusion_model.eval()   # we only care about shapes for now

    batch_size = 2
    H = 224
    W = 224

    rgb_dummy = torch.randn(batch_size, 3, H, W, device=device)
    ms_dummy  = torch.randn(batch_size, 6, H, W, device=device)

    with torch.no_grad():
        out = fusion_model(rgb_dummy, ms_dummy)

    print("\nFusionLoRAWrapper output:")
    print("keys:", out.keys())
    for k, v in out.items():
        print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")


# ------------------------------------------------------------------
# 4. Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) build visual encoder from SkySense
    visual_encoder = build_visual_encoder(device)

    # 2) print its stage shapes
    print("\n=== Visual encoder dense feature shapes ===")
    debug_visual_encoder(visual_encoder, device)

    # 3) run the fusion architecture test
    print("\n=== Testing FusionLoRAWrapper ===")
    test_fusion_wrapper(visual_encoder, device)
