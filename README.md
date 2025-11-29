# 🌍 Fusion-LoRA: Spectral Adaptation of SkySense-O
**Parameter-Efficient Multispectral Fine-Tuning for Remote Sensing**

---

## 📘 Overview
This project extends the **SkySense-O** open-world remote sensing foundation model to multispectral data using a **Fusion-LoRA** approach.

We introduce:
- 🧩 **Spectral Tokenizer** – converts 6-band (RGB + NIR + SWIR) imagery into pseudo-RGB features  
- 🔁 **Cross-Attention Fusion (CAF)** – aligns spectral and RGB token streams  
- 🔒 **LoRA adapters** – lightweight fine-tuning of the frozen Swin-V2 visual encoder  
- ⚙️ **Gated Late Fusion (GLF)** – merges representations before decoding  

The goal is to achieve **parameter-efficient domain adaptation** on datasets such as EuroSAT-MS or BigEarthNet-S2, within a compact fine-tuning setup.

---

## 🧱 Repository Structure
```
fusion-lora-skysense/
│
├── fusion_lora/
│   ├── bigearthnet_dataset.py              # BigEarthNet-S2 dataset loader
│   ├── spectral_tokenizer.py               # 6-band → 1024-dim spectral tokens
│   ├── caf_module.py                       # Cross-attention fusion module
│   ├── glf_module.py                       # Global-local fusion
│   ├── lora_layers.py                      # Generic LoRA modules
│   ├── clip_lora_injector.py               # NEW: Injects LoRA into CLIP attention layers
│   ├── clip_linear_clip.py                 # NEW: Linear projection / classifier for CLIP
│   ├── model_wrapper.py                    # Combines backbone + tokenizer + CAF + GLF + LoRA
│   ├── earthgpt_fuse_classifier_LoRA.py    # Old classifier version for fusion
│   ├── earthgpt_fuse_classifier_checker.py
│   ├── earthgpt_fuse_classifier_clip_lora.py  # NEW: Classifier when CLIP has LoRA adapters
│   └── __init__.py
│
├── train_scripts/
│   ├── train_bigearthnet_cls.py            # Baseline: SkySense-O only
│   ├── train_bigearthnet_clip_lora_v2.py   # NEW: Train CLIP-visual-LoRA (no spectral fusion)
│   ├── train_bigearthnet_clip_lora.py
│   ├── train_fusion_lora.py                # MAIN: Spectral + CLIP-LoRA + CAF + GLF
│   ├── eval_bigearthnet.py
│   ├── eval_bigearthnet_clip_lora.py
│   ├── eval_fusion_lora.py
│   ├── test_fusion_lora_with_skysense_o.py
│   ├── demo_inference_bigearthnet.py
│   └── debug_*                             # Tooling & debugging utilities
│
├── datasets/
│   ├── datasets/                           # (Your local BE-S2 data folders)
│   └── scripts/
│       ├── prepare_bigearthnet_s2.py
│       ├── prepare_bigearthnet_s2_cli.py
│       ├── metadata.parquet
│       ├── config.json
│       └── requirements.txt
│
├── external/skysense_o/                    # Original SkySense-O repository
│   ├── configs/
│   ├── datasets/
│   ├── skysense_o/
│   ├── demo/
│   ├── run_train.sh
│   └── train_net.py
│
├── paper/                                  # Figures, diagrams for report/paper
├── results/                                # Metrics, F1 CSVs, confusion stats
├── notebooks/                              # Jupyter analysis
├── jobs/                                   # HPC job files (if used)
│
├── README.md
└── LICENSE

```
---

## ⚙️ Environment Setup
1. **Clone the repository (with submodule):**
   ```bash
   git clone --recurse-submodules git@github.com:Eyobed-Abby/fusion-lora-skysense.git
   cd fusion-lora-skysense

2. Create virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
3. Verify setup:
```bash
   python train_scripts/train_fusion_lora.py
```
🛰️ Dataset Preparation

Use EuroSAT-MS or BigEarthNet-S2 Sentinel-2 tiles

Each sample: [6, 256, 256] (float32, normalized to [0–1])

Save tensors under:
```bash
datasets/eurosat_ms/train_tensors/
datasets/eurosat_ms/test_tensors/
```
Development Workflow (for all contributors)
1️⃣ Branch Naming Convention

| Task Type     | Example Branch          |
| ------------- | ----------------------- |
| Feature       | `feature/dataloader`    |
| Fix           | `fix/preprocessing-bug` |
| Experiment    | `exp/lora-rank16`       |
| Documentation | `docs/readme-update`    |

2️⃣ Branch Workflow Summary

main branch      → stable, reviewed, protected
develop branch   → optional (for integration)
feature branches → each teammate works here


3️⃣ Steps for Each Member
# create your own branch
```bash
git checkout -b feature/tokenizer-update
```
# make changes, commit locally
```bash
git add .
git commit -m "Implement spectral tokenizer"
```
# push your branch to GitHub
```bash
git push origin feature/tokenizer-update
```
4️⃣ Pull Request (PR)

Once your task is done, open a Pull Request (PR) to main.

The project lead (Winner Abula) will review, test, and merge.

Please don't push directly to main.
