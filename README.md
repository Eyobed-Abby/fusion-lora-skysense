# 🌍 Fusion-LoRA: Spectral Adaptation of SkySense-O
**Parameter-Efficient Multispectral Fine-Tuning for Remote Sensing**

---

## 📘 Overview
**Fine-tuning SkySense for 6-band Sentinel-2 with spectral fusion & LoRA adapters**

We introduce:
- 🧩 **Spectral Tokenizer** – converts 6-band (RGB + NIR + SWIR) imagery into pseudo-RGB features  
- 🔁 **Cross-Attention Fusion (CAF)** – aligns spectral and RGB token streams  
- 🔒 **LoRA adapters** – lightweight fine-tuning of the frozen Swin-V2 visual encoder  
- ⚙️ **Gated Late Fusion (GLF)** – merges representations before decoding  

**Goal:**  
Improve land-cover classification on BigEarthNet-S2 while:
- Keeping the **pretrained SkySense-CLIP** backbone mostly frozen.
- Training only small, efficient **fusion + LoRA modules**.

---

## 🧱 Repository Structure
```
fusion-lora-skysense/
│
├── datasets/
│   ├── datasets/                     # (Optional) Original BE data folders
│   └── scripts/
│       ├── prepare_bigearthnet_s2.py
│       ├── prepare_bigearthnet_s2_cli.py
│       ├── metadata.parquet          # Metadata for BEN-S2 samples
│       ├── config.json
│       ├── requirements.txt
│       └── README.md
│
├── external/skysense_o/              # Cloned SkySense-O project
│   ├── configs/
│   ├── datasets/
│   ├── demo/
│   ├── skysense_o/
│   ├── demo.sh
│   ├── run_train.sh
│   ├── train_net.py
│   └── project.html
│
├── fusion_lora/
│   ├── bigearthnet_dataset.py        # Our custom BigEarthNet-S2 Dataset class
│   ├── caf_module.py                 # CAF module
│   ├── glf_module.py                 # GLF module
│   ├── lora_layers.py                # LoRA implementation
│   ├── model_wrapper.py              # Main wrapper combining backbone + LoRA + fusion modules
│   ├── spectral_tokenizer.py         # Spectral-to-feature tokenizer
│   ├── earthgpt_fuse_classifier_LoRA.py
│   ├── earthgpt_fuse_classifier_checker.py
│   └── __init__.py
│
├── train_scripts/
│   ├── train_bigearthnet_cls.py      # Baseline SkySense-O on BE-S2
│   ├── train_fusion_lora.py          # **Main training entry for Spectral-LoRA**
│   ├── eval_bigearthnet.py           # Evaluation
│   ├── eval_fusion_lora.py
│   ├── test_fusion_lora_with_skysense_o.py
│   ├── demo_inference_bigearthnet.py
│   └── debug_*                       # Debug tools for backbone, spectral input, etc.
│
├── notebooks/                        # Jupyter experiments
├── results/                          # Metrics, outputs, and plots
├── jobs/                             # HPC job files (if used)
├── paper/                            # Figures and tables for the report
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
