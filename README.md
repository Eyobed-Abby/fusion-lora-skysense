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
├─ external/skysense_o/ # SkySense-O submodule (keep untouched)
│
├─ fusion_lora/ # Fusion-LoRA implementation
│ ├─ model_wrapper.py # main wrapper combining tokenizer + LoRA + fusion
│ ├─ spectral_tokenizer.py # 6→3 projection (Conv1×1)
│ ├─ lora_layers.py # LoRA modules and injection helpers
│ ├─ caf_module.py # cross-attention fusion
│ ├─ glf_module.py # gated late fusion
│ └─ utils/
│
├─ datasets/
│ ├─ utils/prepare_spectral_data.py
│ ├─ utils/loader.py
│ ├─ eurosat_ms/train_tensors/ # [6,256,256] .pt tensors
│ └─ config.json
│
├─ train_scripts/
│ ├─ train_fusion_lora.py
│ ├─ eval_fusion_lora.py
│ └─ cfgs/
│
├─ paper/ # ICIP-style report draft
│ ├─ ICIP2025_FusionLoRA.tex
│ └─ figures/
│
├─ results/
│ ├─ logs/
│ └─ checkpoints/
│
├─ notebooks/ # demos, visualization, data previews
│
├─ .gitignore
├─ requirements.txt
└─ README.md
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

# make changes, commit locally
git add .
git commit -m "Implement spectral tokenizer"

# push your branch to GitHub
git push origin feature/tokenizer-update

4️⃣ Pull Request (PR)

Once your task is done, open a Pull Request (PR) to main.

The project lead (Winner Abula) will review, test, and merge.

Do not push directly to main.
