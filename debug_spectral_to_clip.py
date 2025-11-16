import torch
from pathlib import Path

from fusion_lora.spectral_tokenizer import SpectralTokenizer

def main():
    # Load one of your prepared samples
    sample_path = Path("datasets/bigearthnet_s2/train_tensors/sample_000003.pt")
    x = torch.load(sample_path)   # [6, 256, 256]
    x = x.unsqueeze(0)            # [1, 6, 256, 256]
    print("Input:", x.shape, "range:", (x.min().item(), x.max().item()))

    tokenizer = SpectralTokenizer(in_channels=6, out_dim=3)
    out = tokenizer(x, target_size=(384, 384))

    print("Tokenized:", out.shape, "range:", (out.min().item(), out.max().item()))

if __name__ == "__main__":
    main()
