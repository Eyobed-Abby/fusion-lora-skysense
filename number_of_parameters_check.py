from fusion_lora.earthgpt_fuse_classifier_LoRA import EarthGPTFuseClassifier
from fusion_lora.bigearthnet_dataset import BigEarthNetSpectralDataset

ds = BigEarthNetSpectralDataset("datasets/bigearthnet_s2", split="train")
model = EarthGPTFuseClassifier(num_classes=ds.num_classes)

total, trainable = 0, 0
for name, p in model.named_parameters():
    total += p.numel()
    if p.requires_grad:
        trainable += p.numel()
        print("[TRAINABLE]", name, p.shape)

print(f"Total params:     {total/1e6:.2f} M")
print(f"Trainable params: {trainable/1e6:.2f} M")

