import os
import torch

# Download DeiT-Small pretrained weights to ~/.cache/torch/checkpoints/
model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True)

cache_dir = os.path.expanduser("~/.cache/torch/checkpoints")
os.makedirs(cache_dir, exist_ok=True)
save_path = os.path.join(cache_dir, "deit_small_patch16_224.pth")
torch.save(model.state_dict(), save_path)

print(f"Saved. Size: {os.path.getsize(save_path)/1e6:.1f}MB")
print(f"Path: {save_path}")
