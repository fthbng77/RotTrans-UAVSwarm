import torch
import os  # Bu satırı ekleyin

# Modeli yükle
model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True)

# Modeli kaydet
os.makedirs("/home/fatih/.cache/torch/checkpoints", exist_ok=True)
save_path = "/home/fatih/.cache/torch/checkpoints/deit_small_patch16_224.pth"
torch.save(model.state_dict(), save_path)

# Boyutu kontrol et
print(f"Model başarıyla kaydedildi. Boyut: {os.path.getsize(save_path)/1e6:.1f}MB")
print(f"Dosya yolu: {save_path}")