import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from config import cfg
from model import make_model
from datasets.make_dataloader import make_dataloader

# ============ YOL AYARLARI =============
query_dir = '/home/fatih/github/RotTrans/output_Swarm_UAV_RE-ID_Stage_2_31.07/query'
gallery_dir = '/home/fatih/github/RotTrans/output_Swarm_UAV_RE-ID_Stage_2_31.07/gallery'
config_file = 'configs/UAV-Swarm/vit_transreid_stride_384.yml'
weight_path = '/home/fatih/github/RotTrans/output/UAVSwarm_train/transformer_50.pth'
save_dir = './visual_results'
os.makedirs(save_dir, exist_ok=True)

# ============ CİHAZ ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ CONFIG ============
cfg.merge_from_file(config_file)
cfg.TEST.WEIGHT = weight_path
cfg.freeze()

# ============ MODEL ============
model = make_model(cfg, num_class=122, camera_num=6, view_num=1)
model.load_param(cfg.TEST.WEIGHT)
model.eval()
model.to(device)

# ============ DÖNÜŞÜMLER ============
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)  # batch size=1

def extract_feature(img_tensor):
    with torch.no_grad():
        feat = model(img_tensor)
        feat = feat.cpu().numpy()
    return feat

# ============ PATHS ============
query_paths = sorted([os.path.join(query_dir, f) for f in os.listdir(query_dir)])
gallery_paths = sorted([os.path.join(gallery_dir, f) for f in os.listdir(gallery_dir)])

# ============ FEATURE EXTRACTION ============
print("Extracting query features...")
query_features = [extract_feature(load_image(p)) for p in query_paths]
query_features = np.vstack(query_features)

print("Extracting gallery features...")
gallery_features = [extract_feature(load_image(p)) for p in gallery_paths]
gallery_features = np.vstack(gallery_features)

# ============ DISTANCE MATRIX ============
print("Computing distance matrix...")
def compute_distmat(qf, gf):
    m, n = qf.shape[0], gf.shape[0]
    distmat = np.zeros((m, n))
    for i in range(m):
        distmat[i] = np.linalg.norm(gf - qf[i], axis=1)
    return distmat

distmat = compute_distmat(query_features, gallery_features)
np.save(os.path.join(save_dir, 'dist_mat.npy'), distmat)

# ============ GÖRSELLEŞTİRME ============
print("Visualizing results...")
topk = 5
for i, q_path in enumerate(query_paths):
    q_img = Image.open(q_path).convert("RGB")
    indices = np.argsort(distmat[i])[:topk]

    fig, axs = plt.subplots(1, topk + 1, figsize=(15, 5))
    axs[0].imshow(q_img)
    axs[0].set_title('Query')
    axs[0].axis('off')

    for j in range(topk):
        g_img = Image.open(gallery_paths[indices[j]]).convert("RGB")
        axs[j+1].imshow(g_img)
        axs[j+1].set_title(f'Rank-{j+1}')
        axs[j+1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'result_{i}.png')
    plt.savefig(save_path)
    plt.close()

print(f"Tüm sonuçlar {save_dir} klasörüne kaydedildi.")
