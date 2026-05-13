import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from config import cfg
from model import make_model


def parse_args():
    p = argparse.ArgumentParser(description="Query-Gallery visual matching")
    p.add_argument('--config_file', default='configs/UAV-Swarm/vit_transreid_stride_384.yml',
                   help='YACS config file')
    p.add_argument('--query_dir', required=True, help='Directory of query images')
    p.add_argument('--gallery_dir', required=True, help='Directory of gallery images')
    p.add_argument('--weight', required=True, help='Path to trained ReID weights (.pth)')
    p.add_argument('--save_dir', default='./visual_results', help='Output dir for visualisations')
    p.add_argument('--num_class', type=int, default=122, help='Trained num classes')
    p.add_argument('--camera_num', type=int, default=6)
    p.add_argument('--view_num', type=int, default=1)
    p.add_argument('--topk', type=int, default=5)
    p.add_argument('opts', nargs=argparse.REMAINDER, default=None,
                   help='Override cfg options, e.g. INPUT.SIZE_TEST [256,256]')
    return p.parse_args()


def main():
    args = parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.TEST.WEIGHT = os.path.expanduser(args.weight)
    cfg.freeze()

    save_dir = os.path.expanduser(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    query_dir = os.path.expanduser(args.query_dir)
    gallery_dir = os.path.expanduser(args.gallery_dir)
    if not os.path.isdir(query_dir):
        raise FileNotFoundError(f"query_dir not found: {query_dir}")
    if not os.path.isdir(gallery_dir):
        raise FileNotFoundError(f"gallery_dir not found: {gallery_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = make_model(cfg, num_class=args.num_class,
                       camera_num=args.camera_num, view_num=args.view_num)
    model.load_param(cfg.TEST.WEIGHT)
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])

    def load_image(path):
        img = Image.open(path).convert("RGB")
        return transform(img).unsqueeze(0).to(device)

    @torch.no_grad()
    def extract_feature(img_tensor):
        feat = model(img_tensor)
        return feat.cpu().numpy()

    img_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    query_paths = sorted([os.path.join(query_dir, f) for f in os.listdir(query_dir)
                          if f.lower().endswith(img_exts)])
    gallery_paths = sorted([os.path.join(gallery_dir, f) for f in os.listdir(gallery_dir)
                            if f.lower().endswith(img_exts)])
    if not query_paths or not gallery_paths:
        raise RuntimeError("Empty query or gallery directory")

    print("Extracting query features...")
    query_features = np.vstack([extract_feature(load_image(p)) for p in query_paths])

    print("Extracting gallery features...")
    gallery_features = np.vstack([extract_feature(load_image(p)) for p in gallery_paths])

    print("Computing distance matrix...")
    m, n = query_features.shape[0], gallery_features.shape[0]
    distmat = np.zeros((m, n))
    for i in range(m):
        distmat[i] = np.linalg.norm(gallery_features - query_features[i], axis=1)
    np.save(os.path.join(save_dir, 'dist_mat.npy'), distmat)

    print("Visualizing results...")
    topk = args.topk
    for i, q_path in enumerate(query_paths):
        q_img = Image.open(q_path).convert("RGB")
        indices = np.argsort(distmat[i])[:topk]

        fig, axs = plt.subplots(1, topk + 1, figsize=(15, 5))
        axs[0].imshow(q_img)
        axs[0].set_title('Query')
        axs[0].axis('off')

        for j in range(topk):
            g_img = Image.open(gallery_paths[indices[j]]).convert("RGB")
            axs[j + 1].imshow(g_img)
            axs[j + 1].set_title(f'Rank-{j + 1}')
            axs[j + 1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'result_{i}.png'))
        plt.close()

    print(f"All results saved to {save_dir}")


if __name__ == '__main__':
    main()
