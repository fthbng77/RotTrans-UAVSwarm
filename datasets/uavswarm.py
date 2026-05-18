import os
from .bases import BaseImageDataset
import pdb
import torch

class UAVSwarmDataset:
    def __init__(self, root='./data', **kwargs):
        root = os.environ.get('REID_DATA_ROOT', root)
        root = os.path.expanduser(root) if root else './data'
        candidate = os.path.join(root, 'train')
        self.root = candidate if os.path.isdir(candidate) else root
        if not os.path.isdir(self.root):
            raise RuntimeError(
                f"UAVSwarm dataset root '{self.root}' not found. "
                f"Set DATASETS.ROOT_DIR in config or REID_DATA_ROOT env var. "
                f"Expected layout: <root>/train/UAVSwarm-XX/{{img1,gt}}/"
            )
        self.train = self.load_train()
        self.train = self.relabel_train_pids(self.train)
        all_ids = [pid for _, pid, _, _ in self.train]
        print(f"[DEBUG] Max label in dataset: {max(all_ids)}, Total unique IDs: {len(set(all_ids))}")


        self.query = []    # Şimdilik boş bırakabilirsin
        self.gallery = []  # Şimdilik boş bırakabilirsin
        self.num_train_pids = self.get_num_pids()
        self.num_train_cams = 1
        self.num_train_vids = self.get_num_vids()
   

    def load_train(self):
        data = []
        train_dir = self.root  # örn: data/train/
        
        # Her bir UAVSwarm-XX klasörü
        for folder_name in os.listdir(train_dir):
            folder_path = os.path.join(train_dir, folder_name)
            img_dir = os.path.join(folder_path, 'img1')
            gt_dir = os.path.join(folder_path, 'gt')
            
            if not (os.path.isdir(img_dir) and os.path.isdir(gt_dir)):
                continue
            
            # Sadece gt.txt oku. gt_train_half.txt / gt_val_half.txt aynı
            # detection'ları farklı bölüntülerde tekrar içeriyor; üçünü birden
            # okumak her örneği ~2x sayar.
            gt_path = os.path.join(gt_dir, 'gt.txt')
            if not os.path.isfile(gt_path):
                continue
            frame_id = 'gt'

            with open(gt_path, 'r') as f:
                for line in f.readlines():
                        # MOT format: frame, id, x, y, w, h, conf, cls, vis
                        parts = line.strip().split(',')
                        if len(parts) < 9:
                            continue
                        frame_number = int(parts[0])
                        pid = int(parts[1])

                        img_name = f"{frame_number:06d}.jpg"
                        img_path = os.path.join(img_dir, img_name)
                        
                        if not os.path.exists(img_path):
                            continue
                        
                        camid = 0  # UAVSwarm için kamera id yoksa 0 ver
                        
                        # Data formatı: (img_path, pid, camid, frame_id)
                        data.append((img_path, pid, camid, frame_id))
        
        return data

    def relabel_train_pids(self, data):
        pid2label = {pid: label for label, pid in enumerate(sorted({pid for _, pid, _, _ in data}))}
        return [(img_path, pid2label[pid], camid, frame_id) for img_path, pid, camid, frame_id in data]

    def get_num_pids(self):
        pids = set()
        for _, pid, _, _ in self.train:
            pids.add(pid)
        return len(pids)

    def load_param(self, model_path):
        if not model_path or not os.path.exists(model_path):
            print(f"No valid pretrained model path provided ('{model_path}'), skipping load_param.")
            return
        param_dict = torch.load(model_path, weights_only=False)
        self.load_state_dict(param_dict, strict=False)


    def get_num_vids(self):
        # Her bir UAVSwarm-XX klasörü bir video olarak sayılır
        train_dir = self.root
        num_vids = 0
        for folder_name in os.listdir(train_dir):
            folder_path = os.path.join(train_dir, folder_name)
            img_dir = os.path.join(folder_path, 'img1')
            gt_dir = os.path.join(folder_path, 'gt')
            if os.path.isdir(img_dir) and os.path.isdir(gt_dir):
                num_vids += 1
        return num_vids
