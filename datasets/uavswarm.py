import os
from .bases import BaseImageDataset
import pdb

class UAVSwarmDataset:
    def __init__(self, root):
        self.root = '/home/fatih/github/RotTrans/data/train'
        self.train = self.load_train()
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
            
            # GT dosyalarını oku (her frame için)
            for gt_file in sorted(os.listdir(gt_dir)):
                gt_path = os.path.join(gt_dir, gt_file)
                frame_id = os.path.splitext(gt_file)[0]
                
                # Her gt dosyası satır satır, her satır bir obje
                with open(gt_path, 'r') as f:
                    for line in f.readlines():
                        # line formatı örnek: 
                        # id, frame, x, y, w, h, conf, type, visibility
                        parts = line.strip().split(',')
                        if len(parts) < 9:
                            continue
                        pid = int(parts[0])
                        frame_number = parts[1]  # zaten dosya ismi ile aynı olabilir
                        
                        # Görüntü dosyası yolu (örnek): img1/000001.jpg
                        img_name = f"{int(frame_number):06d}.jpg"
                        img_path = os.path.join(img_dir, img_name)
                        
                        if not os.path.exists(img_path):
                            continue
                        
                        camid = 0  # UAVSwarm için kamera id yoksa 0 ver
                        
                        # Data formatı: (img_path, pid, camid, frame_id)
                        data.append((img_path, pid, camid, frame_id))
        
        return data

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