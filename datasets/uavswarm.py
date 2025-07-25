import os
from .bases import ImageDataset

class UAVSwarmDataset:
    def __init__(self, root):
        self.root = '/home/fatih/github/RotTrans/data/train'
        self.train = self.load_train()
        self.query = []    # Şimdilik boş bırakabilirsin
        self.gallery = []  # Şimdilik boş bırakabilirsin
        self.num_train_pids = self.get_num_pids()
        self.num_train_cams = 1

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
                        
                        # Data formatı: (img_path, pid, camid)
                        data.append((img_path, pid, camid))
        
        return data

    def get_num_pids(self):
        pids = set()
        for _, pid, _ in self.train:
            pids.add(pid)
        return len(pids)