import os
from collections import defaultdict
from .bases import BaseImageDataset
import pdb

class UAVSwarmDataset:
    # Test query/gallery split parametreleri.
    QUERY_RATIO = 0.2       # her drone'un ilk %20 detection'ı query, kalanı gallery
    MIN_DETECTIONS = 5      # daha az detection'lı drone'lar atlanır

    def __init__(self, root='./data', **kwargs):
        root = os.environ.get('REID_DATA_ROOT', root)
        root = os.path.expanduser(root) if root else './data'
        # root parametresi data/ klasörü ya da onun altındaki train/ klasörü olabilir.
        # Geriye uyumluluk: train klasörüne doğrudan path verildiyse onun parent'ını data köküne çek.
        if os.path.isdir(os.path.join(root, 'train')):
            self.data_root = root
        elif os.path.basename(root.rstrip('/')) == 'train':
            self.data_root = os.path.dirname(root.rstrip('/'))
        else:
            self.data_root = root

        self.train_root = os.path.join(self.data_root, 'train')
        self.test_root = os.path.join(self.data_root, 'test')

        if not os.path.isdir(self.train_root):
            raise RuntimeError(
                f"UAVSwarm train root '{self.train_root}' not found. "
                f"Set DATASETS.ROOT_DIR in config or REID_DATA_ROOT env var. "
                f"Expected layout: <root>/train/UAVSwarm-XX/{{img1,gt}}/"
            )
        # Eski kod self.root kullanıyor olabilir, geriye uyumluluk için tut.
        self.root = self.train_root

        self.train = self.load_train()
        all_ids = [pid for _, pid, _, _ in self.train]
        print(f"[DEBUG] Train: Max label {max(all_ids)}, Total unique IDs {len(set(all_ids))}")

        if os.path.isdir(self.test_root):
            self.query, self.gallery = self.load_test_query_gallery()
            test_pids = set(pid for _, pid, _, _ in self.query)
            print(f"[DEBUG] Test : {len(test_pids)} drones, {len(self.query)} query, {len(self.gallery)} gallery")
        else:
            print(f"[WARN] Test root '{self.test_root}' not found, query/gallery empty.")
            self.query = []
            self.gallery = []

        self.num_train_pids = self.get_num_pids()
        self.num_train_cams = 2 if self.query else 1   # query=0, gallery=1
        self.num_train_vids = self.get_num_vids()
   

    def load_train(self):
        data = []
        train_dir = self.train_root
        frame_id = 'gt'

        # (seq_name, local_pid) -> contiguous global pid (0..N-1).
        # MOT id'leri her sequence için 1'den başlar; aynı sayı farklı
        # sequence'te farklı drone demek. Tek bir sınıfa düşmesin diye
        # global etiketle yeniden numaralandırıyoruz.
        pid_map = {}

        for folder_name in sorted(os.listdir(train_dir)):
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

            with open(gt_path, 'r') as f:
                for line in f:
                    # MOT format: frame, id, x, y, w, h, conf, cls, vis
                    parts = line.strip().split(',')
                    if len(parts) < 9:
                        continue
                    frame_number = int(parts[0])
                    local_pid = int(parts[1])

                    key = (folder_name, local_pid)
                    if key not in pid_map:
                        pid_map[key] = len(pid_map)
                    global_pid = pid_map[key]

                    img_name = f"{frame_number:06d}.jpg"
                    img_path = os.path.join(img_dir, img_name)

                    if not os.path.exists(img_path):
                        continue

                    camid = 0
                    data.append((img_path, global_pid, camid, frame_id))

        return data

    def load_test_query_gallery(self):
        """Test set'i drone-bazlı temporal split ile query/gallery'e böl.

        Her sequence için gt.txt okunur, her drone'un kronolojik
        detection'ları toplanır, ilk %20'si query (camid=0) kalan
        %80'i gallery (camid=1) olur. Aynı drone hem query hem
        gallery'de bulunur — ReID similarity matching için zorunlu.

        Test pid'leri train'den bağımsız numaralanır (classifier'a
        girmiyor, sadece evaluator ranking için).
        """
        query, gallery = [], []
        frame_id = 'gt'
        test_pid_map = {}

        for folder_name in sorted(os.listdir(self.test_root)):
            folder_path = os.path.join(self.test_root, folder_name)
            img_dir = os.path.join(folder_path, 'img1')
            gt_path = os.path.join(folder_path, 'gt', 'gt.txt')
            if not (os.path.isdir(img_dir) and os.path.isfile(gt_path)):
                continue

            # local_pid -> [(frame_number, ...) ] kronolojik
            per_drone = defaultdict(list)
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 9:
                        continue
                    frame_number = int(parts[0])
                    local_pid = int(parts[1])
                    img_name = f"{frame_number:06d}.jpg"
                    img_path = os.path.join(img_dir, img_name)
                    if not os.path.exists(img_path):
                        continue
                    per_drone[local_pid].append((frame_number, img_path))

            for local_pid, dets in per_drone.items():
                if len(dets) < self.MIN_DETECTIONS:
                    continue
                dets.sort(key=lambda t: t[0])
                key = (folder_name, local_pid)
                if key not in test_pid_map:
                    test_pid_map[key] = len(test_pid_map)
                global_pid = test_pid_map[key]
                split = max(1, int(self.QUERY_RATIO * len(dets)))
                for _, p in dets[:split]:
                    query.append((p, global_pid, 0, frame_id))
                for _, p in dets[split:]:
                    gallery.append((p, global_pid, 1, frame_id))

        return query, gallery

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
        train_dir = self.train_root
        num_vids = 0
        for folder_name in os.listdir(train_dir):
            folder_path = os.path.join(train_dir, folder_name)
            img_dir = os.path.join(folder_path, 'img1')
            gt_dir = os.path.join(folder_path, 'gt')
            if os.path.isdir(img_dir) and os.path.isdir(gt_dir):
                num_vids += 1
        return num_vids