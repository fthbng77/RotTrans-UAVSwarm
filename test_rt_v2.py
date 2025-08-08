import os, cv2, time, json, numpy as np
from glob import glob
from collections import deque

# ==== YOLO dedektör (Ultralytics) ====
# pip install ultralytics
from ultralytics import YOLO

import torch
from torchvision import transforms
from PIL import Image
from datetime import datetime
# ==== RotTrans ReID ====
from config import cfg
from model import make_model

########################################
# KULLANICI AYARLARI
########################################
FRAMES_DIR = "/home/fatih/github/RotTrans/data/Test_Swarm_UAV_RE-ID_08.08/UAVSwarm-FB/img1/"     
test_name = os.path.basename(os.path.dirname(FRAMES_DIR.rstrip('/')))
run_name = f"{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

OUTPUT_DIR = os.path.join("./tracking_out", run_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAKE_VIDEO = True                            # True -> output.mp4 üretir
FPS = 25

# YOLO model (drone tespiti için kendi ağırlığın varsa onu kullan)
YOLO_WEIGHTS = "best.pt"                  # örnek
YOLO_CONF    = 0.25
YOLO_IOU     = 0.5
YOLO_IMGSZ   = 1280

# ReID (RotTrans)
CFG_FILE     = "configs/UAV-Swarm/vit_transreid_stride_384.yml"
REID_WEIGHT  = "/home/fatih/github/RotTrans/output/UAVSwarm_train/transformer_50.pth"
NUM_CLASS    = 122                           # eğitiminle uyumluysa; sadece feature istiyoruz zaten
COS_THRESH   = 0.55                          # ReID eşiği (0.6–0.8 arası deneyin)
IOU_THRESH   = 0.3                           # kısa kopmalarda IoU ile eşleştirme
LOST_TTL     = 10.0                          # sn: bu süreden uzun kayıpları unut
EMA_ALPHA    = 0.8                           # feat güncelleme
########################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====== ReID Modelini yükle ======
cfg.merge_from_file(CFG_FILE)
cfg.TEST.WEIGHT = REID_WEIGHT
cfg.freeze()

reid_model = make_model(cfg, num_class=NUM_CLASS, camera_num=1, view_num=1)
# Esnek yükleyici: classifier shape uymuyorsa atlar
def load_param_flexible(model, trained_path, map_location='cpu'):
    state = torch.load(trained_path, map_location=map_location)
    if 'state_dict' in state:
        state = state['state_dict']
    model_state = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in state.items():
        k2 = k.replace('module.', '')
        if k2 in model_state and model_state[k2].shape == v.shape:
            model_state[k2].copy_(v)
            loaded += 1
        else:
            skipped += 1
    print(f"[ReID load] loaded:{loaded}, skipped:{skipped}")

load_param_flexible(reid_model, REID_WEIGHT, map_location=device)
reid_model.eval().to(device)

tfm = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
])

@torch.no_grad()
def extract_feats_batch(crops_bgr):
    """BGR crop list -> (N,D) L2-normalize embedding."""
    if len(crops_bgr) == 0:
        return np.zeros((0, 384), dtype=np.float32)  # 384 örnek; modeline göre değişebilir
    imgs = []
    for bgr in crops_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        imgs.append(tfm(pil))
    tens = torch.stack(imgs, dim=0).to(device)
    feats = reid_model(tens)
    if isinstance(feats, (list, tuple)):
        feats = feats[0]
    feats = torch.nn.functional.normalize(feats, dim=1)
    return feats.cpu().numpy()

def cosine_sim_matrix(Q, G):
    # Q:(M,D), G:(N,D) -> (M,N)
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
    return Qn @ Gn.T

def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0, x2-x1); ih = max(0, y2-y1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-12
    return inter/ua

class Track:
    def __init__(self, tid, bbox, feat, t):
        self.id = tid
        self.bbox = bbox
        self.feat = feat / (np.linalg.norm(feat)+1e-12)
        self.last_time = t
        self.miss = 0
        self.history = deque(maxlen=30)

    def update(self, bbox, feat, t, alpha=EMA_ALPHA):
        self.bbox = bbox
        self.feat = alpha*self.feat + (1-alpha)*feat
        self.feat = self.feat / (np.linalg.norm(self.feat)+1e-12)
        self.last_time = t
        self.miss = 0
        self.history.append(bbox)

# YOLO dedektör:
det_model = YOLO(YOLO_WEIGHTS)

# Frame listesi
frame_paths = sorted(glob(os.path.join(FRAMES_DIR, "*.*")))
frame_paths = [p for p in frame_paths if p.lower().endswith((".jpg",".jpeg",".png"))]
assert len(frame_paths) > 0, "FRAMES_DIR boş!"

tracks = {}
LOST = {}       # id -> dict(feat, last_time, bbox)
next_id = 1

# CSV log
csv_path = os.path.join(OUTPUT_DIR, "tracks.csv")
csv_f = open(csv_path, "w")
csv_f.write("frame,id,x1,y1,x2,y2,score\n")

# Video writer (varsayılan ilk frame boyutu)
vw = None

t_start = time.time()
for fi, fpath in enumerate(frame_paths):
    frame = cv2.imread(fpath)
    h, w = frame.shape[:2]
    now = time.time()

    # Video writer init
    if MAKE_VIDEO and vw is None:
        vw = cv2.VideoWriter(
            os.path.join(OUTPUT_DIR, "output.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS, (w, h)
        )

    # 1) Tespit
    y = det_model.predict(
        frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False
    )[0]

    det_bboxes, det_scores = [], []
    for b, s in zip(y.boxes.xyxy.cpu().numpy(), y.boxes.conf.cpu().numpy()):
        x1,y1,x2,y2 = [int(v) for v in b]
        x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
        if x2<=x1 or y2<=y1: continue
        det_bboxes.append([x1,y1,x2,y2])
        det_scores.append(float(s))

    # 2) ReID feature’larını batch çıkar
    crops = [frame[y1:y2, x1:x2] for (x1,y1,x2,y2) in det_bboxes]
    det_feats = extract_feats_batch(crops)  # (N,D)

    # 3) Mevcut track’lerle eşleştirme (IoU + Cosine)
    used_tracks = set()
    assigned = [-1]*len(det_bboxes)

    # Önce IoU ile yakın olanları bağla
    for di, db in enumerate(det_bboxes):
        best_tid, best_i = None, 0.0
        for tid, tr in tracks.items():
            i = iou(db, tr.bbox)
            if i > best_i:
                best_i, best_tid = i, tid
        if best_i >= IOU_THRESH:
            tracks[best_tid].update(db, det_feats[di], now)
            used_tracks.add(best_tid)
            assigned[di] = best_tid

    # IoU ile eşleşmeyenler için ReID ile LOST & aktif track’lere bak
    # (aktif track'lerden kullanılmayanlara cosine ile sor)
    # 3a) aktif track cosine
    remain_idx = [i for i,a in enumerate(assigned) if a == -1]
    if remain_idx:
        active_ids, active_feats = [], []
        for tid, tr in tracks.items():
            if tid not in used_tracks:
                active_ids.append(tid)
                active_feats.append(tr.feat)
        if len(active_ids) > 0:
            sims = cosine_sim_matrix(det_feats[remain_idx], np.stack(active_feats, axis=0))
            for ri, row in enumerate(sims):
                k = int(np.argmax(row))
                if row[k] >= COS_THRESH:
                    di = remain_idx[ri]
                    tid = active_ids[k]
                    tracks[tid].update(det_bboxes[di], det_feats[di], now)
                    used_tracks.add(tid)
                    assigned[di] = tid

    # 3b) LOST havuzu ile ReID
    remain_idx = [i for i,a in enumerate(assigned) if a == -1]
    if remain_idx and len(LOST)>0:
        cand_ids, cand_feats = [], []
        for lid, info in list(LOST.items()):
            if now - info["last_time"] <= LOST_TTL:
                cand_ids.append(lid)
                cand_feats.append(info["feat"])
            else:
                LOST.pop(lid, None)
        if len(cand_ids) > 0:
            sims = cosine_sim_matrix(det_feats[remain_idx], np.stack(cand_feats, axis=0))
            for rpos, row in enumerate(sims):
                k = int(np.argmax(row))
                if row[k] >= COS_THRESH:
                    di = remain_idx[rpos]
                    tid = cand_ids[k]
                    tracks[tid] = Track(tid, det_bboxes[di], det_feats[di], now)
                    LOST.pop(tid, None)
                    assigned[di] = tid

    # 3c) hala eşleşmeyenler -> yeni ID
    for di in [i for i,a in enumerate(assigned) if a == -1]:
        tid = next_id; next_id += 1
        tracks[tid] = Track(tid, det_bboxes[di], det_feats[di], now)
        assigned[di] = tid

    # 4) Görünmeyen track’ler -> miss++
    to_del = []
    for tid, tr in tracks.items():
        if tr.last_time < now - 0.001 and tid not in set(assigned):
            tr.miss += 1
            if tr.miss >= 5:
                LOST[tid] = dict(bbox=tr.bbox, feat=tr.feat, last_time=tr.last_time)
                to_del.append(tid)
    for tid in to_del:
        tracks.pop(tid, None)

    # 5) Çiz & kaydet
    vis = frame.copy()
    for di, db in enumerate(det_bboxes):
        tid = assigned[di]
        x1,y1,x2,y2 = db
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(vis, f"ID {tid}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        csv_f.write(f"{fi},{tid},{x1},{y1},{x2},{y2},{det_scores[di]:.3f}\n")

    out_path = os.path.join(OUTPUT_DIR, f"vis_{fi:05d}.jpg")
    cv2.imwrite(out_path, vis)
    if vw is not None:
        vw.write(vis)

csv_f.close()
if vw is not None:
    vw.release()

print(f"[DONE] Frames:{len(frame_paths)}  out:{OUTPUT_DIR}")

