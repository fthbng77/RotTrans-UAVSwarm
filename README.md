# RotTrans-UAVSwarm

Vision Transformer tabanli, rotasyon ve translasyon arttirimli (augmentation) bir **Person Re-Identification (ReID)** sistemi. UAV (Insansiz Hava Araci) suru senaryolarina yonelik olarak gelistirilmistir.

## Genel Bakis

Bu proje, hava araci suruleri tarafindan elde edilen goruntulerde kisi yeniden tanimlamayi (Re-ID) gerceklestirmek icin **DeiT/ViT** tabanli bir transformer mimarisi kullanir. Temel yenilik, patch tokenlerine **rotasyon ve shuffle** islemleri uygulayarak farkli bakis acilarinda daha dayanikli ozellik (feature) cikarmaktir.

### Temel Ozellikler

- **RotTrans Modulu**: Patch tokenlerine rastgele rotasyon (-10 ile +10 derece) ve shuffle islemleri uygulayarak gorunum degisimlerine karsi dayanklilik saglar.
- **Joint Patch Mining (JPM)**: Global ve rotasyonlu dallar (branch) ile coklu ozellik cikarimi. 1 global + 4 rotasyon dali paralel calisir.
- **Coklu Veri Seti Destegi**: UAV-Swarm, UAV-Human, PRAI-1581, VRAI, UAV-VeID.
- **YOLO + ReID Takip**: Gercek zamanli nesne tespiti ve ReID tabanli coklu nesne takibi (MOT).

## Proje Yapisi

```
RotTrans-UAVSwarm/
├── train.py                 # Egitim scripti
├── test.py                  # Test ve metrik hesaplama
├── test_rt_v2.py            # YOLO + ReID gercek zamanli takip
├── test_vis.py              # Sorgu-galeri gorsel eslestirme
├── requirements.txt         # Bagimliliklar
│
├── config/                  # YACS tabanli konfigürasyon sistemi
│   └── defaults.py          # Varsayilan parametreler
│
├── configs/                 # Veri setine ozel YAML konfigürasyonlari
│   ├── transformer_base.yml
│   ├── UAV-Swarm/           # UAV-Swarm veri seti ayarlari
│   ├── UAV-Human/           # UAV-Human veri seti ayarlari
│   ├── PRAI-1581/           # PRAI-1581 veri seti ayarlari
│   ├── VRAI/                # VRAI veri seti ayarlari
│   └── UAV-VeID/            # UAV araç ReID veri seti ayarlari
│
├── model/                   # Model mimarileri
│   ├── make_model.py        # Model fabrikasi (Backbone, Transformer, JPM)
│   └── backbones/
│       ├── resnet.py        # ResNet50 omurgasi
│       └── vit_pytorch.py   # ViT/DeiT TransReID implementasyonu
│
├── loss/                    # Kayip fonksiyonlari
│   ├── make_loss.py         # Kayip fonksiyonu fabrikasi
│   ├── softmax_loss.py      # Label smoothing ile CrossEntropy
│   ├── triplet_loss.py      # Hard mining ile Triplet Loss
│   ├── center_loss.py       # Center Loss
│   ├── arcface.py           # ArcFace / CosFace / CircleLoss
│   └── metric_learning.py   # Metrik ogrenme kayiplari
│
├── datasets/                # Veri seti yukleyicileri
│   ├── make_dataloader.py   # DataLoader fabrikasi
│   ├── sampler.py           # RandomIdentitySampler
│   ├── uavswarm.py          # UAV-Swarm veri seti (MOT formati)
│   ├── uavhuman.py          # UAV-Human veri seti
│   ├── prai1581.py          # PRAI-1581 veri seti
│   ├── vrai.py              # VRAI veri seti
│   └── uav_veid.py          # UAV araç ReID veri seti
│
├── processor/               # Egitim ve cikarsama dongusu
│   └── processor.py         # do_train() ve do_inference()
│
├── solver/                  # Optimizasyon
│   ├── make_optimizer.py    # SGD / Adam / AdamW
│   ├── cosine_lr.py         # Cosine annealing zamanlayici
│   └── lr_scheduler.py      # Warmup + multi-step zamanlayici
│
└── utils/                   # Yardimci araclar
    ├── metrics.py           # CMC, mAP, mINP degerlendirme
    ├── reranking.py         # Re-ranking algoritmasi
    ├── logger.py            # Log yapilandirmasi
    └── meter.py             # AverageMeter
```

## Model Mimarisi

```
Girdi Goruntu (256x256)
        │
        ▼
   DeiT-Small Backbone (patch_size=16, stride=12)
        │
        ▼
  Patch Token'lari (B, N, 384)
        │
        ├──► Global Dal ──► Transformer Blogu (b1) ──► CLS Token ──► Siniflandirici₀
        │
        ├──► Rotasyon Dali 1 ──► rotation() ──► b2_1 ──► CLS Token ──► Siniflandirici₁
        ├──► Rotasyon Dali 2 ──► rotation() ──► b2_2 ──► CLS Token ──► Siniflandirici₂
        ├──► Rotasyon Dali 3 ──► rotation() ──► b2_3 ──► CLS Token ──► Siniflandirici₃
        └──► Rotasyon Dali 4 ──► rotation() ──► b2_4 ──► CLS Token ──► Siniflandirici₄

Egitim: Her daldan (cls_scores, features) doner
Test:   BNNeck sonrasi global feature vektoru doner
```

## Kurulum

```bash
git clone https://github.com/fthbng77/RotTrans-UAVSwarm.git
cd RotTrans-UAVSwarm
pip install -r requirements.txt
```

### Bagimliliklar

- Python 3.8+
- PyTorch
- torchvision
- timm (pre-trained modeller)
- yacs (konfigürasyon)
- opencv-python

### Ek Bagimliliklar (Opsiyonel)

- `ultralytics` - YOLO tabanli gercek zamanli takip icin (`test_rt_v2.py`)
- `ptflops`, `thop` - Model karmasiklik analizi icin (`test.py`)
- `matplotlib` - Gorsel sonuc uretimi icin (`test_vis.py`)

## Kullanim

### Pretrained Agirliklari Indirme

DeiT-Small pretrained agirliklarini indirin ve `~/.cache/torch/checkpoints/` dizinine yerlestin:
- `deit_small_patch16_224.pth`

### Egitim

```bash
python train.py --config_file configs/UAV-Swarm/vit_transreid_stride_384.yml \
    OUTPUT_DIR ./output/uavswarm \
    SOLVER.MAX_EPOCHS 200 \
    SOLVER.BASE_LR 0.008
```

Dagitik egitim (multi-GPU):

```bash
python -m torch.distributed.launch --nproc_per_node=2 train.py \
    --config_file configs/UAV-Swarm/vit_transreid_stride_384.yml \
    MODEL.DIST_TRAIN True \
    OUTPUT_DIR ./output/uavswarm
```

### Test ve Degerlendirme

```bash
python test.py --config_file configs/UAV-Swarm/vit_transreid_stride_384.yml \
    TEST.WEIGHT ./output/uavswarm/transformer_50.pth
```

Cikti olarak **CMC (Rank-1, Rank-5, Rank-10)**, **mAP** ve **mINP** metrikleri raporlanir.

### Gercek Zamanli Takip (YOLO + ReID)

`test_rt_v2.py` dosyasindaki ayarlari duzenleyin:

```python
FRAMES_DIR = "/path/to/frames/"        # Kare goruntulerin dizini
YOLO_WEIGHTS = "best.pt"               # YOLO model agirliklari
REID_WEIGHT = "/path/to/transformer_50.pth"  # ReID model agirliklari
```

```bash
python test_rt_v2.py
```

Ciktilar `tracking_out/` dizinine kaydedilir:
- `output.mp4` - Takip videosu
- `tracks.csv` - Kare bazli takip sonuclari (frame, id, x1, y1, x2, y2, score)

### Gorsel Eslestirme Sonuclari

`test_vis.py` dosyasindaki yollari duzenleyin:

```python
query_dir = "/path/to/query"
gallery_dir = "/path/to/gallery"
weight_path = "/path/to/transformer_50.pth"
```

```bash
python test_vis.py
```

Her sorgu goruntusu icin en yakin 5 galeri eslesmesini `visual_results/` dizinine kaydeder.

## Konfigürasyon

Varsayilan UAV-Swarm konfigürasyonu (`configs/UAV-Swarm/vit_transreid_stride_384.yml`):

| Parametre | Deger |
|---|---|
| Backbone | DeiT-Small (384-dim) |
| Girdi Boyutu | 256 x 256 |
| Patch Boyutu | 16, stride 12 |
| JPM (Rotation) | Acik (4 rotasyon dali) |
| Kayip Fonksiyonu | Softmax + Triplet (soft margin) |
| Optimizer | SGD (lr=0.008) |
| Epoch | 200 |
| Batch Boyutu | 16 |
| Warmup | Linear, 5 epoch |
| LR Zamanlayici | Cosine annealing |

## Desteklenen Veri Setleri

| Veri Seti | Aciklama | Konfigürasyon |
|---|---|---|
| **UAV-Swarm** | UAV suru tabanli kisi ReID (MOT formati) | `configs/UAV-Swarm/` |
| **UAV-Human** | UAV tabanli insan ReID | `configs/UAV-Human/` |
| **PRAI-1581** | Yaya ReID | `configs/PRAI-1581/` |
| **VRAI** | Gorunum bagmsiz ReID | `configs/VRAI/` |
| **UAV-VeID** | UAV tabanli arac ReID | `configs/UAV-VeID/` |

## Kayip Fonksiyonlari

- **CrossEntropy + Label Smoothing**: Kimlik siniflandirma kaybi
- **Triplet Loss (Hard Mining)**: Metrik ogrenme kaybi
- **ArcFace / CosFace / AMSoftmax / CircleLoss**: Acisal margin tabanli alternatif kimlik kayiplari
- **Center Loss**: Sinif icindeki varyans azaltma (opsiyonel)

## Degerlendirme Metrikleri

- **CMC Egrisi**: Rank-1, Rank-5, Rank-10, Rank-50 dogruluk oranlari
- **mAP**: Ortalama ortalama hassasiyet (Mean Average Precision)
- **mINP**: Ortalama ters negatif cezasi (Mean Inverse Negative Penalty)
