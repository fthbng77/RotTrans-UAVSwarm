"""Visualize UAVSwarm MOT-format data.

For every sequence under <root>/UAVSwarm-XX/:
  - reads gt/gt.txt (MOT format: frame, id, x, y, w, h, conf, cls, vis)
  - draws bbox + ID on each frame
  - writes annotated MP4 to <out>/UAVSwarm-XX.mp4 (unless --no-video)
  - prints per-sequence ID stats: unique IDs, per-ID frame span,
    re-appearance gaps (max_gap>1 = drone left frame and came back).

Usage:
    python utils/vis_dataset.py --root data/train --out vis_output
    python utils/vis_dataset.py --root data/test  --out vis_test --no-video
"""
import argparse
import os
from collections import defaultdict

import cv2

PALETTE = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
    (255, 128, 0), (0, 255, 128), (128, 255, 0), (255, 0, 128),
    (200, 200, 200), (50, 200, 50), (200, 50, 200), (50, 50, 200),
]


def color_for(pid: int):
    return PALETTE[pid % len(PALETTE)]


def parse_gt(gt_path: str):
    per_frame = defaultdict(list)
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            try:
                frame = int(parts[0])
                pid = int(parts[1])
                x, y, w, h = (int(float(p)) for p in parts[2:6])
            except ValueError:
                continue
            per_frame[frame].append((pid, x, y, w, h))
    return per_frame


def id_stats(per_frame):
    frames_of = defaultdict(list)
    for f, dets in per_frame.items():
        for pid, *_ in dets:
            frames_of[pid].append(f)
    stats = {}
    for pid, frames in frames_of.items():
        frames.sort()
        gaps = [frames[i] - frames[i - 1] for i in range(1, len(frames))]
        max_gap = max(gaps) if gaps else 0
        stats[pid] = dict(
            first=frames[0],
            last=frames[-1],
            count=len(frames),
            max_gap=max_gap,
        )
    return stats


def render_sequence(seq_dir, out_path, fps=30):
    img_dir = os.path.join(seq_dir, "img1")
    gt_path = os.path.join(seq_dir, "gt", "gt.txt")
    if not (os.path.isdir(img_dir) and os.path.isfile(gt_path)):
        return None
    per_frame = parse_gt(gt_path)
    img_names = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    if not img_names:
        return None
    first = cv2.imread(os.path.join(img_dir, img_names[0]))
    H, W = first.shape[:2]
    writer = None
    if out_path:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for name in img_names:
        frame_idx = int(os.path.splitext(name)[0])
        img = cv2.imread(os.path.join(img_dir, name))
        for pid, x, y, w, h in per_frame.get(frame_idx, []):
            c = color_for(pid)
            cv2.rectangle(img, (x, y), (x + w, y + h), c, 2)
            label = f"ID {pid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x, y - th - 4), (x + tw + 2, y), c, -1)
            cv2.putText(img, label, (x + 1, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"frame {frame_idx}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if writer is not None:
            writer.write(img)
    if writer is not None:
        writer.release()
    return per_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/train")
    ap.add_argument("--out", default="vis_output")
    ap.add_argument("--seq", default=None)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seqs = sorted(os.listdir(args.root))
    if args.seq:
        seqs = [args.seq]

    for seq in seqs:
        seq_dir = os.path.join(args.root, seq)
        if not os.path.isdir(seq_dir):
            continue
        print(f"\n=== {seq} ===")
        out_path = None if args.no_video else os.path.join(args.out, f"{seq}.mp4")
        per_frame = render_sequence(seq_dir, out_path, fps=args.fps)
        if per_frame is None:
            continue
        stats = id_stats(per_frame)
        reentered = [pid for pid, s in stats.items() if s["max_gap"] > 1]
        print(f"  frames: {len(per_frame)}  unique IDs: {len(stats)}")
        print(f"  IDs re-entering (max_gap>1): {len(reentered)} -> {sorted(reentered)}")
        for pid in sorted(stats):
            s = stats[pid]
            marker = "  <-- REENTRY" if s["max_gap"] > 1 else ""
            print(f"    ID {pid:>3}: frames {s['first']}->{s['last']}  count={s['count']}  max_gap={s['max_gap']}{marker}")


if __name__ == "__main__":
    main()
