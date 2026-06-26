import csv
import base64
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
TRACKING_OUT = ROOT / "tracking_out"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

JOBS = {}
JOBS_LOCK = threading.Lock()
LIVE_SESSIONS = {}
LIVE_LOCK = threading.Lock()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def safe_path(base, requested):
    target = (base / requested).resolve()
    base_resolved = base.resolve()
    if target == base_resolved or base_resolved in target.parents:
        return target
    raise ValueError("Unsafe path")


def clean_name(value, fallback="upload"):
    stem = Path(value or fallback).stem
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return cleaned or fallback


def list_files(patterns):
    out = []
    for pattern in patterns:
        out.extend(ROOT.glob(pattern))
    ignored = {".git", ".venv", "node_modules", "__pycache__"}
    return sorted(
        {
            p.resolve()
            for p in out
            if p.is_file() and not any(part in ignored for part in p.relative_to(ROOT).parts)
        }
    )


def list_sequences():
    sequences = []
    for img_dir in sorted(DATA_DIR.glob("**/img1")):
        frame_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        frame_count = len(frame_paths)
        if not frame_count:
            continue
        parent = img_dir.parent
        split = parent.parent.name if parent.parent != DATA_DIR else ""
        source_videos = sorted((parent / "source").glob("*.*")) if (parent / "source").exists() else []
        source_video = next((p for p in source_videos if p.suffix.lower() in VIDEO_SUFFIXES), None)
        sequences.append(
            {
                "name": parent.name,
                "path": rel(img_dir),
                "frames": frame_count,
                "split": split,
                "first_frame": rel(frame_paths[0]),
                "preview_frames": [rel(p) for p in frame_paths],
                "source_video": rel(source_video) if source_video else None,
            }
        )
    return sequences


def list_weights():
    reid = [{"name": p.name, "path": rel(p), "size": p.stat().st_size} for p in list_files(["outputs/**/*.pth", "**/*.pth"])]
    yolo = [{"name": p.name, "path": rel(p), "size": p.stat().st_size} for p in list_files(["*.pt", "outputs/**/*.pt", "detection/**/*.pt"])]
    return {"reid": reid, "yolo": yolo}


def read_track_summary(csv_path):
    if not csv_path.exists():
        return {"rows": 0, "ids": 0, "frames": 0, "preview": []}

    rows = 0
    ids = set()
    frames = set()
    preview = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            ids.add(row.get("id", ""))
            frames.add(row.get("frame", ""))
            if len(preview) < 120:
                preview.append(row)
    return {"rows": rows, "ids": len(ids), "frames": len(frames), "preview": preview}


def list_runs():
    runs = []
    if not TRACKING_OUT.exists():
        return runs
    run_dirs = sorted(
        [p for p in TRACKING_OUT.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        video = run_dir / "output.mp4"
        tracks = run_dir / "tracks.csv"
        frames = sorted(run_dir.glob("vis_*.jpg"))
        runs.append(
            {
                "id": run_dir.name,
                "path": rel(run_dir),
                "modified": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(timespec="seconds"),
                "video": rel(video) if video.exists() else None,
                "tracks": rel(tracks) if tracks.exists() else None,
                "first_frame": rel(frames[0]) if frames else None,
                "preview_frames": [rel(p) for p in frames],
                "frame_count": len(frames),
                "summary": read_track_summary(tracks),
            }
        )
    return runs


def save_uploaded_images(files, name="images"):
    image_files = [file for file in files if Path(file["filename"]).suffix.lower() in IMAGE_SUFFIXES]
    if not image_files:
        raise ValueError("At least one image file is required")

    sequence_name = f"{clean_name(name, 'images')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    img_dir = UPLOADS_DIR / sequence_name / "img1"
    img_dir.mkdir(parents=True, exist_ok=False)

    for index, file_info in enumerate(image_files, start=1):
        suffix = Path(file_info["filename"]).suffix.lower() or ".jpg"
        (img_dir / f"{index:06d}{suffix}").write_bytes(file_info["content"])

    return {"name": sequence_name, "path": rel(img_dir), "frames": len(image_files), "split": "uploads"}


def save_uploaded_video(file_info, frame_stride=1, max_frames=300):
    try:
        import cv2
    except ImportError as exc:
        raise ValueError("opencv-python is required for video upload") from exc

    suffix = Path(file_info["filename"]).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported video type: {suffix or 'unknown'}")

    frame_stride = max(1, int(frame_stride))
    max_frames = max(1, int(max_frames))
    base_name = clean_name(file_info["filename"], "video")
    sequence_name = f"{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sequence_dir = UPLOADS_DIR / sequence_name
    source_dir = sequence_dir / "source"
    img_dir = sequence_dir / "img1"
    source_dir.mkdir(parents=True, exist_ok=False)
    img_dir.mkdir(parents=True, exist_ok=False)

    video_path = source_dir / f"{base_name}{suffix}"
    video_path.write_bytes(file_info["content"])

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Video could not be opened")

    read_count = 0
    saved = 0
    try:
        while saved < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            read_count += 1
            if (read_count - 1) % frame_stride != 0:
                continue
            output_path = img_dir / f"{saved + 1:06d}.jpg"
            if not cv2.imwrite(str(output_path), frame):
                raise ValueError("Failed to write extracted frame")
            saved += 1
    finally:
        capture.release()

    if saved == 0:
        raise ValueError("No frames extracted from video")

    return {
        "name": sequence_name,
        "path": rel(img_dir),
        "frames": saved,
        "split": "uploads",
        "video": rel(video_path),
    }


def start_live_session(name="camera"):
    sequence_name = f"{clean_name(name, 'camera')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    img_dir = UPLOADS_DIR / sequence_name / "img1"
    img_dir.mkdir(parents=True, exist_ok=False)
    session = {
        "id": sequence_name,
        "name": sequence_name,
        "path": rel(img_dir),
        "frames": 0,
        "split": "uploads",
        "img_dir": img_dir,
    }
    with LIVE_LOCK:
        LIVE_SESSIONS[sequence_name] = session
    return {k: v for k, v in session.items() if k != "img_dir"}


def save_live_frame(session_id, image_data):
    with LIVE_LOCK:
        session = LIVE_SESSIONS.get(session_id)
    if not session:
        raise ValueError("live session not found")

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    try:
        raw = base64.b64decode(image_data, validate=True)
    except Exception as exc:
        raise ValueError("invalid frame image data") from exc

    frame_no = int(session["frames"]) + 1
    frame_path = session["img_dir"] / f"{frame_no:06d}.jpg"
    frame_path.write_bytes(raw)
    session["frames"] = frame_no
    return {
        "id": session["id"],
        "name": session["name"],
        "path": session["path"],
        "frames": session["frames"],
        "split": session["split"],
        "last_frame": rel(frame_path),
    }


def stop_live_session(session_id):
    with LIVE_LOCK:
        session = LIVE_SESSIONS.pop(session_id, None)
    if not session:
        raise ValueError("live session not found")
    frame_paths = sorted(session["img_dir"].glob("*.*"))
    return {
        "name": session["name"],
        "path": session["path"],
        "frames": len([p for p in frame_paths if p.suffix.lower() in IMAGE_SUFFIXES]),
        "split": session["split"],
    }


def launch_tracking(payload):
    frames_dir = payload.get("frames_dir", "")
    reid_weight = payload.get("reid_weight", "")
    yolo_weight = payload.get("yolo_weight", "best.pt")
    cfg_file = payload.get("cfg_file", "configs/UAV-Swarm/vit_transreid_stride_384.yml")

    if not frames_dir or not reid_weight:
        raise ValueError("frames_dir and reid_weight are required")

    frames_path = safe_path(ROOT, frames_dir)
    reid_path = safe_path(ROOT, reid_weight)
    cfg_path = safe_path(ROOT, cfg_file)
    if yolo_weight == "best.pt" and not (ROOT / "best.pt").exists() and (ROOT / "detection" / "best.pt").exists():
        yolo_weight = "detection/best.pt"
    yolo_path = safe_path(ROOT, yolo_weight) if yolo_weight else ROOT / "best.pt"

    if not frames_path.is_dir():
        raise ValueError(f"frames_dir not found: {frames_dir}")
    if not reid_path.is_file():
        raise ValueError(f"reid_weight not found: {reid_weight}")
    if not cfg_path.is_file():
        raise ValueError(f"cfg_file not found: {cfg_file}")

    run_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = TRACKING_OUT / f"{run_id}.log"
    TRACKING_OUT.mkdir(exist_ok=True)

    command = [
        sys.executable,
        str(ROOT / "test_rt_v2.py"),
        "--frames_dir",
        str(frames_path),
        "--reid_weight",
        str(reid_path),
        "--cfg_file",
        str(cfg_path),
        "--output_dir",
        str(TRACKING_OUT),
        "--fps",
        str(int(payload.get("fps", 25))),
        "--yolo_conf",
        str(float(payload.get("yolo_conf", 0.25))),
        "--yolo_iou",
        str(float(payload.get("yolo_iou", 0.5))),
        "--yolo_imgsz",
        str(int(payload.get("yolo_imgsz", 1280))),
        "--cos_thresh",
        str(float(payload.get("cos_thresh", 0.55))),
        "--iou_thresh",
        str(float(payload.get("iou_thresh", 0.3))),
    ]
    if yolo_weight:
        command.extend(["--yolo_weights", str(yolo_path)])

    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=str(ROOT), stdout=log_handle, stderr=subprocess.STDOUT)
    with JOBS_LOCK:
        JOBS[run_id] = {
            "id": run_id,
            "pid": process.pid,
            "status": "running",
            "started": datetime.now().isoformat(timespec="seconds"),
            "log": rel(log_path),
            "process": process,
            "log_handle": log_handle,
        }
    return run_id


def job_snapshot():
    snapshots = []
    with JOBS_LOCK:
        for job in JOBS.values():
            process = job["process"]
            code = process.poll()
            if code is not None and job["status"] == "running":
                job["status"] = "done" if code == 0 else "error"
                job["returncode"] = code
                job["finished"] = datetime.now().isoformat(timespec="seconds")
                job["log_handle"].close()
            snapshots.append({k: v for k, v in job.items() if k not in {"process", "log_handle"}})
    return snapshots


def json_bytes(payload):
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def media_type(path):
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"
