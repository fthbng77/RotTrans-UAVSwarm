import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from runner import (
    ROOT,
    job_snapshot,
    json_bytes,
    launch_tracking,
    list_runs,
    list_sequences,
    list_weights,
    media_type,
    safe_path,
    save_uploaded_images,
    save_uploaded_video,
    save_live_frame,
    start_live_session,
    stop_live_session,
)


HTML = r"""<!doctype html>
<html lang="tr">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>RotTrans UAVSwarm UI</title>
    <style>
      :root {
        --bg: #f5f7fb;
        --panel: #ffffff;
        --soft: #edf2f7;
        --line: #d7dee8;
        --text: #17202a;
        --muted: #667085;
        --accent: #0f766e;
        --accent-dark: #115e59;
        --dark: #101828;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        background: var(--bg);
        color: var(--text);
        font-family: Arial, Helvetica, sans-serif;
      }
      button, input, select { font: inherit; }
      button { cursor: pointer; }
      button:disabled { cursor: not-allowed; opacity: .6; }
      .shell {
        display: grid;
        grid-template-columns: minmax(330px, 410px) 1fr;
        min-height: 100vh;
      }
      .sidebar {
        display: flex;
        flex-direction: column;
        gap: 16px;
        padding: 20px;
        border-right: 1px solid var(--line);
        background: #eef3f8;
      }
      .brand { display: flex; align-items: center; gap: 12px; min-height: 50px; }
      .mark {
        display: grid;
        width: 44px;
        height: 44px;
        place-items: center;
        border-radius: 8px;
        background: var(--accent);
        color: white;
        font-weight: 800;
      }
      h1, h2, p { margin: 0; }
      h1 { font-size: 18px; }
      h2 { font-size: 16px; }
      .muted, .brand p, label, .status, .runItem span, .job { color: var(--muted); }
      .panel {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--panel);
        padding: 16px;
      }
      .sectionTitle {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 14px;
      }
      .tabs, .form, .runs, .uploadBox { display: flex; flex-direction: column; gap: 12px; }
      .tabs { flex-direction: row; }
      .tab {
        flex: 1;
        min-height: 36px;
        border: 1px solid var(--line);
        border-radius: 6px;
        background: white;
        color: var(--text);
      }
      .tab.active { border-color: var(--accent); background: #ecfdf9; color: var(--accent-dark); font-weight: 700; }
      label {
        display: flex;
        flex-direction: column;
        gap: 6px;
        font-size: 12px;
        font-weight: 700;
      }
      input, select {
        min-height: 40px;
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 6px;
        background: white;
        color: var(--text);
        padding: 0 10px;
      }
      input[type="file"] { padding: 8px; }
      input:focus, select:focus { border-color: var(--accent); outline: 2px solid rgba(15,118,110,.16); }
      .grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
      .primary, .secondary, .ghost {
        min-height: 40px;
        border: 0;
        border-radius: 6px;
        padding: 0 14px;
        font-weight: 700;
      }
      .primary { background: var(--accent); color: white; }
      .primary:hover { background: var(--accent-dark); }
      .secondary { background: #334155; color: white; }
      .secondary:hover { background: #1f2937; }
      .ghost { min-height: 32px; background: var(--soft); color: var(--text); }
      .ghost:hover { background: #dfe8f1; }
      .status { min-height: 18px; margin-top: 10px; font-size: 13px; overflow-wrap: anywhere; }
      .runItem {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 5px;
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 6px;
        background: white;
        padding: 11px;
        text-align: left;
      }
      .runItem.active, .runItem:hover { border-color: var(--accent); background: #ecfdf9; }
      .runItem strong { max-width: 100%; overflow-wrap: anywhere; font-size: 13px; }
      .runItem span { font-size: 12px; }
      .workspace { display: flex; min-width: 0; flex-direction: column; gap: 16px; padding: 20px; }
      .topbar { display: flex; align-items: center; justify-content: flex-end; gap: 16px; min-height: 40px; }
      .jobs { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; }
      .job { border: 1px solid var(--line); border-radius: 999px; background: white; padding: 6px 10px; font-size: 12px; }
      .job.running { color: #b54708; border-color: #f7b267; }
      .job.done { color: #047857; border-color: #86efac; }
      .job.error { color: #b42318; border-color: #fda29b; }
      .viewer {
        display: grid;
        min-height: 420px;
        place-items: center;
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--dark);
      }
      .viewer video, .viewer img { display: block; max-width: 100%; max-height: 70vh; }
      .placeholder { color: #cbd5e1; padding: 24px; text-align: center; }
      .player {
        position: relative;
        display: grid;
        width: 100%;
        height: 100%;
        min-height: 420px;
        background: var(--dark);
      }
      .playerStage {
        position: relative;
        display: grid;
        width: 100%;
        height: 100%;
        border: 0;
        background: transparent;
        padding: 0;
        min-height: 360px;
        place-items: center;
        overflow: hidden;
        cursor: pointer;
      }
      .mediaFrame {
        position: relative;
        display: grid;
        width: fit-content;
        height: fit-content;
        max-width: 100%;
        max-height: 68vh;
        align-self: center;
        justify-self: center;
        place-items: center;
      }
      .mediaFrame img,
      .mediaFrame video {
        display: block;
        max-width: 100%;
        max-height: 68vh;
        object-fit: contain;
      }
      .playHint {
        position: absolute;
        left: 50%;
        top: 50%;
        display: grid;
        width: 58px;
        height: 58px;
        place-items: center;
        border: 1px solid rgba(255,255,255,.16);
        border-radius: 999px;
        background: rgba(15,23,42,.78);
        line-height: 1;
        opacity: .92;
        pointer-events: none;
        transform: translate(-50%, -50%);
        transition: opacity .16s ease, transform .16s ease;
      }
      .playHint::before {
        content: "";
        width: 0;
        height: 0;
        margin-left: 4px;
        border-top: 11px solid transparent;
        border-bottom: 11px solid transparent;
        border-left: 17px solid #e5edf6;
      }
      .player.playing .playHint {
        opacity: 0;
        transform: translate(-50%, -50%) scale(.92);
      }
      .player:not(.playing) .mediaFrame:hover .playHint,
      .player:not(.playing) .playerStage:focus-visible .playHint {
        opacity: .92;
        transform: translate(-50%, -50%) scale(1);
      }
      .cameraPreview {
        display: grid;
        min-height: 260px;
        place-items: center;
        overflow: hidden;
        border-radius: 8px;
        background: #101828;
      }
      .cameraPreview video {
        display: block;
        width: 100%;
        max-height: 320px;
        object-fit: contain;
      }
      .liveStatus {
        color: #cbd5e1;
        font-size: 12px;
        white-space: nowrap;
      }
      .metrics { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
      .metric { border: 1px solid var(--line); border-radius: 8px; background: white; padding: 16px; }
      .metric span { display: block; font-size: 26px; font-weight: 800; }
      .metric small { color: var(--muted); }
      .tableWrap { overflow: auto; border: 1px solid var(--line); border-radius: 8px; background: white; }
      table { width: 100%; border-collapse: collapse; font-size: 13px; }
      th, td { border-bottom: 1px solid var(--line); padding: 10px 12px; text-align: left; white-space: nowrap; }
      th { background: var(--soft); color: var(--muted); font-size: 12px; }
      @media (max-width: 980px) {
        .shell { grid-template-columns: 1fr; }
        .sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
        .topbar { align-items: flex-start; flex-direction: column; }
        .jobs { justify-content: flex-start; }
      }
      @media (max-width: 560px) {
        .sidebar, .workspace { padding: 12px; }
        .grid2, .metrics { grid-template-columns: 1fr; }
        .viewer { min-height: 260px; }
        .player, .playerStage { min-height: 260px; }
      }
    </style>
  </head>
  <body>
    <main class="shell">
      <aside class="sidebar">
        <header class="brand">
          <div class="mark">RT</div>
          <div>
            <h1>RotTrans UAVSwarm</h1>
            <p>Tracking console</p>
          </div>
        </header>

        <section class="panel">
          <div class="sectionTitle">
            <h2>Veri ekle</h2>
            <button id="refreshBtn" class="ghost" type="button">Yenile</button>
          </div>
          <div class="tabs">
            <button id="imageTab" class="tab active" type="button">Resim</button>
            <button id="videoTab" class="tab" type="button">Video</button>
            <button id="cameraTab" class="tab" type="button">Kamera</button>
          </div>
          <div id="imageUpload" class="uploadBox">
            <label>
              Resimler
              <input id="imageFiles" type="file" accept="image/*" multiple />
            </label>
            <label>
              Sequence adi
              <input id="imageName" type="text" value="images" />
            </label>
            <button id="uploadImagesBtn" class="secondary" type="button">Resimleri ekle</button>
          </div>
          <div id="videoUpload" class="uploadBox" style="display:none">
            <label>
              Video
              <input id="videoFile" type="file" accept="video/mp4,video/avi,video/quicktime,video/x-matroska,video/webm" />
            </label>
            <div class="grid2">
              <label>
                Frame araligi
                <input id="frameStride" type="number" min="1" max="120" step="1" value="1" />
              </label>
              <label>
                Maks frame
                <input id="maxFrames" type="number" min="1" max="10000" step="1" value="300" />
              </label>
            </div>
            <button id="uploadVideoBtn" class="secondary" type="button">Videoyu ekle</button>
          </div>
          <div id="cameraUpload" class="uploadBox" style="display:none">
            <div class="cameraPreview">
              <video id="cameraVideo" autoplay muted playsinline></video>
            </div>
            <div class="grid2">
              <label>
                Kamera FPS
                <input id="cameraFps" type="number" min="1" max="10" step="1" value="2" />
              </label>
              <label>
                Sequence adi
                <input id="cameraName" type="text" value="camera" />
              </label>
            </div>
            <button id="startCameraBtn" class="secondary" type="button">Kamerayi baslat</button>
            <button id="stopCameraBtn" class="ghost" type="button" disabled>Kamerayi durdur</button>
          </div>
          <p id="status" class="status"></p>
        </section>

        <section class="panel">
          <div class="sectionTitle">
            <h2>Yeni takip</h2>
            <span id="activeJobs" class="muted">0 aktif job</span>
          </div>
          <form id="runForm" class="form">
            <label>
              Frame dizini
              <select id="framesDir"></select>
            </label>
            <label>
              ReID agirligi
              <select id="reidWeight"></select>
            </label>
            <label>
              YOLO agirligi
              <select id="yoloWeight"></select>
            </label>
            <div class="grid2">
              <label>YOLO conf <input id="yoloConf" type="number" min="0" max="1" step="0.01" value="0.25" /></label>
              <label>Cos thresh <input id="cosThresh" type="number" min="0" max="1" step="0.01" value="0.55" /></label>
              <label>IoU thresh <input id="iouThresh" type="number" min="0" max="1" step="0.01" value="0.30" /></label>
              <label>FPS <input id="fps" type="number" min="1" max="120" step="1" value="25" /></label>
            </div>
            <button id="runBtn" class="primary" type="submit">Takibi baslat</button>
          </form>
        </section>

        <section class="panel">
          <div class="sectionTitle"><h2>Calismalar</h2></div>
          <div id="runsList" class="runs"></div>
        </section>
      </aside>

      <section class="workspace">
        <header class="topbar">
          <div id="jobs" class="jobs"></div>
        </header>
        <section id="viewer" class="viewer">
          <div class="placeholder">Cikti secildiginde video veya frame burada gorunur.</div>
        </section>
        <section class="metrics">
          <div class="metric"><span id="metricFrames">0</span><small>frame</small></div>
          <div class="metric"><span id="metricIds">0</span><small>ID</small></div>
        </section>
        <section class="tableWrap">
          <table>
            <thead>
              <tr><th>Frame</th><th>ID</th><th>x1</th><th>y1</th><th>x2</th><th>y2</th><th>Score</th></tr>
            </thead>
            <tbody id="tracksBody"></tbody>
          </table>
        </section>
      </section>
    </main>

    <script>
      const state = {
        runs: [],
        sequences: [],
        selectedRun: null,
        playerTimer: null,
        playerFrames: [],
        playerIndex: 0,
        previewMode: "sequence",
        imagePreviewUrls: [],
        cameraStream: null,
        cameraTimer: null,
        liveSession: null,
        frameCache: new Map(),
        jobs: [],
        pendingJobId: null,
      };
      const $ = (id) => document.getElementById(id);

      async function jsonFetch(url, options) {
        const response = await fetch(url, options);
        const payload = await response.json().catch(() => null);
        if (!response.ok) throw new Error(payload?.error || `${url} failed`);
        return payload;
      }

      function media(path) {
        if (!path) return "";
        const encoded = String(path).split("/").map(encodeURIComponent).join("/");
        return `/media/${encoded}`;
      }

      function frameSrc(path) {
        if (!path) return "";
        if (path.startsWith("blob:") || path.startsWith("data:") || path.startsWith("http")) return path;
        return media(path);
      }

      function option(select, label, value) {
        const node = document.createElement("option");
        node.textContent = label;
        node.value = value;
        select.appendChild(node);
      }

      function setStatus(message) {
        $("status").textContent = message || "";
      }

      function stopFramePlayer() {
        if (state.playerTimer) {
          window.clearInterval(state.playerTimer);
          state.playerTimer = null;
        }
      }

      function clearImagePreviewUrls() {
        state.imagePreviewUrls.forEach((url) => URL.revokeObjectURL(url));
        state.imagePreviewUrls = [];
      }

      function preloadFrame(path) {
        const src = frameSrc(path);
        if (!src) return null;
        if (state.frameCache.has(src)) return state.frameCache.get(src);
        const image = new Image();
        image.src = src;
        state.frameCache.set(src, image);
        return image;
      }

      function setPlayerFrame(index) {
        if (!state.playerFrames.length) return;
        state.playerIndex = Math.max(0, Math.min(index, state.playerFrames.length - 1));
        const img = $("framePlayerImage");
        if (img) {
          const requestedSrc = frameSrc(state.playerFrames[state.playerIndex]);
          const loaded = preloadFrame(state.playerFrames[state.playerIndex]);
          img.dataset.pendingSrc = requestedSrc;
          const showLoaded = () => {
            if (img.dataset.pendingSrc === requestedSrc) img.src = requestedSrc;
          };
          if (loaded?.complete && loaded.naturalWidth > 0) {
            showLoaded();
          } else if (loaded) {
            loaded.onload = showLoaded;
            loaded.onerror = () => {
              if (img.dataset.pendingSrc === requestedSrc) {
                $("viewer").innerHTML = '<div class="placeholder">Onizleme gorseli yuklenemedi. Dosya yolu veya cikti klasoru kontrol edilmeli.</div>';
              }
            };
          }
          for (let offset = 1; offset <= 3; offset += 1) {
            preloadFrame(state.playerFrames[state.playerIndex + offset]);
          };
        }
      }

      function playFramePlayer() {
        stopFramePlayer();
        $("framePlayer")?.classList.add("playing");
        state.playerTimer = window.setInterval(() => {
          const next = state.playerIndex + 1 >= state.playerFrames.length ? 0 : state.playerIndex + 1;
          setPlayerFrame(next);
        }, Math.round(1000 / 24));
      }

      function toggleFramePlayer() {
        if (state.playerTimer) {
          stopFramePlayer();
          $("framePlayer")?.classList.remove("playing");
        } else {
          playFramePlayer();
        }
      }

      function toggleVideoPlayer() {
        const video = $("videoPlayer");
        if (!video) return;
        if (video.paused || video.ended) {
          video.play().catch((error) => setStatus(error.message));
        } else {
          video.pause();
        }
      }

      function syncVideoState() {
        const video = $("videoPlayer");
        const player = $("videoClickPlayer");
        if (!video || !player) return;
        player.classList.toggle("playing", !video.paused && !video.ended);
      }

      function renderVideoPlayer(path, title, fallbackFrames = []) {
        stopFramePlayer();
        state.frameCache.clear();
        state.playerFrames = [];
        state.playerIndex = 0;
        const poster = fallbackFrames?.length ? ` poster="${frameSrc(fallbackFrames[0])}"` : "";
        $("viewer").innerHTML = `
          <div id="videoClickPlayer" class="player" aria-label="${title}">
            <button class="playerStage" type="button" aria-label="Onizlemeyi oynat veya durdur">
              <span class="mediaFrame">
                <video id="videoPlayer" src="${media(path)}"${poster} preload="metadata" playsinline></video>
                <span class="playHint" aria-hidden="true"></span>
              </span>
            </button>
          </div>
        `;
        const video = $("videoPlayer");
        $("videoClickPlayer").addEventListener("click", toggleVideoPlayer);
        video.addEventListener("play", syncVideoState);
        video.addEventListener("pause", syncVideoState);
        video.addEventListener("ended", syncVideoState);
        video.addEventListener("timeupdate", syncVideoState);
        video.addEventListener("loadedmetadata", syncVideoState);
        video.addEventListener("error", () => {
          if (fallbackFrames?.length) {
            setStatus("Video onizlemesi acilamadi; frame onizlemesine gecildi.");
            renderFramePlayer(fallbackFrames, title);
          } else {
            $("viewer").innerHTML = '<div class="placeholder">Video onizlemesi acilamadi.</div>';
          }
        });
      }

      function renderFramePlayer(frames, title) {
        stopFramePlayer();
        state.frameCache.clear();
        state.playerFrames = frames || [];
        state.playerIndex = 0;
        if (!state.playerFrames.length) {
          $("viewer").innerHTML = '<div class="placeholder">Bu kayitta oynatilacak frame yok.</div>';
          return;
        }
        $("viewer").innerHTML = `
          <div id="framePlayer" class="player" aria-label="${title}">
            <button class="playerStage" type="button" aria-label="Onizlemeyi oynat veya durdur">
              <span class="mediaFrame">
                <img id="framePlayerImage" alt="${title}" />
                <span class="playHint" aria-hidden="true"></span>
              </span>
            </button>
          </div>
        `;
        $("framePlayer").addEventListener("click", toggleFramePlayer);
        setPlayerFrame(0);
      }

      function showUpload(type) {
        $("imageUpload").style.display = type === "image" ? "flex" : "none";
        $("videoUpload").style.display = type === "video" ? "flex" : "none";
        $("cameraUpload").style.display = type === "camera" ? "flex" : "none";
        $("imageTab").classList.toggle("active", type === "image");
        $("videoTab").classList.toggle("active", type === "video");
        $("cameraTab").classList.toggle("active", type === "camera");
      }

      async function loadInputs(keepSelection = true) {
        const selectedFrame = $("framesDir").value;
        const selectedReid = $("reidWeight").value;
        const selectedYolo = $("yoloWeight").value;
        const [sequences, weights] = await Promise.all([
          jsonFetch("/api/sequences"),
          jsonFetch("/api/weights"),
        ]);
        state.sequences = sequences;

        $("framesDir").innerHTML = "";
        sequences.forEach((seq) => option($("framesDir"), `${seq.name} - ${seq.frames} frame`, seq.path));
        if (keepSelection && selectedFrame) $("framesDir").value = selectedFrame;

        $("reidWeight").innerHTML = "";
        weights.reid.forEach((weight) => option($("reidWeight"), weight.path, weight.path));
        if (keepSelection && selectedReid) $("reidWeight").value = selectedReid;

        $("yoloWeight").innerHTML = "";
        option($("yoloWeight"), "best.pt", "best.pt");
        weights.yolo.forEach((weight) => option($("yoloWeight"), weight.path, weight.path));
        if (keepSelection && selectedYolo) $("yoloWeight").value = selectedYolo;
        previewSelectedSequence();
      }

      function previewSelectedSequence() {
        if (state.selectedRun) return;
        clearImagePreviewUrls();
        const sequence = state.sequences.find((item) => item.path === $("framesDir").value);
        if (!sequence) return;
        state.previewMode = "sequence";
        $("metricFrames").textContent = sequence.frames || 0;
        $("metricIds").textContent = 0;
        $("tracksBody").innerHTML = "";
        if (sequence.source_video) {
          renderVideoPlayer(sequence.source_video, sequence.name);
        } else {
          renderFramePlayer(sequence.preview_frames || [], sequence.name);
        }
      }

      function previewSelectedImages() {
        const files = Array.from($("imageFiles").files || []).filter((file) => file.type.startsWith("image/"));
        clearImagePreviewUrls();
        state.selectedRun = null;
        state.previewMode = "images";
        renderRuns();

        if (!files.length) {
          state.previewMode = "sequence";
          previewSelectedSequence();
          return;
        }

        state.imagePreviewUrls = files.map((file) => URL.createObjectURL(file));
        $("metricFrames").textContent = files.length;
        $("metricIds").textContent = 0;
        $("tracksBody").innerHTML = "";
        renderFramePlayer(state.imagePreviewUrls, "Secilen resimler");
        setStatus(`${files.length} resim secildi. Onizleme hazir.`);
      }

      function renderRuns() {
        const list = $("runsList");
        list.innerHTML = "";
        if (!state.runs.length) {
          list.innerHTML = '<p class="muted">Henuz cikti yok.</p>';
          return;
        }
        state.runs.forEach((run) => {
          const button = document.createElement("button");
          button.type = "button";
          button.className = `runItem ${state.selectedRun?.id === run.id ? "active" : ""}`;
          button.innerHTML = `<strong>${run.id}</strong><span>${run.frame_count} frame - ${run.summary.ids} ID - ${run.summary.rows} satir</span>`;
          button.addEventListener("click", () => selectRun(run));
          list.appendChild(button);
        });
      }

      function selectRun(run) {
        stopFramePlayer();
        clearImagePreviewUrls();
        state.selectedRun = run;
        state.previewMode = "run";
        $("metricFrames").textContent = run.summary.frames || run.frame_count || 0;
        $("metricIds").textContent = run.summary.ids || 0;

        if (run.video) {
          renderVideoPlayer(run.video, run.id, run.preview_frames || []);
        } else if (run.preview_frames && run.preview_frames.length) {
          renderFramePlayer(run.preview_frames, run.id);
        } else if (run.first_frame) {
          $("viewer").innerHTML = `<img src="${media(run.first_frame)}" alt="${run.id} preview" />`;
        } else {
          $("viewer").innerHTML = '<div class="placeholder">Bu calismada onizleme dosyasi yok.</div>';
        }

        const body = $("tracksBody");
        body.innerHTML = "";
        run.summary.preview.forEach((row) => {
          const tr = document.createElement("tr");
          tr.innerHTML = `<td>${row.frame || ""}</td><td>${row.id || ""}</td><td>${row.x1 || ""}</td><td>${row.y1 || ""}</td><td>${row.x2 || ""}</td><td>${row.y2 || ""}</td><td>${row.score || ""}</td>`;
          body.appendChild(tr);
        });
        renderRuns();
      }

      async function loadRuns() {
        state.runs = await jsonFetch("/api/runs");
        renderRuns();
        if (state.previewMode === "run" && state.selectedRun) {
          const refreshed = state.runs.find((run) => run.id === state.selectedRun.id);
          if (refreshed) state.selectedRun = refreshed;
        }
      }

      async function loadJobs() {
        const jobs = await jsonFetch("/api/jobs");
        state.jobs = jobs;
        $("activeJobs").textContent = `${jobs.filter((job) => job.status === "running").length} aktif job`;
        $("jobs").innerHTML = jobs.slice(0, 4).map((job) => `<span class="job ${job.status}">${job.id}: ${job.status}</span>`).join("");
        const pending = jobs.find((job) => job.id === state.pendingJobId);
        if (pending?.status === "done") {
          state.pendingJobId = null;
          await loadRuns();
          if (state.runs.length) selectRun(state.runs[0]);
          setStatus(`${pending.id} tamamlandi. Onizleme hazir.`);
        } else if (pending?.status === "error") {
          state.pendingJobId = null;
          setStatus(`${pending.id} hata ile bitti. Log dosyasini kontrol et.`);
        }
      }

      async function uploadImages() {
        const files = $("imageFiles").files;
        if (!files.length) return setStatus("Once en az bir resim sec.");
        const body = new FormData();
        Array.from(files).forEach((file) => body.append("images", file));
        body.append("name", $("imageName").value || "images");
        setStatus("Resimler yukleniyor...");
        const sequence = await jsonFetch("/api/uploads/images", { method: "POST", body });
        await loadInputs(false);
        $("framesDir").value = sequence.path;
        state.selectedRun = null;
        state.previewMode = "sequence";
        clearImagePreviewUrls();
        $("imageFiles").value = "";
        previewSelectedSequence();
        setStatus(`${sequence.frames} resim eklendi: ${sequence.name}`);
      }

      async function uploadVideo() {
        const file = $("videoFile").files[0];
        if (!file) return setStatus("Once bir video sec.");
        const body = new FormData();
        body.append("video", file);
        body.append("frame_stride", $("frameStride").value);
        body.append("max_frames", $("maxFrames").value);
        setStatus("Video yukleniyor ve frame'lere ayriliyor...");
        const sequence = await jsonFetch("/api/uploads/video", { method: "POST", body });
        await loadInputs(false);
        $("framesDir").value = sequence.path;
        state.selectedRun = null;
        state.previewMode = "sequence";
        previewSelectedSequence();
        setStatus(`${sequence.frames} frame hazirlandi: ${sequence.name}`);
      }

      async function startCamera() {
        if (!navigator.mediaDevices?.getUserMedia) {
          setStatus("Tarayici kamera erisimini desteklemiyor.");
          return;
        }

        state.liveSession = await jsonFetch("/api/live/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: $("cameraName").value || "camera" }),
        });

        state.cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        $("cameraVideo").srcObject = state.cameraStream;
        state.selectedRun = null;
        state.previewMode = "camera";
        $("viewer").innerHTML = `
          <div class="player">
            <div class="playerStage">
              <video id="mainCameraVideo" autoplay muted playsinline style="display:block;max-width:100%;max-height:68vh;object-fit:contain;"></video>
            </div>
            <div class="playerControls">
              <span class="liveStatus">Canli kamera</span>
            </div>
          </div>
        `;
        $("mainCameraVideo").srcObject = state.cameraStream;
        $("tracksBody").innerHTML = "";
        $("metricFrames").textContent = 0;
        $("metricIds").textContent = 0;
        $("startCameraBtn").disabled = true;
        $("stopCameraBtn").disabled = false;
        setStatus(`Kamera basladi: ${state.liveSession.name}`);

        const canvas = document.createElement("canvas");
        const context = canvas.getContext("2d");
        const capture = async () => {
          const video = $("cameraVideo");
          if (!video.videoWidth || !video.videoHeight || !state.liveSession) return;
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          context.drawImage(video, 0, 0, canvas.width, canvas.height);
          const image = canvas.toDataURL("image/jpeg", 0.82);
          const result = await jsonFetch("/api/live/frame", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: state.liveSession.id, image }),
          });
          $("metricFrames").textContent = result.frames || 0;
        };

        const fps = Math.max(1, Math.min(10, Number($("cameraFps").value || 2)));
        state.cameraTimer = window.setInterval(() => {
          capture().catch((error) => setStatus(error.message));
        }, Math.round(1000 / fps));
      }

      async function stopCamera() {
        if (state.cameraTimer) {
          window.clearInterval(state.cameraTimer);
          state.cameraTimer = null;
        }
        if (state.cameraStream) {
          state.cameraStream.getTracks().forEach((track) => track.stop());
          state.cameraStream = null;
        }
        $("cameraVideo").srcObject = null;
        $("startCameraBtn").disabled = false;
        $("stopCameraBtn").disabled = true;

        if (!state.liveSession) {
          setStatus("Kamera durduruldu.");
          return;
        }

        const sequence = await jsonFetch("/api/live/stop", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: state.liveSession.id }),
        });
        state.liveSession = null;
        await loadInputs(false);
        $("framesDir").value = sequence.path;
        state.selectedRun = null;
        state.previewMode = "sequence";
        previewSelectedSequence();
        setStatus(`${sequence.frames} kamera frame'i kaydedildi: ${sequence.name}`);
      }

      async function runTracking(event) {
        event.preventDefault();
        const payload = {
          frames_dir: $("framesDir").value,
          reid_weight: $("reidWeight").value,
          yolo_weight: $("yoloWeight").value,
          yolo_conf: Number($("yoloConf").value),
          cos_thresh: Number($("cosThresh").value),
          iou_thresh: Number($("iouThresh").value),
          fps: Number($("fps").value),
        };
        $("runBtn").disabled = true;
        setStatus("Takip baslatiliyor...");
        try {
          const result = await jsonFetch("/api/runs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          setStatus(`${result.job_id} basladi`);
          state.pendingJobId = result.job_id;
          await Promise.all([loadJobs(), loadRuns()]);
        } finally {
          $("runBtn").disabled = false;
        }
      }

      async function refreshAll() {
        await Promise.all([loadInputs(), loadRuns(), loadJobs()]);
      }

      function bind() {
        $("imageTab").addEventListener("click", () => showUpload("image"));
        $("videoTab").addEventListener("click", () => showUpload("video"));
        $("cameraTab").addEventListener("click", () => showUpload("camera"));
        $("imageFiles").addEventListener("change", previewSelectedImages);
        $("startCameraBtn").addEventListener("click", () => startCamera().catch((error) => setStatus(error.message)));
        $("stopCameraBtn").addEventListener("click", () => stopCamera().catch((error) => setStatus(error.message)));
        $("framesDir").addEventListener("change", () => {
          state.selectedRun = null;
          state.previewMode = "sequence";
          renderRuns();
          previewSelectedSequence();
        });
        $("uploadImagesBtn").addEventListener("click", () => uploadImages().catch((error) => setStatus(error.message)));
        $("uploadVideoBtn").addEventListener("click", () => uploadVideo().catch((error) => setStatus(error.message)));
        $("refreshBtn").addEventListener("click", () => refreshAll().catch((error) => setStatus(error.message)));
        $("runForm").addEventListener("submit", (event) => runTracking(event).catch((error) => setStatus(error.message)));
      }

      bind();
      refreshAll().catch((error) => setStatus(error.message));
      window.setInterval(() => {
        loadJobs().catch((error) => setStatus(error.message));
        loadRuns().catch((error) => setStatus(error.message));
      }, 4000);
    </script>
  </body>
</html>
"""


def parse_multipart(handler):
    content_type = handler.headers.get("Content-Type", "")
    match = re.search(r'boundary="?([^";]+)"?', content_type)
    if not match:
        raise ValueError("multipart boundary not found")

    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        raise ValueError("empty upload")

    boundary = ("--" + match.group(1)).encode("utf-8")
    body = handler.rfile.read(length)
    fields = {}
    files = {}

    for part in body.split(boundary):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue

        header_blob, _, content = part.partition(b"\r\n\r\n")
        if not header_blob:
            continue

        headers = header_blob.decode("utf-8", errors="replace").split("\r\n")
        disposition = next((h for h in headers if h.lower().startswith("content-disposition:")), "")
        name_match = re.search(r'name="([^"]+)"', disposition)
        if not name_match:
            continue

        name = name_match.group(1)
        filename_match = re.search(r'filename="([^"]*)"', disposition)
        if filename_match:
            files.setdefault(name, []).append(
                {
                    "filename": filename_match.group(1),
                    "content": content.rstrip(b"\r\n"),
                }
            )
        else:
            fields[name] = content.rstrip(b"\r\n").decode("utf-8", errors="replace")

    return fields, files


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def send_json(self, payload, status=200):
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def bad_request(self, message, status=400):
        self.send_json({"error": message}, status=status)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/sequences":
            return self.send_json(list_sequences())
        if path == "/api/weights":
            return self.send_json(list_weights())
        if path == "/api/runs":
            return self.send_json(list_runs())
        if path == "/api/jobs":
            return self.send_json(job_snapshot())
        if path.startswith("/media/"):
            return self.serve_media(path.removeprefix("/media/"))

        return self.bad_request("Not found", status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            if path == "/api/uploads/images":
                fields, files = parse_multipart(self)
                sequence = save_uploaded_images(files.get("images", []), fields.get("name", "images"))
                return self.send_json(sequence, status=201)

            if path == "/api/uploads/video":
                fields, files = parse_multipart(self)
                video_files = files.get("video", [])
                if not video_files:
                    raise ValueError("video file is required")
                sequence = save_uploaded_video(
                    video_files[0],
                    frame_stride=fields.get("frame_stride", "1"),
                    max_frames=fields.get("max_frames", "300"),
                )
                return self.send_json(sequence, status=201)

            if path == "/api/runs":
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                run_id = launch_tracking(payload)
                return self.send_json({"job_id": run_id, "status": "running"}, status=202)

            if path == "/api/live/start":
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                session = start_live_session(payload.get("name", "camera"))
                return self.send_json(session, status=201)

            if path == "/api/live/frame":
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                frame = save_live_frame(payload.get("session_id", ""), payload.get("image", ""))
                return self.send_json(frame, status=201)

            if path == "/api/live/stop":
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                sequence = stop_live_session(payload.get("session_id", ""))
                return self.send_json(sequence)

            return self.bad_request("Not found", status=404)
        except Exception as exc:
            return self.bad_request(str(exc), status=400)

    def serve_media(self, requested):
        try:
            target = safe_path(ROOT, requested)
        except ValueError:
            return self.bad_request("Unsafe path", status=403)

        if not target.exists() or not target.is_file():
            return self.bad_request("Not found", status=404)

        file_size = target.stat().st_size
        start = 0
        end = file_size - 1
        status = 200
        range_header = self.headers.get("Range", "")
        if range_header.startswith("bytes="):
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if not match:
                return self.bad_request("Invalid range", status=416)
            start_text, end_text = match.groups()
            if start_text:
                start = int(start_text)
                end = int(end_text) if end_text else end
            elif end_text:
                suffix_length = int(end_text)
                start = max(file_size - suffix_length, 0)
            end = min(end, file_size - 1)
            if start > end or start >= file_size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
            status = 206

        content_length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", media_type(target))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(content_length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        try:
            with target.open("rb") as handle:
                handle.seek(start)
                remaining = content_length
                while True:
                    if remaining <= 0:
                        break
                    chunk = handle.read(min(1024 * 512, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def main():
    port = int(os.environ.get("ROTTRANS_UI_PORT", "7860"))
    server = ReusableThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"RotTrans UI: http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nRotTrans UI stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
