"use strict";
const $ = (id) => document.getElementById(id);
const maxBytes = 64 * 1024 * 1024;
const supported = /\.(jpe?g|png|webp|mp4|mov|avi|webm|mkv)$/i;
let previewUrl;
let busy = false;
const meanings = {
  synthetic_signals: ["Synthetic signals detected", "The model found signals associated with generated images. Review the original source and context."],
  low_synthetic_signals: ["Low synthetic signals", "The model found few synthetic signals in the analyzed crops. This does not establish authenticity."],
  inconclusive: ["Inconclusive", "Evidence is incomplete, scores fall between review bands, or sampled frames disagree. Review the face coverage and source."]
};
function validate(file) {
  if (!file) throw new Error("Choose a media file first.");
  if (!supported.test(file.name)) throw new Error("Choose a supported image or video file.");
  if (file.size === 0) throw new Error("The file is empty.");
  if (file.size > maxBytes) throw new Error("The file exceeds 64 MiB.");
}
function showError(message) { $("error").textContent = message; $("error").hidden = false; }
function selectFile() {
  $("error").hidden = true;
  $("result").hidden = true;
  $("preview").replaceChildren(); $("preview").hidden = true;
  if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = undefined; }
  const file = $("fileInput").files[0];
  $("fileLabel").textContent = file ? file.name : "Choose a file to inspect";
  if (!file) return;
  try {
    validate(file); previewUrl = URL.createObjectURL(file);
    const el = document.createElement(/\.(jpe?g|png|webp)$/i.test(file.name) ? "img" : "video");
    el.src = previewUrl;
    if (el.tagName === "VIDEO") { el.controls = true; el.preload = "metadata"; } else el.alt = "Selected media preview";
    $("preview").append(el); $("preview").hidden = false;
  } catch (error) { showError(error.message); }
}
function send(form, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", form.action);
    xhr.timeout = 180000;
    xhr.upload.onprogress = (event) => { if (event.lengthComputable && onProgress) onProgress(Math.round(event.loaded / event.total * 100)); };
    xhr.upload.onload = () => { if (onProgress) onProgress(100); };
    xhr.onload = () => {
      let data;
      try { data = JSON.parse(xhr.responseText); } catch { reject(new Error("The server returned an unreadable response.")); return; }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new Error(data.error || "The request failed."));
    };
    xhr.onerror = () => reject(new Error("Connection failed. Check that the server is running and retry."));
    xhr.ontimeout = () => reject(new Error("The request timed out. The server may still be processing it; wait before retrying."));
    xhr.send(new FormData(form));
  });
}
function renderResult(data) {
  if (!meanings[data.verdict] || !Array.isArray(data.frames)) throw new Error("The analysis result is incomplete.");
  $("verdict").textContent = meanings[data.verdict][0];
  $("resultMeaning").textContent = meanings[data.verdict][1];
  $("score").textContent = data.synthetic_score.toFixed(3);
  $("facts").replaceChildren();
  const fields = [["Receipt ID", data.receipt_id], ["SHA-256", data.file_sha256], ["Sampled / inferred", `${data.frames.length} / ${data.unique_frames_analyzed}`], ["Processing time", `${data.processing_seconds.toFixed(2)} s`]];
  for (const [title, value] of fields) {
    const row = document.createElement("div"), dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = title; dd.textContent = value; row.append(dt, dd); $("facts").append(row);
  }
  $("frames").replaceChildren();
  for (const frame of data.frames) {
    const row = document.createElement("tr");
    for (const value of [frame.index + 1, frame.timestamp_seconds.toFixed(3), frame.generated_image_score.toFixed(3), frame.face_manipulation_score === null ? "Not assessed" : frame.face_manipulation_score.toFixed(3), frame.face_coverage.replaceAll("_", " "), frame.synthetic_score.toFixed(3), frame.duplicate_of === null ? "Analyzed" : `Identical to frame ${frame.duplicate_of + 1}`]) {
      const td = document.createElement("td"); td.textContent = value; row.append(td);
    }
    $("frames").append(row);
  }
  $("limitations").replaceChildren();
  for (const text of data.limitations) { const li = document.createElement("li"); li.textContent = text; $("limitations").append(li); }
  $("modelDetails").textContent = JSON.stringify({generated_image_model:data.model, facial_specialist:data.specialist_model, coverage_complete:data.coverage_complete, aggregation:data.aggregation}, null, 2);
  $("receiptLink").href = `/receipts/${encodeURIComponent(data.receipt_id)}`;
  $("receiptId").value = data.receipt_id;
  $("result").hidden = false; $("resultTitle").focus();
}
if ($("uploadForm")) {
  $("fileInput").addEventListener("change", selectFile);
  for (const eventName of ["dragover", "drop"]) $("dropzone").addEventListener(eventName, (event) => {
    event.preventDefault();
    if (eventName === "drop" && !busy) {
      if (event.dataTransfer.files.length !== 1) { showError("Drop exactly one file."); return; }
      $("fileInput").files = event.dataTransfer.files; selectFile();
    }
  });
  $("uploadForm").addEventListener("submit", async (event) => {
    event.preventDefault(); if (busy) return;
    $("error").hidden = true; $("result").hidden = true;
    try { validate($("fileInput").files[0]); } catch (error) { showError(error.message); return; }
    busy = true; $("analyzeBtn").disabled = true; $("fileInput").disabled = false;
    $("progress").hidden = false; $("uploadProgress").value = 0;
    $("analyzeBtn").textContent = "Analyzing…";
    try {
      const response = send($("uploadForm"), (percent) => {
        $("uploadProgress").value = percent;
        $("progressText").textContent = percent < 100 ? `Uploading ${percent}%` : "Upload complete. Decoding and analyzing sampled frames…";
      });
      $("fileInput").disabled = true;
      renderResult(await response);
    } catch (error) { showError(error.message); }
    finally { busy = false; $("analyzeBtn").disabled = false; $("fileInput").disabled = false; $("analyzeBtn").textContent = "Analyze media ↗"; $("progress").hidden = true; }
  });
  $("verifyForm").addEventListener("submit", async (event) => {
    event.preventDefault(); const button = event.currentTarget.querySelector("button"); button.disabled = true;
    $("verifyResult").textContent = "Comparing file bytes…";
    try {
      validate($("verifyFile").files[0]);
      const data = await send($("verifyForm"));
      $("verifyResult").textContent = data.file_matches ? "Exact file match. The local receipt passed its integrity check. This does not prove authenticity." : "File mismatch. These bytes differ from the file in the receipt.";
    } catch (error) { $("verifyResult").textContent = error.message; }
    finally { button.disabled = false; }
  });
  fetch("/health").then(async (response) => {
    const data = await response.json();
    $("modelStatus").textContent = data.model_available ? "Analyzer ready" : "Model unavailable";
  }).catch(() => { $("modelStatus").textContent = "Server unavailable"; });
}
window.addEventListener("pagehide", () => { if (previewUrl) URL.revokeObjectURL(previewUrl); });
