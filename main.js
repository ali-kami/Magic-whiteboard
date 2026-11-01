// MagicBoard – step 3: draw with the pen gesture (pinch thumb and index).
// Pipeline: webcam -> MediaPipe HandLandmarker -> cursor + pen state ->
// strokes on the whiteboard, plus a small camera preview with the skeleton.

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

// --- Pen gesture tuning (normalized thumb–index distance; 1.0 = hand size) ---
const PEN_ON_THRESHOLD = 0.25; // pinch tighter than this -> pen turns ON
const PEN_OFF_THRESHOLD = 0.35; // open wider than this -> pen turns OFF
const PEN_HOLD_FRAMES = 3; // new state must hold this many frames in a row

// --- Pen look ---
const PEN_COLOR = "#000";
const PEN_WIDTH = 4; // px

// --- Cursor smoothing (One Euro filter) ---
// MIN_CUTOFF: lower = smoother when the hand is slow/still (but more lag).
// BETA: higher = less lag when the hand moves fast (but more jitter).
// D_CUTOFF: smoothing of the speed estimate; rarely needs changing.
const MIN_CUTOFF = 1.0;
const BETA = 0.007;
const D_CUTOFF = 1.0;

// Points trimmed from the start AND the end of every finished stroke, to
// remove the tails caused by the pinch/release moving the cursor.
const TRIM_POINTS = 3;

// If the cursor moves farther than this (px) in one frame, treat it as a
// tracking glitch: end the stroke and start a new one.
const MAX_JUMP_PX = 150;

const MEDIAPIPE_VERSION = "0.10.14";
const MEDIAPIPE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// Landmark indices (MediaPipe numbers the 21 hand points 0–20)
const WRIST = 0;
const THUMB_TIP = 4;
const INDEX_TIP = 8;
const MIDDLE_KNUCKLE = 9; // base of the middle finger
const MIDDLE_TIP = 12;

// Which pairs of landmarks to connect with lines
const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],         // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],         // index
  [5, 9], [9, 10], [10, 11], [11, 12],    // middle
  [9, 13], [13, 14], [14, 15], [15, 16],  // ring
  [13, 17], [17, 18], [18, 19], [19, 20], // pinky
  [0, 17],                                // palm edge
];

// Fingertips that get a big colored dot and a label
const HIGHLIGHTS = [
  { index: THUMB_TIP, color: "red", label: "T" },
  { index: INDEX_TIP, color: "dodgerblue", label: "I" },
  { index: MIDDLE_TIP, color: "limegreen", label: "M" },
];

// The "active zone": the centered part of the camera frame that maps to the
// whole whiteboard. 0.6 = the middle 60% of the width and of the height.
const ACTIVE_ZONE = 0.6;

const CURSOR_DIAMETER = 20; // px, hollow ring while hovering
const PEN_DOT_DIAMETER = 16; // px, solid dot while drawing (slightly smaller)
const CURSOR_COLOR = "#444"; // dark gray

// ---------------------------------------------------------------------------
// Page elements
// ---------------------------------------------------------------------------

const video = document.getElementById("video");
const preview = document.getElementById("preview");
const overlay = document.getElementById("overlay"); // skeleton, on the preview
const overlayCtx = overlay.getContext("2d");
const board = document.getElementById("board"); // whiteboard: strokes are drawn here
const boardCtx = board.getContext("2d");
const cursorCanvas = document.getElementById("cursor");
const cursorCtx = cursorCanvas.getContext("2d");
const messageEl = document.getElementById("message");
const infoEl = document.getElementById("info");
const fpsEl = document.getElementById("fps");
const handEl = document.getElementById("hand");
const distTIEl = document.getElementById("distTI");
const distTMEl = document.getElementById("distTM");
const cursorPosEl = document.getElementById("cursorPos");
const modeEl = document.getElementById("mode");
const smoothingEl = document.getElementById("smoothing");

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

function showMessage(text, isError = false) {
  messageEl.textContent = text;
  messageEl.classList.toggle("error", isError);
  messageEl.hidden = false;
}

function hideMessage() {
  messageEl.hidden = true;
}

// ---------------------------------------------------------------------------
// Hand tracking
// ---------------------------------------------------------------------------

// Load MediaPipe from the CDN and create a HandLandmarker.
// GPU delegate, VIDEO running mode, 1 hand.
async function createHandTracker() {
  // Dynamic import so a CDN failure can be caught and shown on screen
  const { HandLandmarker, FilesetResolver } = await import(
    `${MEDIAPIPE_URL}/vision_bundle.mjs`
  );
  const fileset = await FilesetResolver.forVisionTasks(`${MEDIAPIPE_URL}/wasm`);
  return HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 1,
  });
}

// Ask the browser for the webcam and start the video element.
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  // Make the preview box the same shape as the webcam image
  preview.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
}

// Run the model on the current video frame.
// Returns the 21 landmarks of the first hand, or null if no hand is visible.
// Each landmark is { x, y, z } with x and y between 0 and 1 (fraction of the
// video frame, NOT mirrored).
function detectHand(handTracker, timestampMs) {
  const result = handTracker.detectForVideo(video, timestampMs);
  return result.landmarks.length > 0 ? result.landmarks[0] : null;
}

// Distance between two landmarks, in video pixels.
// We convert from 0–1 to pixels first so the x and y axes are scaled equally
// (the video isn't square, so skipping this would skew the distance).
function pixelDistance(a, b) {
  const dx = (a.x - b.x) * video.videoWidth;
  const dy = (a.y - b.y) * video.videoHeight;
  return Math.hypot(dx, dy);
}

// Hand size = wrist to middle knuckle. Dividing by this makes distances
// roughly the same whether the hand is close to or far from the camera.
function handSize(landmarks) {
  return pixelDistance(landmarks[WRIST], landmarks[MIDDLE_KNUCKLE]);
}

// Normalized distance between two landmarks (1.0 = one "hand size").
function normalizedDistance(landmarks, i, j) {
  return pixelDistance(landmarks[i], landmarks[j]) / handSize(landmarks);
}

// ---------------------------------------------------------------------------
// Cursor
// ---------------------------------------------------------------------------

// Where the cursor is on the whiteboard, in CSS pixels: { x, y }.
// Returns null when no hand is visible.
//
// 1. Take the midpoint of thumb tip and index tip.
// 2. Mirror x (so moving your hand right moves the cursor right).
// 3. Stretch the active zone (middle 60% of the camera frame) over the whole
//    board, so you never need to reach the edges of the camera view.
// 4. Clamp so the cursor stays on screen.
function getCursor(landmarks) {
  if (!landmarks) return null;

  const midX = (landmarks[THUMB_TIP].x + landmarks[INDEX_TIP].x) / 2;
  const midY = (landmarks[THUMB_TIP].y + landmarks[INDEX_TIP].y) / 2;

  const zoneStart = (1 - ACTIVE_ZONE) / 2; // 0.2
  const u = (1 - midX - zoneStart) / ACTIVE_ZONE; // 0 = left edge of zone, 1 = right
  const v = (midY - zoneStart) / ACTIVE_ZONE; // 0 = top edge of zone, 1 = bottom

  return {
    x: clamp(u, 0, 1) * board.clientWidth,
    y: clamp(v, 0, 1) * board.clientHeight,
  };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

// One Euro filter: a low-pass filter whose cutoff rises with speed. Slow
// movement gets heavy smoothing (no jitter); fast movement gets light
// smoothing (no lag). Filters one number; we use one per axis.
class OneEuroFilter {
  constructor(minCutoff, beta, dCutoff) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.reset();
  }

  reset() {
    this.x = null; // last filtered value (null = no history yet)
    this.dx = 0; // last filtered speed
    this.t = 0; // time of last sample, in seconds
  }

  // How much of the new sample to let through, given a cutoff and time step
  static alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  // value: new raw sample; t: its time in seconds. Returns the smoothed value.
  filter(value, t) {
    if (this.x === null) {
      this.x = value;
      this.t = t;
      return value;
    }
    const dt = t - this.t;
    if (dt <= 0) return this.x; // same frame twice: nothing new

    const speed = (value - this.x) / dt;
    this.dx += OneEuroFilter.alpha(this.dCutoff, dt) * (speed - this.dx);

    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x += OneEuroFilter.alpha(cutoff, dt) * (value - this.x);
    this.t = t;
    return this.x;
  }
}

const cursorFilterX = new OneEuroFilter(MIN_CUTOFF, BETA, D_CUTOFF);
const cursorFilterY = new OneEuroFilter(MIN_CUTOFF, BETA, D_CUTOFF);

let smoothingOn = true; // toggled with the 's' key

function resetCursorFilters() {
  cursorFilterX.reset();
  cursorFilterY.reset();
}

// Smooth a raw cursor { x, y } (or null). timestampMs: time of this frame.
// When the hand disappears the filters reset, so when it comes back the cursor
// starts at the new position instead of sliding in from the old one.
// With smoothing off the raw cursor is returned as is.
function smoothCursor(rawCursor, timestampMs) {
  if (!rawCursor) {
    resetCursorFilters();
    return null;
  }
  if (!smoothingOn) return rawCursor;
  const t = timestampMs / 1000;
  return {
    x: cursorFilterX.filter(rawCursor.x, t),
    y: cursorFilterY.filter(rawCursor.y, t),
  };
}

// Tracking-glitch detector. Call once per new camera frame with the RAW
// cursor. Returns true if it moved more than MAX_JUMP_PX since the last frame.
let lastRawCursor = null;
function detectJump(rawCursor) {
  const jumped =
    rawCursor !== null &&
    lastRawCursor !== null &&
    Math.hypot(rawCursor.x - lastRawCursor.x, rawCursor.y - lastRawCursor.y) > MAX_JUMP_PX;
  lastRawCursor = rawCursor;
  return jumped;
}

// Draw the cursor on its own layer. Pass null to just clear it.
//   hovering: hollow dark gray ring
//   drawing:  solid dot in the pen color, slightly smaller
function drawCursor(cursor, penOn) {
  cursorCtx.clearRect(0, 0, cursorCanvas.clientWidth, cursorCanvas.clientHeight);
  if (!cursor) return;

  if (penOn) {
    cursorCtx.fillStyle = PEN_COLOR;
    cursorCtx.beginPath();
    cursorCtx.arc(cursor.x, cursor.y, PEN_DOT_DIAMETER / 2, 0, Math.PI * 2);
    cursorCtx.fill();
    return;
  }

  const lineWidth = 2;
  cursorCtx.strokeStyle = CURSOR_COLOR;
  cursorCtx.lineWidth = lineWidth;
  cursorCtx.beginPath();
  // Subtract half the line width so the outer edge is exactly CURSOR_DIAMETER
  cursorCtx.arc(cursor.x, cursor.y, CURSOR_DIAMETER / 2 - lineWidth / 2, 0, Math.PI * 2);
  cursorCtx.stroke();
}

// ---------------------------------------------------------------------------
// Pen gesture
// ---------------------------------------------------------------------------

const pen = {
  on: false, // current (debounced) pen state
  framesInOtherState: 0, // how many frames in a row want to switch it
};

// Call once per new camera frame. Updates pen.on from the pinch distance.
//
// Hysteresis: turning ON needs a tighter pinch than staying on, so a distance
// hovering around one number can't flicker. Between the two thresholds the pen
// keeps whatever state it has. A switch also has to be wanted for
// PEN_HOLD_FRAMES frames in a row.
function updatePen(landmarks) {
  if (!landmarks) {
    // No hand: pen is off immediately
    pen.on = false;
    pen.framesInOtherState = 0;
    return;
  }

  const distance = normalizedDistance(landmarks, THUMB_TIP, INDEX_TIP);
  const wantsOn = pen.on ? distance <= PEN_OFF_THRESHOLD : distance < PEN_ON_THRESHOLD;

  if (wantsOn === pen.on) {
    pen.framesInOtherState = 0;
  } else if (++pen.framesInOtherState >= PEN_HOLD_FRAMES) {
    pen.on = wantsOn;
    pen.framesInOtherState = 0;
  }
}

// ---------------------------------------------------------------------------
// Strokes and the whiteboard
// ---------------------------------------------------------------------------

// A stroke is an array of points: [{ x, y }, ...] in board pixels.
const strokes = []; // finished strokes
let currentStroke = null; // the stroke being drawn right now, or null
let boardDirty = true; // true when the board needs to be redrawn

// Finished strokes are drawn once onto this offscreen canvas. Each frame the
// board just copies it and draws the current stroke on top, so drawing stays
// fast no matter how many strokes there are. The full stroke list is only
// redrawn onto it after a resize (see rebuildCache).
const cache = document.createElement("canvas");
const cacheCtx = cache.getContext("2d");

// Pen look; set before drawing strokes (resizing a canvas resets its settings)
function applyPenStyle(ctx) {
  ctx.strokeStyle = PEN_COLOR;
  ctx.lineWidth = PEN_WIDTH;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}

// Match the cache to the board's pixel size, then redraw every finished stroke
function rebuildCache() {
  cache.width = board.width;
  cache.height = board.height;
  const dpr = window.devicePixelRatio || 1;
  cacheCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  applyPenStyle(cacheCtx);
  for (const stroke of strokes) drawStroke(cacheCtx, stroke);
}

function startStroke(cursor) {
  currentStroke = [{ x: cursor.x, y: cursor.y }];
  boardDirty = true;
}

// Cut the tails off a finished stroke: drop TRIM_POINTS points from each end.
// If fewer than 2 points would be left, the stroke was a tap or a tiny mark,
// so keep it as a single dot (at its middle point) instead of losing it.
function trimStroke(stroke) {
  if (stroke.length - 2 * TRIM_POINTS >= 2) {
    return stroke.slice(TRIM_POINTS, stroke.length - TRIM_POINTS);
  }
  return [stroke[Math.floor(stroke.length / 2)]];
}

// While a stroke is in progress all its points are drawn; trimming only
// happens here, when it is finished.
function endStroke() {
  if (currentStroke) {
    const finished = trimStroke(currentStroke);
    strokes.push(finished);
    drawStroke(cacheCtx, finished); // add it to the cache, once
    currentStroke = null;
    boardDirty = true;
  }
}

function clearBoard() {
  strokes.length = 0;
  cacheCtx.clearRect(0, 0, cache.width, cache.height); // (overshoots because of the dpr transform; harmless)
  currentStroke = null; // if the pen is still on, a new stroke starts next frame
  boardDirty = true;
}

// Call once per new camera frame, after updatePen().
// Pen on -> start/extend a stroke. Pen off or hand lost -> end it.
// jumped: the raw cursor made a tracking-glitch jump this frame (see
// detectJump), so the current stroke ends and a new one starts.
function updateStroke(cursor, jumped) {
  if (jumped) endStroke(); // don't draw a long stray line

  if (!pen.on || !cursor) {
    endStroke();
    return;
  }

  if (!currentStroke) {
    startStroke(cursor);
  } else {
    currentStroke.push({ x: cursor.x, y: cursor.y });
    boardDirty = true;
  }
}

// Trace one stroke as a smooth line. Each point acts as a control point of a
// quadratic curve that ends at the midpoint to the next point, which rounds
// off the corners between samples.
function drawStroke(ctx, stroke) {
  if (stroke.length === 1) {
    // A single tap: just a dot
    ctx.fillStyle = PEN_COLOR;
    ctx.beginPath();
    ctx.arc(stroke[0].x, stroke[0].y, PEN_WIDTH / 2, 0, Math.PI * 2);
    ctx.fill();
    return;
  }

  ctx.beginPath();
  ctx.moveTo(stroke[0].x, stroke[0].y);
  for (let i = 1; i < stroke.length - 1; i++) {
    const midX = (stroke[i].x + stroke[i + 1].x) / 2;
    const midY = (stroke[i].y + stroke[i + 1].y) / 2;
    ctx.quadraticCurveTo(stroke[i].x, stroke[i].y, midX, midY);
  }
  const end = stroke[stroke.length - 1];
  ctx.lineTo(end.x, end.y);
  ctx.stroke();
}

// Update the board (only when something changed): the cached finished
// strokes, plus the stroke in progress on top.
function drawBoard() {
  if (!boardDirty) return;
  boardDirty = false;

  const w = board.clientWidth;
  const h = board.clientHeight;
  boardCtx.clearRect(0, 0, w, h);
  boardCtx.drawImage(cache, 0, 0, w, h);

  applyPenStyle(boardCtx);
  if (currentStroke) drawStroke(boardCtx, currentStroke); // show it as it grows
}

// ---------------------------------------------------------------------------
// Camera preview drawing
// ---------------------------------------------------------------------------

// Match a canvas's pixel size to its on-screen size (and screen sharpness),
// so we can draw in CSS pixels.
function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  canvas.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
}

function resizeCanvases() {
  fitCanvas(board);
  fitCanvas(cursorCanvas);
  fitCanvas(overlay);
  rebuildCache(); // resizing wipes a canvas, so redraw the strokes
  boardDirty = true;
}

// Convert a landmark (0–1 in the video frame) to a position on the preview.
// The preview box has the same shape as the video, so this is just a scale,
// plus flipping x to match the mirrored video.
function toPreview(landmark) {
  return {
    x: (1 - landmark.x) * overlay.clientWidth,
    y: landmark.y * overlay.clientHeight,
  };
}

// Faint rectangle showing the active zone, so you know where to keep your hand
function drawActiveZone() {
  const w = overlay.clientWidth;
  const h = overlay.clientHeight;
  const margin = (1 - ACTIVE_ZONE) / 2;
  overlayCtx.strokeStyle = "rgba(255, 255, 255, 0.55)";
  overlayCtx.lineWidth = 1.5;
  overlayCtx.setLineDash([6, 4]);
  overlayCtx.strokeRect(w * margin, h * margin, w * ACTIVE_ZONE, h * ACTIVE_ZONE);
  overlayCtx.setLineDash([]);
}

function drawHand(landmarks) {
  const ctx = overlayCtx;
  const points = landmarks.map(toPreview);

  // Connecting lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (const [a, b] of CONNECTIONS) {
    ctx.moveTo(points[a].x, points[a].y);
    ctx.lineTo(points[b].x, points[b].y);
  }
  ctx.stroke();

  // All 21 landmarks as small dots
  ctx.fillStyle = "yellow";
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  // Bigger colored fingertips with labels
  ctx.font = "bold 12px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (const { index, color, label } of HIGHLIGHTS) {
    const p = points[index];
    ctx.fillStyle = color;
    ctx.strokeStyle = "white";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Label with a dark outline so it's readable on any background
    ctx.lineWidth = 3;
    ctx.strokeStyle = "black";
    ctx.strokeText(label, p.x + 11, p.y);
    ctx.fillStyle = "white";
    ctx.fillText(label, p.x + 11, p.y);
  }
}

// Redraw the whole preview overlay: active zone, then the hand if there is one
function drawPreviewOverlay(landmarks) {
  overlayCtx.clearRect(0, 0, overlay.clientWidth, overlay.clientHeight);
  drawActiveZone();
  if (landmarks) drawHand(landmarks);
}

// ---------------------------------------------------------------------------
// Info box
// ---------------------------------------------------------------------------

function updateInfoBox(fps, landmarks, cursor) {
  fpsEl.textContent = fps;
  if (landmarks) {
    handEl.textContent = "yes";
    distTIEl.textContent = normalizedDistance(landmarks, THUMB_TIP, INDEX_TIP).toFixed(2);
    distTMEl.textContent = normalizedDistance(landmarks, THUMB_TIP, MIDDLE_TIP).toFixed(2);
    cursorPosEl.textContent = `${Math.round(cursor.x)}, ${Math.round(cursor.y)}`;
    modeEl.textContent = pen.on ? "PEN" : "HOVER";
  } else {
    handEl.textContent = "no";
    distTIEl.textContent = "–";
    distTMEl.textContent = "–";
    cursorPosEl.textContent = "–";
    modeEl.textContent = "–";
  }
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

function startLoop(handTracker) {
  let lastVideoTime = -1;
  let frameCount = 0;
  let fps = 0;
  let fpsWindowStart = performance.now();
  let landmarks = null;
  let cursor = null;

  function tick() {
    const now = performance.now();

    // Only run the model (and the pen/stroke logic) when the webcam has
    // produced a new frame, so "3 frames in a row" means 3 camera frames
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      landmarks = detectHand(handTracker, now);
      const rawCursor = getCursor(landmarks);
      // Glitches are checked on the RAW cursor: smoothing would soften a
      // sudden jump and could let it slip under the limit
      const jumped = detectJump(rawCursor);
      if (jumped) resetCursorFilters(); // new position, no sliding from the old one
      cursor = smoothCursor(rawCursor, now);
      updatePen(landmarks);
      updateStroke(cursor, jumped);
      frameCount++;
    }

    // FPS = processed frames per second, refreshed twice a second
    if (now - fpsWindowStart >= 500) {
      fps = Math.round((frameCount * 1000) / (now - fpsWindowStart));
      frameCount = 0;
      fpsWindowStart = now;
    }

    drawBoard();
    drawCursor(cursor, pen.on);
    if (!preview.classList.contains("hidden")) drawPreviewOverlay(landmarks);
    updateInfoBox(fps, landmarks, cursor);

    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------

async function main() {
  resizeCanvases();
  window.addEventListener("resize", resizeCanvases);

  // Keys: 'c' shows/hides the camera preview, 'x' clears the board,
  // 's' turns cursor smoothing on/off
  window.addEventListener("keydown", (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const key = event.key.toLowerCase();
    if (key === "c") preview.classList.toggle("hidden");
    if (key === "x") clearBoard();
    if (key === "s") {
      smoothingOn = !smoothingOn;
      resetCursorFilters(); // start fresh so there's no jump when switching
      smoothingEl.textContent = smoothingOn ? "on" : "off";
    }
  });

  // 1. Camera
  showMessage("Requesting camera access…");
  try {
    await startCamera();
    resizeCanvases(); // preview shape may have changed
  } catch (err) {
    console.error(err);
    const denied = err.name === "NotAllowedError" || err.name === "SecurityError";
    showMessage(
      denied
        ? "Camera access was denied. Allow the camera for this page in your browser settings, then reload."
        : `Could not start the camera: ${err.message || err.name}`,
      true
    );
    return;
  }

  // 2. Hand tracking model
  showMessage("Loading hand tracking model…");
  let handTracker;
  try {
    handTracker = await createHandTracker();
  } catch (err) {
    console.error(err);
    showMessage(
      `Could not load MediaPipe hand tracking (check your internet connection and that your browser supports WebGPU/WebGL): ${err.message || err}`,
      true
    );
    return;
  }

  // 3. Go
  hideMessage();
  infoEl.hidden = false;
  startLoop(handTracker);
}

main();
