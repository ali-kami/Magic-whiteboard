// MagicBoard – step 9: eraser (open palm) and clear-all (double fist pump).
// Pipeline: webcam -> MediaPipe HandLandmarker -> cursor + gesture mode + hand
// size (depth) -> strokes on the whiteboard, plus a small camera preview with
// the skeleton. Keys: c h b z x s f (see the cheat sheet in the info box).

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

// --- Pen gesture tuning (normalized thumb–index distance; 1.0 = hand size) ---
// Gesture modes: HOVER (no pinch), PEN (thumb touches index), HIGHLIGHT (thumb
// touches index AND middle). Each "touch" uses hysteresis: it starts below the
// ON threshold and only ends above the OFF threshold.
const PEN_ON_THRESHOLD = 0.25; // thumb-index below this -> touching
const PEN_OFF_THRESHOLD = 0.35; // thumb-index above this -> not touching
const HL_ON_THRESHOLD = 0.25; // thumb-middle below this -> touching
const HL_OFF_THRESHOLD = 0.35; // thumb-middle above this -> not touching
const MODE_HOLD_FRAMES = 3; // a new mode must hold this many frames in a row

// --- Eraser (open palm: all 5 fingers extended and spread wide) ---
const ERASER_HOLD_MS = 400; // hold the pose this long before erasing starts
const ERASER_RADIUS = 40; // px
// Spread = normalized distance between index tip and pinky tip (watch the
// "Spread" line in the info box). It has to reach THRESHOLD to start erasing
// and only has to stay above RELEASE to keep erasing (so it doesn't flicker).
const ERASER_SPREAD_THRESHOLD = 1.2;
const ERASER_SPREAD_RELEASE = 1.05;
const ERASER_COLOR = "#ff3b30";

// --- Clear all (double fist pump: open, fist, open, fist) ---
// Fist = all four fingertips this close to the palm center, in hand sizes
// (watch the "Fist" line in the info box; lower = tighter fist needed).
const FIST_CURL_THRESHOLD = 0.75;
const PUMP_WINDOW_MS = 1500; // open-fist-open-fist must finish within this
const POSE_HOLD_FRAMES = 2; // an open/fist pose must hold this many frames to count
const CLEAR_MESSAGE_MS = 2000; // how long "Cleared. Press z to undo" shows

const UNDO_LIMIT = 200; // how many actions 'z' can step back through

// A stroke cut short by a direct PEN <-> HIGHLIGHT switch is deleted if it has
// fewer than this many points (the stray mark made while the fingers settle)
const SHORT_STROKE_POINTS = 10;

// --- Highlighter look ---
const HIGHLIGHT_COLOR = "#ffe600"; // yellow
const HIGHLIGHT_ALPHA = 0.35; // 35% opacity
const HIGHLIGHT_WIDTH_FACTOR = 3; // width = this x the current thickness level

// --- Board colors (toggle with 'b') ---
// Strokes don't store a color: they are drawn in the theme's "ink", so
// switching the theme recolors everything already on the board.
const DARK_BOARD_DEFAULT = true; // black board + white ink looks best on a cheap projector
const THEMES = {
  dark: { ink: "#fff", ring: "#ccc" }, // ring = hover cursor color
  light: { ink: "#000", ring: "#444" },
};
// (the board background color itself is in style.css: body.dark)

// --- Pen thickness from hand distance to the camera ---
// Depth signal = hand size in video pixels (bigger = closer). Watch the "Hand
// size" line in the info box and set the range to what you see:
//   HAND_SIZE_FAR  = smoothed size with your hand as far as you'd draw  -> thinnest
//   HAND_SIZE_NEAR = smoothed size with your hand as close as you'd draw -> thickest
const HAND_SIZE_FAR = 100; // px
const HAND_SIZE_NEAR = 260; // px
const SIZE_SMOOTHING = 0.15; // 0–1; per-frame weight of the new size. Lower = steadier but slower
const THICKNESS_LEVELS = [2, 4, 8, 14, 22]; // px, thinnest (far) to thickest (near)
// How far past a level boundary (as a fraction of one level) the hand must go
// before the level changes. Stops flicker when you sit on a boundary.
const LEVEL_HYSTERESIS = 0.2;
// When the pen turns on, lock to the thickness level from this many frames
// earlier (pinching nudges your hand and would otherwise change the level).
// Only used when LIVE_THICKNESS is false.
const THICKNESS_LOCK_LOOKBACK = 6; // frames
// false: thickness is locked when a stroke starts (whole stroke is one width)
// true:  thickness keeps following your hand while drawing (width varies)
const LIVE_THICKNESS = false;

// --- Cursor smoothing (One Euro filter) ---
// MIN_CUTOFF: lower = smoother when the hand is slow/still (but more lag).
// BETA: higher = less lag when the hand moves fast (but more jitter).
// D_CUTOFF: smoothing of the speed estimate; rarely needs changing.
const MIN_CUTOFF = 0.5;
const BETA = 0.003;
const D_CUTOFF = 1.0;

// --- Stabilizer ("lazy brush"), toggle with 'v' ---
// The point that actually draws is tied to the cursor by an invisible string
// this long (px). It only moves when the cursor pulls the string tight, so
// shakes smaller than this are ignored. Bigger = steadier but more lag, and
// the line trails the cursor by up to this much.
const STABILIZER_RADIUS = 12;
const STABILIZER_COLOR = "#ff5252"; // the drawing point and string, shown while drawing

// --- Webcam resolution to request (the browser may give something else) ---
const CAMERA_WIDTH = 1280;
const CAMERA_HEIGHT = 720;

// --- Stroke cleanup ---
// Moving-average passes over a finished stroke's points (3 points wide).
// 0 = off; more passes = smoother but rounder corners.
const CLEANUP_PASSES = 1;

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
const THUMB_IP = 3; // the thumb's middle joint
const THUMB_TIP = 4;
const INDEX_TIP = 8;
const MIDDLE_KNUCKLE = 9; // base of the middle finger
const MIDDLE_TIP = 12;
const PINKY_TIP = 20;

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
// whole whiteboard. 0.8 = the middle 80% of the width and of the height.
// Bigger = easier to reach the edges, but each hand movement covers more board
// (less precise). Smaller = the opposite.
const ACTIVE_ZONE = 0.8;

// The cursor matches the pen thickness: a hollow ring while hovering, a solid
// dot while drawing. Minimums keep thin pens visible.
const CURSOR_MIN_SIZE = 12; // px, smallest hover ring diameter. Raise it to see the cursor better on a wall
const DOT_MIN_DIAMETER = 4; // px, smallest drawing dot

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
const spreadEl = document.getElementById("spread");
const fistEl = document.getElementById("fist");
const toastEl = document.getElementById("toast");
const stabilizerEl = document.getElementById("stabilizer");
const cameraEl = document.getElementById("camera");
const handSizeEl = document.getElementById("handSize");
const thicknessEl = document.getElementById("thickness");

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
    video: { width: { ideal: CAMERA_WIDTH }, height: { ideal: CAMERA_HEIGHT } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  // Show the resolution we actually got (the camera may not support the one we asked for)
  const got = `${video.videoWidth}x${video.videoHeight}`;
  const asked = `${CAMERA_WIDTH}x${CAMERA_HEIGHT}`;
  cameraEl.textContent = got === asked ? got : `${got} (asked ${asked})`;

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
// Hand depth -> pen thickness
// ---------------------------------------------------------------------------

const INDEX_KNUCKLE = 5;
const PINKY_KNUCKLE = 17;

// Hand size in video pixels, used as a depth signal (bigger = closer to the
// camera). Average of palm length (wrist to middle knuckle) and palm width
// (index knuckle to pinky knuckle); turning your hand changes one a bit but
// not the other, so the average is steadier than either alone.
function handDepthSize(landmarks) {
  return (
    (pixelDistance(landmarks[WRIST], landmarks[MIDDLE_KNUCKLE]) +
      pixelDistance(landmarks[INDEX_KNUCKLE], landmarks[PINKY_KNUCKLE])) /
    2
  );
}

let handSizeRaw = null; // this frame's hand size (px), null = no hand
let handSizeSmoothed = null; // exponential moving average of it

// Call once per new camera frame.
function updateHandSize(landmarks) {
  if (!landmarks) {
    handSizeRaw = null;
    handSizeSmoothed = null; // start fresh when the hand comes back
    return;
  }
  handSizeRaw = handDepthSize(landmarks);
  handSizeSmoothed =
    handSizeSmoothed === null
      ? handSizeRaw
      : handSizeSmoothed + SIZE_SMOOTHING * (handSizeRaw - handSizeSmoothed);
}

let thicknessLevel = null; // index into THICKNESS_LEVELS, null until a hand is seen
let thickness = THICKNESS_LEVELS[1]; // current pen thickness in px

// Recent hover levels, one per frame, oldest first (at most
// THICKNESS_LOCK_LOOKBACK long). Pinching moves your hand a little and can
// change the level, so when the pen turns on we lock to the level from a few
// frames earlier, before the pinch began.
const levelHistory = [];
let wasDrawing = false;

// Map the smoothed hand size to one of the thickness levels.
// Call once per new camera frame, after updateMode() and updateHandSize().
function updateThickness() {
  const drawingJustStarted = isDrawing() && !wasDrawing;
  wasDrawing = isDrawing();

  if (handSizeSmoothed === null) {
    levelHistory.length = 0; // hand lost: old levels are stale
    return;
  }

  // Drawing just started: lock to the level from THICKNESS_LOCK_LOOKBACK
  // frames ago (the oldest one in the history)
  if (drawingJustStarted && !LIVE_THICKNESS && levelHistory.length > 0) {
    thicknessLevel = levelHistory[0];
    thickness = THICKNESS_LEVELS[thicknessLevel];
    return;
  }

  // Locked: while a stroke is being drawn the thickness stays as it was
  // when the stroke started (unless LIVE_THICKNESS is on)
  if (isDrawing() && !LIVE_THICKNESS) return;

  const count = THICKNESS_LEVELS.length;
  const t = clamp((handSizeSmoothed - HAND_SIZE_FAR) / (HAND_SIZE_NEAR - HAND_SIZE_FAR), 0, 1);
  const position = t * count; // 0 .. count; each level is 1 wide

  // Snap to a level, and only move off the current one once we're clearly
  // past its edge
  if (
    thicknessLevel === null ||
    position >= thicknessLevel + 1 + LEVEL_HYSTERESIS ||
    position < thicknessLevel - LEVEL_HYSTERESIS
  ) {
    thicknessLevel = Math.min(Math.floor(position), count - 1);
  }
  thickness = THICKNESS_LEVELS[thicknessLevel];

  levelHistory.push(thicknessLevel);
  if (levelHistory.length > THICKNESS_LOCK_LOOKBACK) levelHistory.shift();
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

  const zoneStart = (1 - ACTIVE_ZONE) / 2; // 0.1 for an 80% zone
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
//   hovering:  hollow ring, as wide as the pen thickness (at least CURSOR_MIN_SIZE)
//   pen:       solid dot in the ink color, as wide as the stroke
//   highlight: solid yellow dot, as wide as the (3x) highlighter stroke
//   erase:     red circle of ERASER_RADIUS
// While waiting to erase (eraseProgress 0..1) a red ring fills up around the ring.
// While drawing with the stabilizer on, drawingPoint ({x, y}) is shown as a
// small dot joined to the cursor by a faint line.
function drawCursor(cursor, mode, penThickness, drawingPoint, eraseProgress) {
  cursorCtx.clearRect(0, 0, cursorCanvas.clientWidth, cursorCanvas.clientHeight);
  if (!cursor) return;

  if (mode === MODE_ERASE) {
    // Red circle: everything inside it is erased
    cursorCtx.fillStyle = ERASER_COLOR;
    cursorCtx.globalAlpha = 0.15;
    cursorCtx.beginPath();
    cursorCtx.arc(cursor.x, cursor.y, ERASER_RADIUS, 0, Math.PI * 2);
    cursorCtx.fill();
    cursorCtx.globalAlpha = 1;
    cursorCtx.strokeStyle = ERASER_COLOR;
    cursorCtx.lineWidth = 3;
    cursorCtx.stroke();
    return;
  }

  if (mode === MODE_PEN || mode === MODE_HIGHLIGHT) {
    // Stabilizer: the string from the cursor to the point that is drawing
    if (drawingPoint) {
      cursorCtx.strokeStyle = STABILIZER_COLOR;
      cursorCtx.globalAlpha = 0.5;
      cursorCtx.lineWidth = 1.5;
      cursorCtx.beginPath();
      cursorCtx.moveTo(cursor.x, cursor.y);
      cursorCtx.lineTo(drawingPoint.x, drawingPoint.y);
      cursorCtx.stroke();
      cursorCtx.globalAlpha = 1;
    }

    const highlight = mode === MODE_HIGHLIGHT;
    const width = penThickness * (highlight ? HIGHLIGHT_WIDTH_FACTOR : 1);
    cursorCtx.fillStyle = highlight ? HIGHLIGHT_COLOR : currentTheme().ink;
    cursorCtx.globalAlpha = highlight ? 0.7 : 1; // see-through enough to read, but visible
    cursorCtx.beginPath();
    cursorCtx.arc(cursor.x, cursor.y, Math.max(width, DOT_MIN_DIAMETER) / 2, 0, Math.PI * 2);
    cursorCtx.fill();
    cursorCtx.globalAlpha = 1;

    // Small dot where the line is actually being drawn
    if (drawingPoint) {
      cursorCtx.fillStyle = STABILIZER_COLOR;
      cursorCtx.beginPath();
      cursorCtx.arc(drawingPoint.x, drawingPoint.y, 3, 0, Math.PI * 2);
      cursorCtx.fill();
    }
    return;
  }

  const lineWidth = 2;
  const diameter = Math.max(penThickness, CURSOR_MIN_SIZE);
  cursorCtx.strokeStyle = currentTheme().ring;
  cursorCtx.lineWidth = lineWidth;
  cursorCtx.beginPath();
  // Subtract half the line width so the outer edge is exactly `diameter`
  cursorCtx.arc(cursor.x, cursor.y, diameter / 2 - lineWidth / 2, 0, Math.PI * 2);
  cursorCtx.stroke();

  // Waiting for the eraser to start: a red ring fills up around the cursor
  if (eraseProgress !== null) {
    cursorCtx.strokeStyle = ERASER_COLOR;
    cursorCtx.lineWidth = 4;
    cursorCtx.globalAlpha = 0.3;
    cursorCtx.beginPath();
    cursorCtx.arc(cursor.x, cursor.y, ERASER_RADIUS, 0, Math.PI * 2);
    cursorCtx.stroke();
    cursorCtx.globalAlpha = 1;
    cursorCtx.beginPath();
    cursorCtx.arc(cursor.x, cursor.y, ERASER_RADIUS, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.min(eraseProgress, 1));
    cursorCtx.stroke();
  }
}

// ---------------------------------------------------------------------------
// Hand pose: open palm, fist
// ---------------------------------------------------------------------------

// The four long fingers: tip landmark and middle joint (PIP) landmark
const FINGERS = [
  { tip: 8, joint: 6 }, // index
  { tip: 12, joint: 10 }, // middle
  { tip: 16, joint: 14 }, // ring
  { tip: 20, joint: 18 }, // pinky
];

// A finger is extended when its tip is farther from the wrist than its
// middle joint (curled fingers fold the tip back toward the wrist).
function isExtended(landmarks, tip, joint) {
  return (
    pixelDistance(landmarks[WRIST], landmarks[tip]) >
    pixelDistance(landmarks[WRIST], landmarks[joint])
  );
}

// Average of the wrist and the four finger knuckles: roughly the palm center
function palmCenter(landmarks) {
  const ids = [WRIST, INDEX_KNUCKLE, MIDDLE_KNUCKLE, 13, PINKY_KNUCKLE];
  return {
    x: ids.reduce((sum, i) => sum + landmarks[i].x, 0) / ids.length,
    y: ids.reduce((sum, i) => sum + landmarks[i].y, 0) / ids.length,
  };
}

// Latest pose readings (null when no hand), refreshed by updatePose():
//   allExtended: all 5 fingers extended
//   spread:      index tip to pinky tip, in hand sizes
//   curl:        the farthest fingertip from the palm center, in hand sizes
//   fist:        all four fingertips curled close to the palm
//   open:        the four fingers extended (thumb ignored)
let pose = null;

// Call once per new camera frame
function updatePose(landmarks) {
  if (!landmarks) {
    pose = null;
    return;
  }
  const fourExtended = FINGERS.every((f) => isExtended(landmarks, f.tip, f.joint));
  const center = palmCenter(landmarks);
  const size = handSize(landmarks);
  const curl = Math.max(...FINGERS.map((f) => pixelDistance(landmarks[f.tip], center) / size));
  const fist = curl <= FIST_CURL_THRESHOLD;
  pose = {
    allExtended: fourExtended && isExtended(landmarks, THUMB_TIP, THUMB_IP),
    spread: normalizedDistance(landmarks, INDEX_TIP, PINKY_TIP),
    curl,
    fist,
    open: fourExtended && !fist,
  };
}

// ---------------------------------------------------------------------------
// Clear-all gesture: open, fist, open, fist
// ---------------------------------------------------------------------------

let stablePose = null; // "open" | "fist" | "other" once it has held POSE_HOLD_FRAMES
let candidatePose = null;
let candidateCount = 0;
const pumpSteps = []; // recent stable open/fist poses: [{ pose, time }], newest last

// True while an open/fist sequence is under way. The eraser can't start then,
// so the open-hand moments between pumps never trigger it.
function pumpInProgress(now) {
  return pumpSteps.length >= 2 && now - pumpSteps[0].time <= PUMP_WINDOW_MS;
}

// Call once per new camera frame. Returns true on the frame the double pump
// is completed.
function updatePump(now) {
  if (!pose) {
    stablePose = candidatePose = null;
    candidateCount = 0;
    pumpSteps.length = 0;
    return false;
  }

  const raw = pose.fist ? "fist" : pose.open ? "open" : "other";
  if (raw === candidatePose) candidateCount++;
  else {
    candidatePose = raw;
    candidateCount = 1;
  }
  if (candidateCount < POSE_HOLD_FRAMES || raw === stablePose) return false;
  stablePose = raw;
  if (raw === "other") return false; // in-between poses don't count

  const last = pumpSteps[pumpSteps.length - 1];
  if (last && last.pose === raw) return false;
  pumpSteps.push({ pose: raw, time: now });
  if (pumpSteps.length > 4) pumpSteps.shift();

  // Last four stable poses alternate, so open first means open, fist, open, fist
  if (
    pumpSteps.length === 4 &&
    pumpSteps[0].pose === "open" &&
    now - pumpSteps[0].time <= PUMP_WINDOW_MS
  ) {
    pumpSteps.length = 0;
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Gesture mode: HOVER / PEN / HIGHLIGHT / ERASE
// ---------------------------------------------------------------------------

const MODE_HOVER = "HOVER";
const MODE_PEN = "PEN";
const MODE_HIGHLIGHT = "HIGHLIGHT";
const MODE_ERASE = "ERASE";

const gesture = {
  mode: MODE_HOVER, // current (debounced) mode
  candidate: null, // the mode we want to switch to, if any
  candidateFrames: 0, // how many frames in a row we have wanted it
  eraseHoldStart: null, // when the open-palm pose began (ms), while waiting to erase
  eraseProgress: null, // 0..1 fill of the waiting ring, null when not waiting
  eraseLostFrames: 0, // frames in a row the open-palm pose has been missing
};

// PEN and HIGHLIGHT draw; HOVER and ERASE don't
function isDrawing() {
  return gesture.mode === MODE_PEN || gesture.mode === MODE_HIGHLIGHT;
}

function resetEraseHold() {
  gesture.eraseHoldStart = null;
  gesture.eraseProgress = null;
  gesture.eraseLostFrames = 0;
}

// Call once per new camera frame, after updatePose(). Updates gesture.mode:
//   open palm held for ERASER_HOLD_MS (from HOVER only) -> ERASE
//   thumb touches index and middle                      -> HIGHLIGHT
//   thumb touches index only                            -> PEN
//   otherwise                                           -> HOVER
//
// Pinch hysteresis: starting to touch needs a tighter pinch than keeping the
// touch, so a distance hovering around one number can't flicker. The mode only
// switches after the new mode has been wanted for MODE_HOLD_FRAMES frames in
// a row. (Your index usually lands first on a 3-finger pinch, so you will often
// pass through PEN on the way to HIGHLIGHT; updateStroke cleans that up.)
function updateMode(landmarks, now) {
  if (!landmarks || !pose) {
    // No hand: back to HOVER immediately
    gesture.mode = MODE_HOVER;
    gesture.candidate = null;
    gesture.candidateFrames = 0;
    resetEraseHold();
    return;
  }

  // --- Erasing: keep going while the palm stays open and spread ---
  if (gesture.mode === MODE_ERASE) {
    const holding = pose.allExtended && pose.spread >= ERASER_SPREAD_RELEASE;
    gesture.eraseLostFrames = holding ? 0 : gesture.eraseLostFrames + 1;
    if (gesture.eraseLostFrames >= MODE_HOLD_FRAMES) {
      gesture.mode = MODE_HOVER;
      resetEraseHold();
    }
    return;
  }

  // --- Waiting to erase: only from HOVER, never during a stroke ---
  const canStartErase =
    gesture.mode === MODE_HOVER && gesture.candidate === null && !pumpInProgress(now);
  const palmPose = pose.allExtended && pose.spread >= ERASER_SPREAD_THRESHOLD;
  if (!canStartErase) {
    resetEraseHold();
  } else if (palmPose) {
    gesture.eraseLostFrames = 0;
    if (gesture.eraseHoldStart === null) gesture.eraseHoldStart = now;
    gesture.eraseProgress = (now - gesture.eraseHoldStart) / ERASER_HOLD_MS;
    if (gesture.eraseProgress >= 1) {
      gesture.mode = MODE_ERASE;
      resetEraseHold();
      return;
    }
  } else if (gesture.eraseHoldStart !== null && ++gesture.eraseLostFrames >= MODE_HOLD_FRAMES) {
    resetEraseHold(); // lost the pose for a few frames: start over
  }

  // --- Pen / highlighter ---
  const thumbIndex = normalizedDistance(landmarks, THUMB_TIP, INDEX_TIP);
  const thumbMiddle = normalizedDistance(landmarks, THUMB_TIP, MIDDLE_TIP);

  const indexTouching = isDrawing()
    ? thumbIndex <= PEN_OFF_THRESHOLD
    : thumbIndex < PEN_ON_THRESHOLD;
  const middleTouching =
    gesture.mode === MODE_HIGHLIGHT
      ? thumbMiddle <= HL_OFF_THRESHOLD
      : thumbMiddle < HL_ON_THRESHOLD;

  // A fist often has the thumb resting against the index: never draw with it
  const wanted =
    indexTouching && !pose.fist
      ? middleTouching
        ? MODE_HIGHLIGHT
        : MODE_PEN
      : MODE_HOVER;

  if (wanted === gesture.mode) {
    gesture.candidate = null;
    gesture.candidateFrames = 0;
    return;
  }
  if (wanted === gesture.candidate) {
    gesture.candidateFrames++;
  } else {
    gesture.candidate = wanted; // wanting something new: start counting again
    gesture.candidateFrames = 1;
  }
  if (gesture.candidateFrames >= MODE_HOLD_FRAMES) {
    gesture.mode = wanted;
    gesture.candidate = null;
    gesture.candidateFrames = 0;
  }
}

// ---------------------------------------------------------------------------
// Stabilizer ("lazy brush")
// ---------------------------------------------------------------------------

let stabilizerOn = true; // toggled with the 'v' key
let drawingPoint = null; // where the line is being drawn; null when not drawing

// Returns the point the stroke should use this frame, given the cursor.
// With the stabilizer on, the drawing point is tied to the cursor by a string
// of STABILIZER_RADIUS. If the cursor is closer than that, the drawing point
// stays put. If it is farther, the string pulls tight and drags the drawing
// point along, so it ends up exactly STABILIZER_RADIUS from the cursor, on the
// line between them. The first call of a stroke starts it at the cursor.
function strokePoint(cursor) {
  if (!stabilizerOn) {
    drawingPoint = null;
    return cursor;
  }
  if (!drawingPoint) {
    drawingPoint = { x: cursor.x, y: cursor.y };
    return drawingPoint;
  }

  const dx = cursor.x - drawingPoint.x;
  const dy = cursor.y - drawingPoint.y;
  const distance = Math.hypot(dx, dy);
  if (distance > STABILIZER_RADIUS) {
    const pull = (distance - STABILIZER_RADIUS) / distance;
    drawingPoint.x += dx * pull;
    drawingPoint.y += dy * pull;
  }
  return drawingPoint;
}

// ---------------------------------------------------------------------------
// Strokes and the whiteboard
// ---------------------------------------------------------------------------

// A stroke is { tool, live, points }, where points is [{ x, y, w }, ...]:
// board pixels plus the line width at that point. Every stroke stores its tool
// and its own widths, so redrawing (undo, theme switch, cache rebuild after a
// resize) always looks the same.
//   tool "pen":       drawn in the theme's ink color
//   tool "highlight": drawn in yellow at 35% opacity, as one path so it doesn't
//                     get darker where it crosses itself (always one width)
//   live false: all points share one width (locked at stroke start)
//   live true:  each segment uses the width of its end point (pen only)
const strokes = []; // finished strokes
let currentStroke = null; // the stroke being drawn right now, or null
let boardDirty = true; // true when the board needs to be redrawn

// Finished strokes are drawn once onto this offscreen canvas. Each frame the
// board just copies it and draws the current stroke on top, so drawing stays
// fast no matter how many strokes there are. The full stroke list is only
// redrawn onto it after a resize (see rebuildCache).
const cache = document.createElement("canvas");
const cacheCtx = cache.getContext("2d");

// Pen look; set before drawing strokes (resizing a canvas resets its settings).
// Line width is set per stroke in drawStroke().
function applyPenStyle(ctx) {
  ctx.strokeStyle = currentTheme().ink;
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

// --- Board theme ---
let darkBoard = DARK_BOARD_DEFAULT;

function currentTheme() {
  return darkBoard ? THEMES.dark : THEMES.light;
}

// Set the board background (body.dark in style.css) and redraw all strokes
// in the new ink color
function applyTheme() {
  document.body.classList.toggle("dark", darkBoard);
  rebuildCache();
  boardDirty = true;
}

// Line width for a tool at the current thickness level
function strokeWidth(tool) {
  return thickness * (tool === "highlight" ? HIGHLIGHT_WIDTH_FACTOR : 1);
}

// Start a new stroke at the cursor, using the current thickness.
// afterSwitch: it starts right after a direct PEN <-> HIGHLIGHT switch.
function startStroke(cursor, tool, afterSwitch) {
  currentStroke = {
    tool,
    live: LIVE_THICKNESS && tool === "pen", // a highlighter keeps one width
    afterSwitch,
    points: [{ x: cursor.x, y: cursor.y, w: strokeWidth(tool) }],
  };
  boardDirty = true;
}

// Cut the tails off a finished stroke: drop TRIM_POINTS points from each end.
// If fewer than 2 points would be left, the stroke was a tap or a tiny mark,
// so keep it as a single dot (at its middle point) instead of losing it.
function trimStroke(stroke) {
  const pts = stroke.points;
  if (pts.length - 2 * TRIM_POINTS >= 2) {
    return { ...stroke, points: pts.slice(TRIM_POINTS, pts.length - TRIM_POINTS) };
  }
  return { ...stroke, points: [pts[Math.floor(pts.length / 2)]] };
}

// Light cleanup of a finished stroke: each point (except the first and last,
// which stay put) becomes the average of itself and its two neighbors. Widths
// are left alone. Needs at least 3 points.
function smoothStroke(stroke) {
  let pts = stroke.points;
  for (let pass = 0; pass < CLEANUP_PASSES && pts.length >= 3; pass++) {
    pts = pts.map((p, i) => {
      if (i === 0 || i === pts.length - 1) return p;
      const a = pts[i - 1];
      const b = pts[i + 1];
      return { x: (a.x + p.x + b.x) / 3, y: (a.y + p.y + b.y) / 3, w: p.w };
    });
  }
  return { ...stroke, points: pts };
}

// Undo history. Each entry is a copy of the stroke list from before one
// action (finishing a stroke, a whole erase, a clear). Strokes are never
// changed after they are finished, so a copy only holds references and is cheap.
const undoStack = [];

function pushUndo(snapshot) {
  undoStack.push(snapshot);
  if (undoStack.length > UNDO_LIMIT) undoStack.shift();
}

// Swap in a different list of strokes (undo, erase) and redraw the board
function replaceStrokes(list) {
  strokes.length = 0;
  strokes.push(...list);
  rebuildCache();
  boardDirty = true;
}

// While a stroke is in progress all its points are drawn; trimming and
// smoothing only happen here, when it is finished.
//
// discardIfShort: the stroke is ending because of a direct PEN <-> HIGHLIGHT
// switch. A short stroke there is just the mark made while the fingers settle,
// so it is deleted. The same goes for a short stroke that itself began right
// after such a switch (e.g. the PEN blip when you let go of the middle finger).
function endStroke(discardIfShort = false) {
  if (!currentStroke) return;

  const tooShort = currentStroke.points.length < SHORT_STROKE_POINTS;
  if (!(tooShort && (discardIfShort || currentStroke.afterSwitch))) {
    const finished = smoothStroke(trimStroke(currentStroke));
    pushUndo(strokes.slice());
    strokes.push(finished);
    drawStroke(cacheCtx, finished); // add it to the cache, once
  }
  currentStroke = null;
  drawingPoint = null; // the next stroke starts fresh at the cursor
  boardDirty = true;
}

// Clear the whole board. Undoable with 'z'. Returns false if it was already empty.
function clearBoard() {
  if (strokes.length === 0) return false;
  pushUndo(strokes.slice());
  strokes.length = 0;
  cacheCtx.clearRect(0, 0, cache.width, cache.height); // (overshoots because of the dpr transform; harmless)
  currentStroke = null; // if the pen is still on, a new stroke starts next frame
  drawingPoint = null;
  boardDirty = true;
  return true;
}

// Undo the last action ('z'): a stroke, a whole erase, or a clear.
// The stroke being drawn is not touched.
function undo() {
  if (undoStack.length === 0) return;
  replaceStrokes(undoStack.pop());
  hideToast(); // "Cleared. Press z to undo" no longer applies
}

// --- Toast: a big message in the middle of the screen ---
let toastTimer = null;

function showToast(text, ms) {
  toastEl.textContent = text;
  toastEl.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(hideToast, ms);
}

function hideToast() {
  clearTimeout(toastTimer);
  toastEl.hidden = true;
}

// The double fist pump: clear the board and say how to get it back
function clearByGesture() {
  showToast(clearBoard() ? "Cleared. Press z to undo" : "Nothing to clear", CLEAR_MESSAGE_MS);
}

// --- Eraser ---

// Cut the part of a stroke that lies inside the circle (center c, radius r).
// Returns null if the circle doesn't touch the stroke, otherwise the pieces
// left over (possibly none): the stroke is split where the circle cuts it.
function clipStroke(stroke, c, r) {
  const pts = stroke.points;
  const r2 = r * r;
  const isInside = (p) => (p.x - c.x) ** 2 + (p.y - c.y) ** 2 <= r2;
  const lerp = (a, b, t) => ({
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
    w: a.w + (b.w - a.w) * t,
  });

  const pieces = [];
  let current = [];
  let touched = false;
  const finishPiece = () => {
    if (current.length >= 2) pieces.push({ ...stroke, points: current });
    current = []; // (a lone leftover point would only be a stray dot)
  };

  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    if (i > 0) {
      const a = pts[i - 1];
      // The part of segment a->p inside the circle, as t in [0, 1]
      const dx = p.x - a.x;
      const dy = p.y - a.y;
      const fx = a.x - c.x;
      const fy = a.y - c.y;
      const A = dx * dx + dy * dy;
      const B = 2 * (fx * dx + fy * dy);
      const C = fx * fx + fy * fy - r2;
      let lo = null;
      let hi = null;
      if (A < 1e-9) {
        if (C <= 0) [lo, hi] = [0, 1];
      } else {
        const disc = B * B - 4 * A * C;
        if (disc > 0) {
          const root = Math.sqrt(disc);
          lo = Math.max((-B - root) / (2 * A), 0);
          hi = Math.min((-B + root) / (2 * A), 1);
        }
      }
      if (lo !== null && hi > lo) {
        touched = true;
        if (lo > 0) current.push(lerp(a, p, lo)); // where the stroke enters the circle
        finishPiece();
        if (hi < 1) current.push(lerp(a, p, hi)); // where it comes out again
      }
    }
    if (isInside(p)) touched = true;
    else current.push(p);
  }
  finishPiece();

  // A stroke that is a single dot outside the circle was never touched
  return touched ? pieces : null;
}

// Erase at one spot. Returns the new list of strokes, or null if nothing changed.
function eraseAt(list, c) {
  let changed = false;
  const out = [];
  for (const stroke of list) {
    const pieces = clipStroke(stroke, c, ERASER_RADIUS);
    if (pieces === null) {
      out.push(stroke);
    } else {
      changed = true;
      out.push(...pieces);
    }
  }
  return changed ? out : null;
}

const eraser = {
  active: false, // true during one continuous erase
  last: null, // where the eraser was on the previous frame
  snapshot: null, // strokes from before this erase, for undo
  changed: false, // did this erase remove anything yet?
};

// Call once per new camera frame, after updateMode().
// A continuous erase (ERASE mode from start to end) is one undo action.
function updateEraser(cursor) {
  if (gesture.mode === MODE_ERASE && cursor) {
    if (!eraser.active) {
      eraser.active = true;
      eraser.snapshot = strokes.slice();
      eraser.changed = false;
      eraser.last = null;
    }

    // Erase along the path since last frame too, so a fast sweep can't skip
    // over a stroke between two camera frames
    const from = eraser.last || cursor;
    const steps = Math.max(1, Math.ceil(Math.hypot(cursor.x - from.x, cursor.y - from.y) / (ERASER_RADIUS / 2)));
    let result = null;
    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      const spot = { x: from.x + (cursor.x - from.x) * t, y: from.y + (cursor.y - from.y) * t };
      result = eraseAt(result || strokes, spot) || result;
    }
    if (result) {
      replaceStrokes(result);
      eraser.changed = true;
    }
    eraser.last = { x: cursor.x, y: cursor.y };
  } else if (eraser.active) {
    if (eraser.changed) pushUndo(eraser.snapshot);
    eraser.active = false;
    eraser.snapshot = null;
    eraser.last = null;
  }
}

// Call once per new camera frame, after updateMode().
// PEN / HIGHLIGHT -> start/extend a stroke. HOVER or hand lost -> end it.
// jumped: the raw cursor made a tracking-glitch jump this frame (see
// detectJump), so the current stroke ends and a new one starts.
function updateStroke(cursor, jumped) {
  if (jumped) endStroke(); // don't draw a long stray line

  if (!isDrawing() || !cursor) {
    endStroke();
    return;
  }

  const tool = gesture.mode === MODE_HIGHLIGHT ? "highlight" : "pen";

  // PEN <-> HIGHLIGHT during a stroke: end it (deleting it if it is just a
  // short blip) and start a new one with the new tool
  let afterSwitch = false;
  if (currentStroke && currentStroke.tool !== tool) {
    endStroke(true);
    afterSwitch = true;
  }

  // Strokes use the stabilized drawing point, not the raw cursor
  const point = strokePoint(cursor);

  if (!currentStroke) {
    startStroke(point, tool, afterSwitch);
  } else {
    // `thickness` is frozen while drawing unless LIVE_THICKNESS is on, so
    // this is the locked width in the normal case
    currentStroke.points.push({ x: point.x, y: point.y, w: strokeWidth(tool) });
    boardDirty = true;
  }
}

// Draw one stroke (pen or highlighter) as a smooth line. Each point acts as a control point of a
// quadratic curve that ends at the midpoint to the next point, which rounds
// off the corners between samples.
function drawStroke(ctx, stroke) {
  const highlight = stroke.tool === "highlight";
  const color = highlight ? HIGHLIGHT_COLOR : currentTheme().ink;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  // One stroke() call per path with globalAlpha: the path is composited once,
  // so a highlighter crossing itself doesn't get darker
  if (highlight) ctx.globalAlpha = HIGHLIGHT_ALPHA;
  drawStrokeShape(ctx, stroke);
  ctx.restore();
}

function drawStrokeShape(ctx, stroke) {
  const pts = stroke.points;

  if (pts.length === 1) {
    // A single tap: just a dot
    ctx.beginPath();
    ctx.arc(pts[0].x, pts[0].y, pts[0].w / 2, 0, Math.PI * 2);
    ctx.fill();
    return;
  }

  const end = pts[pts.length - 1];

  if (!stroke.live) {
    // One width for the whole stroke: trace it as a single path
    ctx.lineWidth = pts[0].w;
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length - 1; i++) {
      const midX = (pts[i].x + pts[i + 1].x) / 2;
      const midY = (pts[i].y + pts[i + 1].y) / 2;
      ctx.quadraticCurveTo(pts[i].x, pts[i].y, midX, midY);
    }
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
    return;
  }

  // Width changes along the stroke: draw it one curve segment at a time,
  // each with its own width (round caps hide the joins between segments)
  let startX = pts[0].x;
  let startY = pts[0].y;
  for (let i = 1; i < pts.length - 1; i++) {
    const midX = (pts[i].x + pts[i + 1].x) / 2;
    const midY = (pts[i].y + pts[i + 1].y) / 2;
    ctx.lineWidth = pts[i].w;
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.quadraticCurveTo(pts[i].x, pts[i].y, midX, midY);
    ctx.stroke();
    startX = midX;
    startY = midY;
  }
  ctx.lineWidth = end.w;
  ctx.beginPath();
  ctx.moveTo(startX, startY);
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
    modeEl.textContent = gesture.mode;
    // raw / smoothed, to help set HAND_SIZE_FAR and HAND_SIZE_NEAR
    handSizeEl.textContent = `${Math.round(handSizeRaw)} / ${Math.round(handSizeSmoothed)}`;
    // Pose readings next to the thresholds they are compared with
    spreadEl.textContent = `${pose.spread.toFixed(2)} (erase at ${ERASER_SPREAD_THRESHOLD.toFixed(2)}+)`;
    fistEl.textContent = `${pose.curl.toFixed(2)} (fist at ${FIST_CURL_THRESHOLD.toFixed(2)} or less)`;
  } else {
    handEl.textContent = "no";
    distTIEl.textContent = "–";
    distTMEl.textContent = "–";
    cursorPosEl.textContent = "–";
    modeEl.textContent = "–";
    handSizeEl.textContent = "–";
    spreadEl.textContent = `– (erase at ${ERASER_SPREAD_THRESHOLD.toFixed(2)}+)`;
    fistEl.textContent = `– (fist at ${FIST_CURL_THRESHOLD.toFixed(2)} or less)`;
  }
  thicknessEl.textContent =
    gesture.mode === MODE_HIGHLIGHT
      ? `${thickness}px (highlighter ${strokeWidth("highlight")}px)`
      : `${thickness}px`;
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
      updatePose(landmarks);
      updateMode(landmarks, now);
      if (updatePump(now)) clearByGesture();
      updateHandSize(landmarks);
      updateThickness(); // after updateMode: it is locked while a stroke is drawn
      updateStroke(cursor, jumped);
      updateEraser(cursor);
      frameCount++;
    }

    // FPS = processed frames per second, refreshed twice a second
    if (now - fpsWindowStart >= 500) {
      fps = Math.round((frameCount * 1000) / (now - fpsWindowStart));
      frameCount = 0;
      fpsWindowStart = now;
    }

    drawBoard();
    drawCursor(cursor, gesture.mode, thickness, drawingPoint, gesture.eraseProgress);
    if (previewVisible()) drawPreviewOverlay(landmarks);
    updateInfoBox(fps, landmarks, cursor);

    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------------------
// Key actions
// ---------------------------------------------------------------------------

function toggleSmoothing() {
  smoothingOn = !smoothingOn;
  resetCursorFilters(); // start fresh so there's no jump when switching
  smoothingEl.textContent = smoothingOn ? "on" : "off";
}

function toggleStabilizer() {
  stabilizerOn = !stabilizerOn;
  drawingPoint = null; // restart from the cursor so there's no jump when switching
  stabilizerEl.textContent = stabilizerOn ? "on" : "off";
}

function toggleFullscreen() {
  if (document.fullscreenElement) {
    document.exitFullscreen();
  } else {
    // The browser can refuse (e.g. not allowed here); that's not worth an error screen
    document.documentElement.requestFullscreen().catch((err) => console.warn(err));
  }
}

// The camera preview is shown unless 'c' hid it or 'h' hid the whole UI
function previewVisible() {
  return !preview.classList.contains("hidden") && !document.body.classList.contains("ui-hidden");
}

// ---------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------

async function main() {
  document.body.classList.toggle("dark", darkBoard); // default board color
  resizeCanvases();
  window.addEventListener("resize", resizeCanvases);

  // Keys (also listed in the info box)
  window.addEventListener("keydown", (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey || event.repeat) return;
    switch (event.key.toLowerCase()) {
      case "c": preview.classList.toggle("hidden"); break; // camera preview
      case "h": document.body.classList.toggle("ui-hidden"); break; // info box + preview
      case "b": darkBoard = !darkBoard; applyTheme(); break; // board color
      case "z": undo(); break;
      case "x": clearBoard(); break;
      case "s": toggleSmoothing(); break;
      case "v": toggleStabilizer(); break;
      case "f": toggleFullscreen(); break;
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
