// MagicBoard – step 1: hand tracking only.
// Pipeline: webcam -> MediaPipe HandLandmarker -> draw skeleton + info box.

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Page elements
// ---------------------------------------------------------------------------

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const messageEl = document.getElementById("message");
const infoEl = document.getElementById("info");
const fpsEl = document.getElementById("fps");
const handEl = document.getElementById("hand");
const distTIEl = document.getElementById("distTI");
const distTMEl = document.getElementById("distTM");

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
// Drawing
// ---------------------------------------------------------------------------

// Match the canvas to the window size (and screen sharpness).
function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = window.innerWidth * dpr;
  canvas.height = window.innerHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // draw in CSS pixels from now on
}

// Convert a landmark (0–1 in the video frame) to a canvas position.
// Mirrors x to match the mirrored video, and applies the same "cover"
// scaling + centering that CSS object-fit: cover uses on the video.
function toScreen(landmark) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const scale = Math.max(window.innerWidth / vw, window.innerHeight / vh);
  const offsetX = (window.innerWidth - vw * scale) / 2;
  const offsetY = (window.innerHeight - vh * scale) / 2;
  return {
    x: offsetX + (1 - landmark.x) * vw * scale, // (1 - x) = mirror
    y: offsetY + landmark.y * vh * scale,
  };
}

function drawHand(landmarks) {
  const points = landmarks.map(toScreen);

  // Connecting lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
  ctx.lineWidth = 3;
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
    ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
    ctx.fill();
  }

  // Big colored fingertips with labels
  ctx.font = "bold 16px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (const { index, color, label } of HIGHLIGHTS) {
    const p = points[index];
    ctx.fillStyle = color;
    ctx.strokeStyle = "white";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 12, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Label with a dark outline so it's readable on any background
    ctx.lineWidth = 4;
    ctx.strokeStyle = "black";
    ctx.strokeText(label, p.x + 18, p.y);
    ctx.fillStyle = "white";
    ctx.fillText(label, p.x + 18, p.y);
  }
}

function updateInfoBox(fps, landmarks) {
  fpsEl.textContent = fps;
  if (landmarks) {
    handEl.textContent = "yes";
    distTIEl.textContent = normalizedDistance(landmarks, THUMB_TIP, INDEX_TIP).toFixed(2);
    distTMEl.textContent = normalizedDistance(landmarks, THUMB_TIP, MIDDLE_TIP).toFixed(2);
  } else {
    handEl.textContent = "no";
    distTIEl.textContent = "–";
    distTMEl.textContent = "–";
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

  function tick() {
    const now = performance.now();

    // Only run the model when the webcam has produced a new frame
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      landmarks = detectHand(handTracker, now);
      frameCount++;
    }

    // FPS = processed frames per second, refreshed twice a second
    if (now - fpsWindowStart >= 500) {
      fps = Math.round((frameCount * 1000) / (now - fpsWindowStart));
      frameCount = 0;
      fpsWindowStart = now;
    }

    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    if (landmarks) drawHand(landmarks);
    updateInfoBox(fps, landmarks);

    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------

async function main() {
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  // 1. Camera
  showMessage("Requesting camera access…");
  try {
    await startCamera();
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
