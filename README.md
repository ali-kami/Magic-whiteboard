# MagicBoard

A whiteboard you draw on with hand gestures, using a webcam and a projector.

**Step 8 (this version): steadier writing** on top of the highlighter gesture, presentation mode, pen thickness, smoothing, clean strokes and the pen gesture. The window is a white
whiteboard, a small camera preview with your hand skeleton sits in the
bottom-right corner, and a cursor follows the midpoint between your thumb and
index fingertips. Pinch thumb and index together to draw; open them to hover.

## Run it

You need Python 3 and a webcam. The camera only works on `localhost` or HTTPS,
so open the page through a local server rather than double-clicking `index.html`.

```bash
cd Magic-whiteboard
python3 -m http.server 8000
```

Then open <http://localhost:8000> in Chrome or Edge and click **Allow** when
asked for camera access. An internet connection
is needed on first load (MediaPipe and the hand model come from a CDN).

## What you should see

- A full-window whiteboard (black by default; white with `b`).
- A small mirrored camera preview (about 25% of the window width) in the bottom-right,
  with the hand skeleton and the red **T**, blue **I**, green **M** fingertips.
- A faint dashed rectangle on the preview: the **active zone**. Keep your hand inside it.
- **Gestures** (one mode at a time; a new mode must hold for 3 camera frames):
  - `HOVER`: no pinch. Just the cursor.
  - `PEN`: thumb touches index (thumb–index below `PEN_ON_THRESHOLD` 0.25, stays on
    until above `PEN_OFF_THRESHOLD` 0.35). Draws ink.
  - `HIGHLIGHT`: thumb touches index **and** middle (thumb–middle below
    `HL_ON_THRESHOLD` 0.25, stays on until above `HL_OFF_THRESHOLD` 0.35). Draws a
    yellow highlighter at 35% opacity, 3x the current thickness, as one path so it
    doesn't darken where it crosses itself.
  All thresholds are constants at the top of `main.js`.
- Your index usually lands first on a 3-finger pinch, so PEN may start for a few
  frames before HIGHLIGHT. On a direct PEN <-> HIGHLIGHT switch the current stroke
  ends, and if it has fewer than `SHORT_STROKE_POINTS` (10) points it is deleted, so
  no stray mark is left. A short stroke that *starts* right after such a switch is
  deleted too (the PEN blip when you let go of the middle finger).
- Each stroke stores its tool (pen or highlight), so undo, `b` theme switch, the
  cache and resize all redraw it correctly. Highlighter color doesn't change with
  the theme. A highlighter always has one width, even with `LIVE_THICKNESS`.
- While hovering the cursor is a hollow ring; while drawing it becomes a solid dot
  (ink color for the pen, yellow and 3x wider for the highlighter). The info box
  shows `Mode: HOVER`, `PEN` or `HIGHLIGHT`.
- A stroke ends if you go back to HOVER, the hand disappears, or the cursor jumps
  more than 150px in one frame (so tracking glitches don't leave stray lines).
- **Smoothing:** the cursor is passed through a One Euro filter (`MIN_CUTOFF`,
  `BETA`, `D_CUTOFF` at the top of `main.js`). Lower `MIN_CUTOFF` = steadier but
  laggier; higher `BETA` = less lag on fast moves. It resets when the hand disappears.
- **Clean strokes:** when a stroke finishes, `TRIM_POINTS` (3) points are removed
  from each end to cut the tails caused by pinching and releasing. Strokes too
  short to trim become a single dot, so quick taps still work. While you are
  still drawing, the whole line is shown, then it tightens up on release.
- **Performance:** finished strokes are cached on an offscreen canvas; each frame
  only copies the cache and draws the stroke in progress on top. The full list is
  redrawn only after a window resize.
- **Glitch protection:** the 150px jump check uses the raw (unsmoothed) cursor.
- **Thickness from depth:** move your hand closer to the camera for a thicker pen,
  farther for a thinner one. The depth signal is hand size in pixels (average of
  wrist–middle knuckle and index knuckle–pinky knuckle), smoothed with an
  exponential moving average (`SIZE_SMOOTHING`). It is mapped between
  `HAND_SIZE_FAR` and `HAND_SIZE_NEAR` to 5 levels: 2, 4, 8, 14, 22 px (clamped
  outside the range, with a little hysteresis so it doesn't flicker on a boundary).
- The thickness is locked when a stroke starts, so each stroke has one width. It
  uses the level from `THICKNESS_LOCK_LOOKBACK` (6) frames before the pen turned on,
  because the pinch itself nudges your hand and can change the level. Set
  `LIVE_THICKNESS = true` in `main.js` to let it change while drawing. Every stroke
  stores its widths, so resizing redraws it correctly.
- The hover ring and the drawing dot are as wide as the current thickness
  (with a minimum size so they stay visible). The info box shows `Hand size`
  (raw / smoothed) and `Thickness`.

**Tuning the range:** hold your hand open at the farthest distance you'd
comfortably draw, read the *smooth* number in `Hand size`, and put it in
`HAND_SIZE_FAR`. Do the same at the closest distance for `HAND_SIZE_NEAR`.
- **Projector use:** the board starts black with white ink. Strokes are stored
  without a color, so `b` recolors everything already drawn. The info box ends with
  a key cheat sheet. `CURSOR_MIN_SIZE` (top of `main.js`) sets the smallest hover
  ring, so raise it if the cursor is hard to see on the wall.
- **Steadier writing:**
  - **Stabilizer ("lazy brush"):** strokes use a drawing point tied to the cursor by an
    invisible string of `STABILIZER_RADIUS` (12px). It only moves when the cursor
    pulls the string tight, so small shakes are ignored. While drawing you see it as a
    small red dot joined to the cursor by a faint line. `v` toggles it.
  - **Active zone** is now 80% of the camera frame (`ACTIVE_ZONE`).
  - **Smoothing defaults:** `MIN_CUTOFF = 0.5`, `BETA = 0.003`.
  - **Stroke cleanup:** a finished stroke gets a 3-point moving average
    (`CLEANUP_PASSES`) after trimming.
  - **Camera:** 1280x720 is requested (`CAMERA_WIDTH` / `CAMERA_HEIGHT`); the info
    box shows the resolution you actually got.

**Still shaky?** Raise `STABILIZER_RADIUS` (e.g. 20), lower `MIN_CUTOFF` (e.g. 0.3),
raise `CLEANUP_PASSES` to 2, or lower `ACTIVE_ZONE` (e.g. 0.7).
**Too laggy?** Do the opposite: lower `STABILIZER_RADIUS` (e.g. 6), raise `MIN_CUTOFF`
(e.g. 0.8) or `BETA` (e.g. 0.01), set `CLEANUP_PASSES` to 0.
- A cursor at the midpoint of your thumb and index tips.
  The active zone (middle 60% of the camera frame) covers the whole board, so
  reaching a zone edge puts the cursor at the matching board edge. The cursor
  never leaves the screen and disappears when no hand is detected.
- An info box (top-left): FPS, hand detected, thumb–index and thumb–middle
  distances (divided by hand size), and the cursor's x, y on the board in pixels.

## Keys

| Key | Action |
| --- | --- |
| `c` | Show / hide the camera preview (tracking keeps running while hidden) |
| `h` | Hide / show the info box and camera preview together (the cursor stays) |
| `b` | Toggle board color: black board + white ink (default) / white board + black ink. Existing strokes change color too |
| `z` | Undo the last finished stroke |
| `f` | Toggle fullscreen |
| `x` | Clear the board |
| `s` | Turn cursor smoothing on / off (shown in the info box) |
| `v` | Turn the stabilizer on / off (shown in the info box) |

If the camera is denied or MediaPipe can't load, a red error message appears
instead of a blank page.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Page structure: board, cursor layer, preview, info box, message |
| `style.css` | Layout and styling |
| `main.js` | Camera, hand tracking, cursor, pen gesture, strokes, preview, info box |

In `main.js`: hand tracking is `createHandTracker()`, `startCamera()`,
`detectHand()` and the distance helpers; the cursor is `getCursor()` and
`drawCursor()`; the gesture mode is `updateMode()`; strokes are `updateStroke()`,
`trimStroke()`, `detectJump()`, `strokePoint()`, `smoothStroke()`, `handDepthSize()`, `updateThickness()`, `drawStroke()`, `rebuildCache()` and `drawBoard()`; smoothing is `OneEuroFilter` and
`smoothCursor()`. The whiteboard (`#board`) and the cursor
(`#cursor`) are separate canvases so strokes aren't erased when the cursor moves.
