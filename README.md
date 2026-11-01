# MagicBoard

A whiteboard you draw on with hand gestures, using a webcam and a projector.

**Step 4 (this version): smoothing and clean strokes** on top of the pen gesture. The window is a white
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

- A full-window white whiteboard.
- A small mirrored camera preview (about 25% of the window width) in the bottom-right,
  with the hand skeleton and the red **T**, blue **I**, green **M** fingertips.
- A faint dashed rectangle on the preview: the **active zone**. Keep your hand inside it.
- **Pen gesture:** pinch thumb and index to draw a black 4px line. The pen turns
  ON when the normalized thumb–index distance drops below `0.25` and OFF when it
  rises above `0.35`, and a change must hold for 3 camera frames. Tune these with
  the constants at the top of `main.js`. The middle finger is ignored for now.
- While hovering the cursor is a hollow dark gray ring; while drawing it becomes a
  solid black dot, slightly smaller. The info box shows `Mode: PEN` or `Mode: HOVER`.
- A stroke ends if the pen turns off, the hand disappears, or the cursor jumps
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
| `x` | Clear the board |
| `s` | Turn cursor smoothing on / off (shown in the info box) |

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
`drawCursor()`; the pen is `updatePen()`; strokes are `updateStroke()`,
`trimStroke()`, `detectJump()`, `drawStroke()`, `rebuildCache()` and `drawBoard()`; smoothing is `OneEuroFilter` and
`smoothCursor()`. The whiteboard (`#board`) and the cursor
(`#cursor`) are separate canvases so strokes aren't erased when the cursor moves.
