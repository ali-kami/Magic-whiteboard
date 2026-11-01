# MagicBoard

A whiteboard you draw on with hand gestures, using a webcam and a projector.

**Step 1 (this version): hand tracking only.** No drawing yet. It shows your
webcam, overlays your hand skeleton, and highlights the thumb, index and
middle fingertips.

## Run it

You need [Node.js](https://nodejs.org) (for `npx`) and a webcam. The camera only
works on `localhost` or HTTPS, so open the page through a local server rather
than double-clicking `index.html`.

```bash
cd Magic-whiteboard
npx serve
```

Then open the address it prints (usually <http://localhost:3000>) in Chrome or
Edge and click **Allow** when asked for camera access. An internet connection
is needed on first load (MediaPipe and the hand model come from a CDN).

## What you should see

- Your webcam feed filling the window, mirrored (move your hand right → it moves right on screen).
- A white hand skeleton with 21 yellow dots when your hand is in view.
- Three big fingertip dots: **T** red (thumb), **I** blue (index), **M** green (middle).
- An info box in the top-left: FPS, whether a hand is detected, and the
  thumb–index and thumb–middle distances.

The distances are divided by hand size (wrist to middle knuckle), so `1.00`
means "one hand-size apart". Pinching thumb and index together should give
a value near `0.00`.

If the camera is denied or MediaPipe can't load, a red error message appears
instead of a blank page.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Page structure: video, canvas, info box, message |
| `style.css` | Full-window layout, mirrored video, info box styling |
| `main.js` | Camera, hand tracking, drawing, info box |

Hand tracking lives in `createHandTracker()`, `startCamera()`, `detectHand()`
and the distance helpers in `main.js`, so later steps can build on them.
