# MagicBoard

Touchless whiteboard using a webcam, hand gestures, and a projector.

## Tech
- Plain HTML, CSS, and JavaScript. No frameworks, no build step.
- MediaPipe Tasks Vision (HandLandmarker) loaded from a CDN.
- Files: `index.html`, `main.js`, `style.css`, `README.md`.

## Run
```bash
python3 -m http.server 8000
```
Then open http://localhost:8000 (the camera needs localhost or HTTPS). Node is not installed on this machine, so don't suggest `npx`.

## Working rules
- We build in small steps. Only build what the current step asks for.
- Never break earlier features.
- Do not make git commits. The user commits.
- Never add AI attribution or Co-Authored-By lines anywhere (commits, PRs, code, docs).
