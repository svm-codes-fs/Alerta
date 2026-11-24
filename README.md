# Alerta
An AI-powered real-time drowsiness detection system that uses computer vision to monitor driver fatigue, track eye closure (PERCLOS), and prevent road accidents with instant alerts.

## Project base and experiments

Built on the open-source [Drowsiness detection [OPEN CV]](https://github.com/shsarv/Machine-Learning-Projects/tree/main/Drowsiness%20detection%20%5BOPEN%20CV%5D) project
(Haar cascade eye detection + Keras CNN eye-state classifier, real-time alarm in `drowsinessdetection.py`).

Experiments added on top of it — baseline, data augmentation, and a learning-rate change, evaluated on the
Driver Drowsiness Dataset (DDD) with a subject-wise split — are documented with real results in [EXPERIMENTS.md](EXPERIMENTS.md).

## Web dashboard

```bash
python run_dashboard.py          # opens http://127.0.0.1:5000 in your browser
```
Options: `--camera 1` (other webcam), `--port 8000`, `--no-browser`. Stop with Ctrl+C.

The dashboard shows the live camera with face/eye boxes, the driver state, and these features:

| # | Feature | How it works |
|---|---|---|
| 1 | **Escalating alarm** | Eyes closed 1 s / 2 s / 3 s → level 1 / 2 / 3; the alarm gets louder each level (30 % / 65 % / 100 % of the volume slider) |
| 4 | **Blink-rate monitor** | Closures shorter than 1 s count as blinks: blinks per minute and average blink length over the last minute |
| 6 | **Distraction alert** | No face in view for N seconds (default 2) → "LOOK AT THE ROAD" + alarm, stronger after 2N seconds |
| 8 | **Personal calibration** | 5 s of normal looking → your eye openness is measured and the closed-eye threshold set to 60 % of it (max 0.5) |
| 9 | **Night mode** | CLAHE contrast enhancement before detection; Off / On / Auto (turns on when the image is dark) |
| 11 | **Emergency alert** | After N critical (level 3) alarms within 10 minutes, a Telegram bot sends a message with a camera snapshot (5 min cooldown) |

Every alarm is listed in the event log and saved to `logs/session_<date>.csv`.
Settings are stored in `config.json` (not committed, it can hold your Telegram bot token).

**Telegram setup (optional):** message `@BotFather` on Telegram → `/newbot` → copy the bot token. Send your new bot any
message, open `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `"chat":{"id": ...}`. Enter both on the
dashboard, tick *Enabled*, press *Save*, then *Send test*.

### How it is built
```
run_dashboard.py         starts everything
alerta/detector.py       webcam loop: Haar face + eyes -> CNN (models/cnnCat2.h5) -> features 1, 4, 6, 8, 9, 11
alerta/alerts.py         escalating alarm (pygame) + Telegram notifier
alerta/server.py         Flask: dashboard page, MJPEG video stream, JSON API
alerta/templates, static dashboard page (HTML/CSS/JS, no internet needed)
```
The original `drowsinessdetection.py` (OpenCV window version) still works on its own.

### Raspberry Pi (future work)
`raspberry_pi/` is a separate starting point for running Alerta as an in-car device (TFLite model, Pi camera,
GPIO buzzer). It is not used by the dashboard; see [raspberry_pi/README.md](raspberry_pi/README.md).
