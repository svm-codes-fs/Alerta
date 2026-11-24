# Alerta on Raspberry Pi — future work (not part of the lab demo)

> This folder is a **separate starting point** for later. Nothing in the main project imports it,
> and the dashboard (`run_dashboard.py`) works without it.

## Goal
Run Alerta as a small, cheap device in the car: a Raspberry Pi with a camera on the dashboard,
a buzzer or speaker for the alarm, and no laptop.

## Suggested hardware
| Part | Notes |
|---|---|
| Raspberry Pi 4 (4 GB) or Pi 5 | Pi 5 gives roughly twice the frame rate |
| Pi Camera Module 3 **NoIR** + 2 IR LEDs | NoIR + infrared light works at night without blinding the driver |
| Small speaker (3.5 mm / USB) or active buzzer on a GPIO pin | for the alarm |
| 12 V → 5 V 3 A car USB adapter | stable power is important |
| Case + windshield/dashboard mount | camera at eye level, ~50–70 cm from the face |

## Plan (step by step)
1. **Convert the model to TensorFlow Lite** on the laptop:
   `python raspberry_pi/convert_to_tflite.py` → `raspberry_pi/eye_model.tflite` (~10 MB, float16).
   TFLite runs fast on the Pi and needs only the small `tflite-runtime` package, not full TensorFlow.
2. **Set up the Pi**: Raspberry Pi OS (64-bit), then `pip install -r raspberry_pi/requirements-pi.txt`.
3. **Camera**: read frames with `picamera2` (see `pi_detector.py`), or a USB webcam with OpenCV.
4. **Port the detection loop**: `pi_detector.py` is a skeleton with TODOs. Reuse the logic in
   `alerta/detector.py` (eye closure timing, escalation, blink rate, distraction, night mode).
5. **Alarm output**: GPIO buzzer with `gpiozero.Buzzer(17)` (beep faster per level), or the speaker.
6. **Emergency alert**: reuse `alerta/alerts.py` `TelegramNotifier`; needs Wi-Fi / a phone hotspot.
7. **Auto-start on boot**: a `systemd` service (template at the bottom of this file).
8. **Measure**: frame rate on the Pi, CPU temperature, false alarms on a test drive (as a passenger!).

## Expected performance (to verify)
- Haar face detection on 320×240 frames: ~10–20 fps on a Pi 4. Face detection is the slowest part.
- If too slow: detect the face every 3rd frame and track the box in between, or switch to
  MediaPipe Face Mesh / a lighter face detector.

## TODO checklist
- [ ] Convert model to TFLite and compare its accuracy with the Keras model on `data/test`
- [ ] Get camera frames on the Pi (picamera2)
- [ ] Port the detection loop (`pi_detector.py` TODOs)
- [ ] GPIO buzzer with escalating beep pattern
- [ ] IR LEDs for night driving
- [ ] Telegram alert over a phone hotspot
- [ ] systemd auto-start
- [ ] Field test and FPS / temperature measurements

## systemd service template
```ini
# /etc/systemd/system/alerta.service
[Unit]
Description=Alerta drowsiness detector
After=network-online.target

[Service]
WorkingDirectory=/home/pi/Alerta
ExecStart=/usr/bin/python3 raspberry_pi/pi_detector.py
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```
Enable with `sudo systemctl enable --now alerta`.
