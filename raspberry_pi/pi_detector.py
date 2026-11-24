"""Raspberry Pi detector — SKELETON for future work, not finished and not used by the dashboard.

Runs headless on the Pi: camera -> face/eyes (Haar) -> TFLite eye model -> buzzer.
Fill in the TODOs; reuse the logic from alerta/detector.py.
"""
import os
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASCADES = os.path.join(os.path.dirname(HERE), 'haar cascade files')
MODEL = os.path.join(HERE, 'eye_model.tflite')     # made by convert_to_tflite.py
CLOSED_THRESHOLD = 0.5                              # TODO: load from calibration (config.json)
ALARM_STEPS = ((3.0, 3), (2.0, 2), (1.0, 1))        # same escalation as the dashboard


def load_interpreter():
    from tflite_runtime.interpreter import Interpreter  # pip install tflite-runtime
    interp = Interpreter(model_path=MODEL)
    interp.allocate_tensors()
    return interp


def predict_open(interp, eye_24x24):
    """Probability that one 24x24 grayscale eye crop is open."""
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    interp.set_tensor(inp['index'], (eye_24x24.astype('float32') / 255.)[None, ..., None])
    interp.invoke()
    return float(interp.get_tensor(out['index'])[0, 1])


def open_camera():
    # TODO: Pi camera via picamera2:
    #   from picamera2 import Picamera2
    #   cam = Picamera2(); cam.configure(cam.create_preview_configuration(main={'size': (320, 240)})); cam.start()
    #   frame = cam.capture_array()
    return cv2.VideoCapture(0)  # USB webcam fallback


def main():
    interp = load_interpreter()
    face = cv2.CascadeClassifier(os.path.join(CASCADES, 'haarcascade_frontalface_alt.xml'))
    eye = cv2.CascadeClassifier(os.path.join(CASCADES, 'haarcascade_lefteye_2splits.xml'))
    cam = open_camera()
    closed_since = None
    # TODO: buzzer = gpiozero.Buzzer(17)
    while True:
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.5)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # TODO: night mode (CLAHE) as in alerta/detector.py
        faces = face.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            # TODO: distraction timer -> alarm after N seconds without a face
            continue
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        top = gray[y:y + int(h * 0.6), x:x + w]
        eyes = eye.detectMultiScale(top)
        if len(eyes) == 0:
            continue
        ex, ey, ew, eh = eyes[0]
        p_open = predict_open(interp, cv2.resize(top[ey:ey + eh, ex:ex + ew], (24, 24)))
        t = time.time()
        if p_open < CLOSED_THRESHOLD:
            closed_since = closed_since or t
        else:
            closed_since = None     # TODO: count blinks here (closure < 1 s)
        closed_for = t - closed_since if closed_since else 0.0
        level = next((lv for s, lv in ALARM_STEPS if closed_for >= s), 0)
        # TODO: buzzer pattern per level, e.g. buzzer.beep(on_time=0.1, off_time=0.4 / level) / buzzer.off()
        # TODO: Telegram emergency alert (reuse alerta/alerts.py TelegramNotifier)
        print(f'p_open={p_open:.2f} closed_for={closed_for:.1f}s level={level}')


if __name__ == '__main__':
    main()
