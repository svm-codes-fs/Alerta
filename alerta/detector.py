"""Webcam loop: the original Haar cascade + CNN pipeline, plus the dashboard features.

1  Escalating alarm      eyes closed 1 s / 2 s / 3 s -> alarm level 1 / 2 / 3 (louder each level)
4  Blink-rate monitor    short closures (< 1 s) counted as blinks: blinks per minute + average length
6  Distraction alert     no face in view for N seconds -> "look at the road" alarm
8  Calibration           5 s of normal looking -> personal open/closed threshold
9  Night mode            CLAHE contrast boost, always on or automatic in dark frames
11 Emergency alert       repeated level-3 alarms -> Telegram message with a snapshot
"""
import collections
import csv
import datetime
import os
import threading
import time

import cv2
import numpy as np

from . import ROOT, config
from .model_loader import load_eye_model

CASCADES = os.path.join(ROOT, 'haar cascade files')
ALARM_STEPS = ((3.0, 3), (2.0, 2), (1.0, 1))   # (seconds eyes closed, alarm level), highest first
LEVEL_NAMES = {0: 'Alert', 1: 'Drowsy', 2: 'Very drowsy', 3: 'Wake up!'}
LEVEL_COLORS = {1: (0, 200, 255), 2: (0, 120, 255), 3: (0, 0, 255)}  # BGR
BLINK_MAX = 1.0           # closures shorter than this are blinks
BLINK_WINDOW = 60.0       # blink rate is measured over the last minute
CALIB_SECONDS = 5.0
NIGHT_BRIGHTNESS = 70     # auto night mode below this average gray level
EMERGENCY_WINDOW = 600.0  # critical alarms counted over the last 10 minutes


class Detector:
    def __init__(self, cfg, alarm, notifier, camera_index=0):
        self.cfg, self.alarm, self.notifier = cfg, alarm, notifier
        self.camera_index = camera_index
        self.model = load_eye_model()
        self.face = cv2.CascadeClassifier(os.path.join(CASCADES, 'haarcascade_frontalface_alt.xml'))
        self.leye = cv2.CascadeClassifier(os.path.join(CASCADES, 'haarcascade_lefteye_2splits.xml'))
        self.reye = cv2.CascadeClassifier(os.path.join(CASCADES, 'haarcascade_righteye_2splits.xml'))
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.lock = threading.Lock()
        self.jpeg = None
        self.running = False
        self.paused = False
        self.error = ''
        self.events = collections.deque(maxlen=100)
        os.makedirs(os.path.join(ROOT, 'logs'), exist_ok=True)
        self.log_path = os.path.join(ROOT, 'logs', datetime.datetime.now().strftime('session_%Y%m%d_%H%M%S.csv'))
        self.session_start = time.time()
        self.closed_since = None       # time the current eye closure started
        self.no_face_since = None      # time the face was last lost
        self.eye_closed = False
        self.p_open = None
        self.blinks = collections.deque()          # (time, duration) of recent blinks
        self.level = 0
        self.level_peak = 0
        self.level_since = None
        self.critical_times = collections.deque()  # times of recent level-3 alarms
        self.alarms_total = 0
        self.calib = None                          # {'start': t, 'samples': [...]} while calibrating
        self.calib_message = ''
        self.fps = 0.0
        self.status = {}

    # ---------- control ----------
    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def start_calibration(self):
        self.calib = {'start': time.time(), 'samples': []}
        self.calib_message = 'Look at the camera normally with your eyes open...'

    def _log(self, event, detail=''):
        now = datetime.datetime.now()
        self.events.appendleft({'time': now.strftime('%H:%M:%S'), 'event': event, 'detail': detail})
        new = not os.path.exists(self.log_path)
        with open(self.log_path, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['time', 'event', 'detail'])
            w.writerow([now.isoformat(timespec='seconds'), event, detail])

    # ---------- main loop ----------
    def _loop(self):
        cap, last = None, time.time()
        while self.running:
            if self.paused:
                if cap is not None:
                    cap.release()
                    cap = None
                self._set_level(0, time.time())
                time.sleep(0.1)
                continue
            if cap is None:
                cap = cv2.VideoCapture(self.camera_index)
                if not cap.isOpened():
                    self.error = 'Cannot open the webcam (is another app using it?)'
                    cap.release()
                    cap = None
                    time.sleep(2)
                    continue
            ok, frame = cap.read()
            if not ok:
                self.error = 'Webcam stopped sending frames (is another app using it?)'
                cap.release()
                cap = None
                time.sleep(1)
                continue
            self.error = ''
            t = time.time()
            self.fps = 0.9 * self.fps + 0.1 / max(t - last, 1e-3)
            last = t
            out = self.process(frame, t)
            ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self.lock:
                    self.jpeg = buf.tobytes()
        if cap is not None:
            cap.release()
        self.alarm.update(0)

    def process(self, frame, t):
        if self.cfg['mirror']:
            frame = cv2.flip(frame, 1)  # selfie view, like a mirror
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())

        # feature 9: night mode - local contrast boost on the lightness channel only,
        # so the preview stays in colour and detection gets the brighter image
        mode = self.cfg['night_mode']
        night = mode == 'on' or (mode == 'auto' and brightness < NIGHT_BRIGHTNESS)
        if night:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[..., 0] = self.clahe.apply(lab[..., 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # face on a half-size image (faster), eyes only inside the upper part of the face
        small = cv2.resize(gray, None, fx=0.5, fy=0.5)
        faces = self.face.detectMultiScale(small, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        face_found = len(faces) > 0
        eyes_found = False
        if face_found:
            x, y, w, h = [int(v) * 2 for v in max(faces, key=lambda f: f[2] * f[3])]
            self.no_face_since = None
            cv2.rectangle(frame, (x, y), (x + w, y + h), (160, 160, 160), 1)
            top = gray[y:y + int(h * 0.6), x:x + w]
            crops = []
            for casc in (self.leye, self.reye):
                found = casc.detectMultiScale(top, minSize=(w // 8, w // 8))
                if len(found):
                    ex, ey, ew, eh = max(found, key=lambda e: e[2] * e[3])
                    crops.append(cv2.resize(top[ey:ey + eh, ex:ex + ew], (24, 24)))
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (80, 220, 80), 1)
            if crops:
                eyes_found = True
                x_in = np.array(crops, dtype='float32')[..., None] / 255.
                self.p_open = float(np.asarray(self.model(x_in, training=False))[:, 1].mean())
                self.eye_closed = self.p_open < self.cfg['closed_threshold']
            # face but no eyes found: keep the last eye state, like the original script
        elif self.no_face_since is None:
            self.no_face_since = t

        # feature 8: calibration collects your normal P(open) for a few seconds
        if self.calib is not None:
            self._calibrate(t, eyes_found)

        # eye closure timing -> blinks (feature 4) and drowsiness duration
        closed = face_found and self.eye_closed
        if closed and self.closed_since is None:
            self.closed_since = t
        elif not closed and self.closed_since is not None:
            if face_found and t - self.closed_since < BLINK_MAX:
                self.blinks.append((t, t - self.closed_since))
            self.closed_since = None
        while self.blinks and t - self.blinks[0][0] > BLINK_WINDOW:
            self.blinks.popleft()
        closed_for = t - self.closed_since if self.closed_since else 0.0

        # feature 1: alarm level rises with how long the eyes stay closed
        level = next((lv for secs, lv in ALARM_STEPS if closed_for >= secs), 0)
        # feature 6: distraction - face out of view too long
        no_face_for = t - self.no_face_since if self.no_face_since else 0.0
        limit = self.cfg['distraction_seconds']
        distracted = no_face_for >= limit
        if distracted:
            level = max(level, 2 if no_face_for >= 2 * limit else 1)
        if self.calib is not None:
            level = 0
        self._set_level(level, t, frame, 'Distracted' if distracted and closed_for < 1.0 else 'Drowsy')

        self._draw(frame, closed, closed_for, distracted, night)
        self.status = self._make_status(t, face_found, eyes_found, closed, closed_for, distracted, no_face_for,
                                        night, brightness)
        return frame

    def _calibrate(self, t, eyes_found):
        c = self.calib
        if eyes_found:
            c['samples'].append(self.p_open)
        elapsed = t - c['start']
        if elapsed >= CALIB_SECONDS and len(c['samples']) >= 10:
            baseline = float(np.median(c['samples']))
            # closed = clearly below your own normal openness, never stricter than the default 0.5
            threshold = round(min(0.5, max(0.15, 0.6 * baseline)), 3)
            self.cfg.update(baseline_open=round(baseline, 3), closed_threshold=threshold)
            config.save(self.cfg)
            self.calib = None
            self.calib_message = f'Done: your eye openness {baseline:.2f}, closed below {threshold:.2f}'
            self._log('Calibrated', self.calib_message)
        elif elapsed >= 3 * CALIB_SECONDS:
            self.calib = None
            self.calib_message = 'Calibration failed: eyes not found. Face the camera in good light and retry.'

    def _set_level(self, level, t, frame=None, kind='Drowsy'):
        if level > self.level:
            if self.level == 0:
                self.alarms_total += 1
                self.level_since = t
                self.level_peak = 0
            self._log(f'{kind} - level {level}', 'Face not in view' if kind == 'Distracted' else LEVEL_NAMES[level])
            if level == 3:
                self._critical(t, frame)
        elif level == 0 and self.level > 0:
            self._log('Recovered', f'after {t - self.level_since:.1f} s (peak level {self.level_peak})')
        self.level_peak = max(self.level_peak, level)
        self.level = level
        self.alarm.update(level)

    def _critical(self, t, frame):
        # feature 11: emergency notification after repeated critical alarms
        self.critical_times.append(t)
        while self.critical_times and t - self.critical_times[0] > EMERGENCY_WINDOW:
            self.critical_times.popleft()
        if len(self.critical_times) >= self.cfg['emergency_after'] and self.notifier.ready():
            ok, buf = cv2.imencode('.jpg', frame)
            text = (f"Alerta emergency: the driver had {len(self.critical_times)} critical drowsiness alarms "
                    f"in the last 10 minutes ({datetime.datetime.now():%d %b %Y %H:%M}).")
            self.notifier.send(text, buf.tobytes() if ok else None)
            self._log('Emergency alert sent', 'Telegram message with snapshot')

    def _draw(self, frame, closed, closed_for, distracted, night):
        h, w = frame.shape[:2]
        if self.level:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), LEVEL_COLORS[self.level], 4 * self.level + 2)
        cv2.rectangle(frame, (0, h - 34), (w, h), (0, 0, 0), cv2.FILLED)
        state = 'NO FACE' if distracted else ('Closed' if closed else 'Open')
        text = f'{state}  {closed_for:.1f}s  {LEVEL_NAMES[self.level]}' + ('  [night]' if night else '')
        cv2.putText(frame, text, (10, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if distracted:
            cv2.putText(frame, 'LOOK AT THE ROAD', (w // 2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 2, cv2.LINE_AA)

    def _make_status(self, t, face_found, eyes_found, closed, closed_for, distracted, no_face_for, night, brightness):
        span = min(BLINK_WINDOW, t - self.session_start)
        n = len(self.blinks)
        return {
            'error': self.error, 'paused': self.paused, 'fps': round(self.fps, 1),
            'face': face_found, 'eyes': eyes_found,
            'eye_state': 'no face' if not face_found else ('closed' if closed else 'open'),
            'p_open': None if self.p_open is None else round(self.p_open, 3),
            'threshold': self.cfg['closed_threshold'], 'baseline_open': self.cfg['baseline_open'],
            'closed_for': round(closed_for, 1),
            'level': self.level, 'level_name': LEVEL_NAMES[self.level],
            'distracted': distracted, 'no_face_for': round(no_face_for, 1),
            'blinks_per_min': round(n * 60 / span, 1) if span >= 10 else None,
            'avg_blink_ms': round(1000 * sum(d for _, d in self.blinks) / n) if n else None,
            'night_active': night, 'brightness': round(brightness),
            'calibrating': self.calib is not None,
            'calib_progress': min(1.0, (t - self.calib['start']) / CALIB_SECONDS) if self.calib else None,
            'calib_message': self.calib_message,
            'alarms_total': self.alarms_total, 'critical_recent': len(self.critical_times),
            'session_minutes': round((t - self.session_start) / 60, 1),
        }

    def get_status(self):
        s = dict(self.status) or {'error': self.error or 'Starting camera...', 'level': 0, 'level_name': 'Alert'}
        s.update(error=self.error or s.get('error', ''), paused=self.paused, events=list(self.events)[:30],
                 telegram={'enabled': self.cfg['telegram_enabled'], 'configured': self.notifier.configured,
                           'last_sent': self.notifier.last_sent, 'last_error': self.notifier.last_error,
                           'sending': self.notifier.sending})
        return s
