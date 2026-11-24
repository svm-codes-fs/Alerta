"""Flask server: serves the dashboard, the live MJPEG video and a small JSON API."""
import argparse
import os
import threading
import time
import webbrowser

from flask import Flask, Response, jsonify, render_template, request

from . import ROOT, config
from .alerts import AlarmPlayer, TelegramNotifier
from .detector import Detector

app = Flask(__name__)
cfg = config.load()
detector = None  # created in main()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    def frames():
        last = None
        while True:
            with detector.lock:
                jpeg = detector.jpeg
            if jpeg is not None and jpeg is not last:
                last = jpeg
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
            time.sleep(0.02)
    return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def status():
    return jsonify(detector.get_status())


@app.route('/api/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        d = request.get_json(force=True)
        if d.get('night_mode') in ('off', 'auto', 'on'):
            cfg['night_mode'] = d['night_mode']
        if 'volume' in d:
            cfg['volume'] = min(1.0, max(0.0, float(d['volume'])))
        if 'muted' in d:
            cfg['muted'] = bool(d['muted'])
        if 'distraction_seconds' in d:
            cfg['distraction_seconds'] = min(10.0, max(1.0, float(d['distraction_seconds'])))
        if d.get('reset_calibration'):
            cfg.update(closed_threshold=0.5, baseline_open=None)
            detector.calib_message = ''
        config.save(cfg)
    return jsonify({k: cfg[k] for k in ('night_mode', 'volume', 'muted', 'distraction_seconds',
                                        'closed_threshold', 'baseline_open')})


@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    detector.start_calibration()
    return jsonify(ok=True)


@app.route('/api/pause', methods=['POST'])
def pause():
    detector.paused = bool(request.get_json(force=True).get('paused'))
    return jsonify(paused=detector.paused)


@app.route('/api/telegram', methods=['GET', 'POST'])
def telegram():
    if request.method == 'POST':
        d = request.get_json(force=True)
        if 'enabled' in d:
            cfg['telegram_enabled'] = bool(d['enabled'])
        if d.get('bot_token'):  # blank keeps the saved token
            cfg['telegram_bot_token'] = d['bot_token'].strip()
        if 'chat_id' in d:
            cfg['telegram_chat_id'] = str(d['chat_id']).strip()
        if 'emergency_after' in d:
            cfg['emergency_after'] = min(10, max(1, int(d['emergency_after'])))
        if d.get('forget_token'):
            cfg['telegram_bot_token'] = ''
        config.save(cfg)
    # the token itself is never sent back to the browser
    return jsonify(enabled=cfg['telegram_enabled'], token_saved=bool(cfg['telegram_bot_token']),
                   chat_id=cfg['telegram_chat_id'], emergency_after=cfg['emergency_after'])


@app.route('/api/telegram/test', methods=['POST'])
def telegram_test():
    with detector.lock:
        jpeg = detector.jpeg
    ok = detector.notifier.send('Alerta test message: emergency alerts are set up correctly.', jpeg)
    return jsonify(ok=ok, error=detector.notifier.last_error)


def main():
    global detector
    ap = argparse.ArgumentParser(description='Alerta drowsiness detection dashboard')
    ap.add_argument('--camera', type=int, default=0, help='webcam index (default 0)')
    ap.add_argument('--port', type=int, default=5000)
    ap.add_argument('--no-browser', action='store_true', help="don't open the dashboard automatically")
    args = ap.parse_args()

    alarm = AlarmPlayer(cfg, os.path.join(ROOT, 'alarm.wav'))
    detector = Detector(cfg, alarm, TelegramNotifier(cfg), camera_index=args.camera)
    detector.start()
    url = f'http://127.0.0.1:{args.port}'
    print(f'Alerta dashboard running at {url}  (Ctrl+C to stop)')
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    app.run(host='127.0.0.1', port=args.port, threaded=True, use_reloader=False)
