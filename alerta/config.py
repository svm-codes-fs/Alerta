"""Settings shared by the detector and the dashboard, saved to config.json (not committed)."""
import json
import os

from . import ROOT

PATH = os.path.join(ROOT, 'config.json')

DEFAULTS = {
    'night_mode': 'auto',          # off | auto | on  (feature 9)
    'volume': 0.5,                 # master alarm volume 0..1 (feature 1)
    'muted': False,
    'distraction_seconds': 2.0,    # no face for this long -> distraction alert (feature 6)
    'closed_threshold': 0.5,       # eye counts as closed when P(open) < this (feature 8 tunes it)
    'baseline_open': None,         # your normal P(open), measured by calibration
    'telegram_enabled': False,     # emergency notification (feature 11)
    'telegram_bot_token': '',
    'telegram_chat_id': '',
    'emergency_after': 2,          # critical alarms within 10 minutes before notifying
}


def load():
    cfg = dict(DEFAULTS)
    try:
        with open(PATH) as f:
            cfg.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return cfg


def save(cfg):
    with open(PATH, 'w') as f:
        json.dump(cfg, f, indent=2)
