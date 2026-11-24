"""Escalating alarm sound (feature 1) and Telegram emergency notification (feature 11)."""
import threading
import time

import requests

# share of the master volume used at each alarm level
LEVEL_VOLUME = {1: 0.3, 2: 0.65, 3: 1.0}


class AlarmPlayer:
    """Loops alarm.wav and turns it up as the alarm level rises; silent at level 0."""

    def __init__(self, cfg, wav_path):
        self.cfg = cfg
        self.channel = None
        try:
            from pygame import mixer
            mixer.init()
            self.sound = mixer.Sound(wav_path)
        except Exception as e:  # no audio device: keep running without sound
            print('Alarm sound disabled:', e)
            self.sound = None

    def update(self, level):
        if self.sound is None:
            return
        volume = 0 if self.cfg['muted'] else self.cfg['volume'] * LEVEL_VOLUME.get(level, 0)
        if volume <= 0:
            if self.channel is not None:
                self.channel.stop()
                self.channel = None
            return
        if self.channel is None or not self.channel.get_busy():
            self.channel = self.sound.play(loops=-1)
        if self.channel is not None:
            self.channel.set_volume(volume)


class TelegramNotifier:
    """Sends an emergency message with a snapshot through a Telegram bot.

    Set up once: create a bot with @BotFather (gives the bot token), send your bot any
    message, then open https://api.telegram.org/bot<TOKEN>/getUpdates to read your chat id.
    Enter both on the dashboard; they are stored only in the local config.json.
    """

    COOLDOWN = 300  # seconds between two automatic emergency messages

    def __init__(self, cfg):
        self.cfg = cfg
        self.last_sent = None
        self.last_error = ''
        self.sending = False

    @property
    def configured(self):
        return bool(self.cfg['telegram_bot_token'] and self.cfg['telegram_chat_id'])

    def ready(self):
        return (self.cfg['telegram_enabled'] and self.configured and not self.sending
                and (self.last_sent is None or time.time() - self.last_sent > self.COOLDOWN))

    def send(self, text, jpeg=None):
        if not self.configured:
            self.last_error = 'Bot token and chat id are not set'
            return False
        self.sending = True
        threading.Thread(target=self._send, args=(text, jpeg), daemon=True).start()
        return True

    def _send(self, text, jpeg):
        url = f"https://api.telegram.org/bot{self.cfg['telegram_bot_token']}/"
        try:
            if jpeg:
                r = requests.post(url + 'sendPhoto', data={'chat_id': self.cfg['telegram_chat_id'], 'caption': text},
                                  files={'photo': ('snapshot.jpg', jpeg, 'image/jpeg')}, timeout=15)
            else:
                r = requests.post(url + 'sendMessage', data={'chat_id': self.cfg['telegram_chat_id'], 'text': text},
                                  timeout=15)
            body = r.json()
            if body.get('ok'):
                self.last_sent, self.last_error = time.time(), ''
            else:
                self.last_error = body.get('description', f'HTTP {r.status_code}')
        except Exception as e:
            # never echo the request URL: it contains the bot token
            self.last_error = type(e).__name__
        finally:
            self.sending = False
