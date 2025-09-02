from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Optional
import logging

LOGGER = logging.getLogger(__name__)

@dataclass
class PlayerConfig:
    debounce_seconds: float = 0.4


class VideoPlayer:
    """
    Controls a video player via python-vlc if available, otherwise falls back to a no-op logger.
    Actions:
      RIGHT -> seek +5s
      LEFT  -> seek -5s
      THUMB_UP   -> volume +5%
      THUMB_DOWN -> volume -5%
      PALM -> toggle pause/play
    """
    def __init__(self, media_path: Optional[str] = None, config: Optional[PlayerConfig] = None):
        self.cfg = config or PlayerConfig()
        self._last_action_time = 0.0
        self._vlc = None
        self._player = None
        try:
            import vlc  # python-vlc
            self._vlc = vlc
            self._instance = vlc.Instance()
            self._player = self._instance.media_player_new()
            if media_path:
                media = self._instance.media_new(media_path)
                self._player.set_media(media)
                self._player.play()
                time.sleep(0.1)
        except Exception as e:
            LOGGER.warning("python-vlc not available or failed to init: %s. Using logging fallback.", e)

    def _debounced(self) -> bool:
        now = time.time()
        if now - self._last_action_time < self.cfg.debounce_seconds:
            return False
        self._last_action_time = now
        return True

    def toggle_play_pause(self):
        if not self._debounced():
            return
        if self._player is not None:
            if self._player.is_playing():
                self._player.pause()
            else:
                self._player.play()
        LOGGER.info("Action: TOGGLE PLAY/PAUSE")

    def seek(self, seconds: float):
        if not self._debounced():
            return
        if self._player is not None:
            length = self._player.get_length()  # ms
            cur = self._player.get_time()       # ms
            if length > 0 and cur >= 0:
                new_ms = int(max(0, min(length, cur + int(seconds*1000))))
                self._player.set_time(new_ms)
        LOGGER.info("Action: SEEK %+0.1fs", seconds)

    def volume_delta(self, delta_percent: float):
        if not self._debounced():
            return
        if self._player is not None:
            cur_vol = max(0, min(100, self._player.audio_get_volume()))
            new_vol = int(max(0, min(100, cur_vol + delta_percent)))
            self._player.audio_set_volume(new_vol)
        LOGGER.info("Action: VOLUME %+d%%", int(delta_percent))

    # Mapping helpers
    def on_right(self):  # seek +5s
        self.seek(+5.0)

    def on_left(self):   # seek -5s
        self.seek(-5.0)

    def on_thumb_up(self):   # volume +5%
        self.volume_delta(+5.0)

    def on_thumb_down(self): # volume -5%
        self.volume_delta(-5.0)

    def on_palm(self):       # toggle play/pause
        self.toggle_play_pause()
