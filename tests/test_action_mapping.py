from src.ui.player import VideoPlayer

def test_debounce_logic(monkeypatch):
    vp = VideoPlayer(media_path=None)
    # Force debounce to allow action then suppress immediate repeats
    vp._last_action_time = 0.0
    assert vp._debounced() is True
    assert vp._debounced() is False
