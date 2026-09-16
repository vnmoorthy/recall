from types import SimpleNamespace

from recall.actions import Speaker


class FakeEngine:
    ready = True
    error = None

    def warm(self):
        pass

    def synthesize(self, _text):
        return b"RIFF"

    def close(self):
        pass


def test_nonzero_playback_exit_is_observable(monkeypatch):
    monkeypatch.setattr(
        "recall.actions.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )
    speaker = Speaker(FakeEngine())
    assert speaker.speak("test")
    speaker.queue.join()
    assert speaker.errors == 1
    speaker.close()
