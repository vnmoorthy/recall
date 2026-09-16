import threading

import pytest

from recall.vlm_lock import exclusive_vlm


def test_vlm_lock_has_bounded_wait(tmp_path):
    acquired = threading.Event()
    release = threading.Event()

    def holder():
        with exclusive_vlm(tmp_path / "vlm.lock"):
            acquired.set()
            release.wait(2)

    thread = threading.Thread(target=holder)
    thread.start()
    assert acquired.wait(1)
    with pytest.raises(TimeoutError):
        with exclusive_vlm(tmp_path / "vlm.lock", timeout=0.05):
            pass
    release.set()
    thread.join(1)
    assert not thread.is_alive()
