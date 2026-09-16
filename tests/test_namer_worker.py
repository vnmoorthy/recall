from types import SimpleNamespace

import numpy as np

from recall.namer import Namer


def test_namer_encodes_saves_and_drains_crop(tmp_path, monkeypatch):
    callbacks = []
    namer = Namer(
        "http://unused", "test-model", tmp_path / "vlm.jsonl",
        on_named=lambda kind, track, path: callbacks.append((kind, track.id, path)),
        crop_dir=tmp_path / "crops",
    )
    monkeypatch.setattr(namer, "_request", lambda task, jpeg: "  Red   ceramic\n mug  ")
    track = SimpleNamespace(
        id=9, state="stationary", class_name="cup", name=None, name_pending=False,
    )
    assert namer.submit("object", track, np.zeros((20, 30, 3), np.uint8))
    namer.close()
    assert track.name == "Red ceramic mug"
    assert callbacks == [("object", 9, "crops/object-9.jpg")]
    assert (tmp_path / "crops/object-9.jpg").is_file()
