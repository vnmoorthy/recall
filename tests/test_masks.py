import numpy as np

from recall.masks import area, centroid, encode_uncompressed_rle, iou, jpeg_crop, project_letterbox_mask


def test_mask_geometry_and_crop():
    mask = np.zeros((100, 120), np.uint8)
    mask[20:60, 30:80] = 255
    assert area(mask) == 2000
    assert centroid(mask) == (54.5, 39.5)
    shifted = np.zeros_like(mask)
    shifted[20:60, 40:90] = 255
    assert 0.65 < iou(mask, shifted) < 0.68
    frame = np.full((100, 120, 3), 127, np.uint8)
    assert jpeg_crop(frame, mask).startswith(b"\xff\xd8")


def test_letterbox_projection_and_rle():
    head = np.zeros((160, 160), np.uint8)
    head[40:120, 40:120] = 255
    projected = project_letterbox_mask(head, (100, 50, 300, 250), (400, 300))
    assert projected.shape == (300, 400)
    assert area(projected) > 0
    rle = encode_uncompressed_rle(projected)
    assert sum(rle) == projected.size
    assert rle[0] > 0
