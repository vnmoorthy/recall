import math

from recall.keypoints import person_bbox, wrists


def test_nonfinite_keypoints_are_ignored():
    points = [[0.0, 0.0, 0.0] for _ in range(17)]
    points[9] = [math.nan, 20.0, 0.9]
    points[10] = [30.0, 40.0, 0.8]
    assert len(wrists(points)) == 1
    assert wrists(points)[0].x == 30.0
    points[10] = [math.inf, 40.0, 0.8]
    assert wrists(points) == ()
    assert person_bbox(points) == (0.0, 0.0, 0.0, 0.0)
