"""Hardware RTSP-to-Insight relay used before perception models are installed."""

from __future__ import annotations

import argparse
import signal
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="rtsp://10.42.0.1:8554/src1")
    parser.add_argument("--insight-host", default="10.42.0.1")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    import pyneat

    source_options = pyneat.RtspEncodedInputOptions()
    source_options.url = args.source
    source_options.codec = pyneat.RtspCodec.H264
    source_options.fallback_h264_width = args.width
    source_options.fallback_h264_height = args.height
    source_options.source_fps = args.fps

    video_options = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.H264)
    video_options.host = args.insight_host
    video_options.channel = 0
    video_options.video_port_base = 9000
    video_options.async_ = True

    graph = pyneat.Graph("recall_preview")
    graph.connect(
        pyneat.groups.rtsp_encoded_input(source_options),
        pyneat.groups.video_sender(video_options),
    )
    options = pyneat.RunOptions()
    options.preset = pyneat.RunPreset.Realtime
    options.queue_depth = 3
    options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run = graph.build(options)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(f"preview {args.source} -> {args.insight_host}:9000", flush=True)
    try:
        while not stopped:
            time.sleep(0.5)
    finally:
        run.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
