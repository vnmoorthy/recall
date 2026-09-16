# Recall: 50-Pass Product Audit

Each pass identified a concrete fault or operational ambiguity and either changed
the implementation or added a verification boundary.

| # | Fault found | Improvement |
|---:|---|---|
| 1 | New person tracks were immediately marked one frame missing. | Missing counts now use the complete active-ID set. |
| 2 | Object visibility used dataclass equality on NumPy masks. | Membership now uses integer track IDs. |
| 3 | A stationary period beginning at timestamp zero was reset. | `None` is tested explicitly. |
| 4 | Expired missing tracks retained full-frame masks forever. | Tracks retire after the reacquisition window; SQLite history remains. |
| 5 | A wrist-attributed moving object did not update the live tracker. | Pickups now feed back the `carried` state. |
| 6 | Carrier identity expired with the 1.5-second wrist history. | EventEngine retains carrier IDs until placement or retirement. |
| 7 | Reappearance after a missing pickup did not emit `put_down`. | Missing-to-present with a carrier now emits attributed placement. |
| 8 | Retired carriers accumulated in memory. | Carrier entries are removed with retired tracks. |
| 9 | SQLite readers and writers used rollback journaling. | WAL mode and normal synchronous durability are enabled. |
| 10 | Lock contention failed immediately. | SQLite now has a five-second busy timeout. |
| 11 | Every object committed a separate transaction each frame. | Object/person state is batch-upserted. |
| 12 | Routine SQLite commits blocked the perception callback. | A bounded keep-latest TrackWriter owns frame-state persistence. |
| 13 | Async stale states could erase completed VLM names. | Name and crop updates use `COALESCE` conflict semantics. |
| 14 | Absolute frame paths produced broken `/media` URLs. | FrameWriter returns `frames/...` public paths. |
| 15 | Frame shutdown abandoned queued evidence. | The frame queue drains with an extended close window. |
| 16 | JPEG failures were invisible. | Frame write errors are counted in status metrics. |
| 17 | Object crops had no contextual margin. | Crops include a configurable 12 percent margin. |
| 18 | Crop encoding ran on the frame thread. | Raw crop arrays move to the naming worker. |
| 19 | Hardware inventory crops were never saved. | Naming saves stable `crops/{kind}-{id}.jpg` files. |
| 20 | Naming shutdown discarded work. | The bounded naming queue drains before exit. |
| 21 | A callback exception could terminate naming. | Callback failures are isolated per task. |
| 22 | Empty or whitespace-heavy names reached SQLite. | Names are collapsed, bounded, and empty values rejected. |
| 23 | API and perception could call one VLM concurrently. | A cross-process `flock` serializes the resident model. |
| 24 | Waiting for that lock was unbounded. | Lock acquisition now has a tested timeout. |
| 25 | Questions had no size limit. | Questions are bounded to 500 characters. |
| 26 | Summary and event windows were unbounded. | Public windows are constrained to one day. |
| 27 | Empty audio reached Whisper. | Empty multipart payloads return HTTP 400. |
| 28 | Audio uploads could exhaust RAM. | Uploads are capped at 25 MiB. |
| 29 | Whisper I/O blocked the async event loop. | Transcription and answering run in the thread pool. |
| 30 | Empty Whisper transcripts generated meaningless questions. | Empty transcripts return HTTP 422. |
| 31 | Multipart filenames and boundaries were static/unsafe. | Names are sanitized and boundaries are random UUIDs. |
| 32 | Model-supplied evidence IDs were trusted. | Event/object IDs are validated against SQLite. |
| 33 | Repeated model IDs leaked into responses. | Evidence IDs are deduplicated and capped. |
| 34 | Adjacent events displayed the same snapshot twice. | Snapshot paths are deduplicated. |
| 35 | Model prose could be blank, huge, or concatenate sentences. | Answer/summary text is normalized and bounded. |
| 36 | Epoch zero was treated as an absent timestamp. | Time parsing distinguishes `None` from zero. |
| 37 | Summary ordering compared clock strings across midnight. | Importance ties now use epoch timestamps. |
| 38 | Segmentation polygons could exceed UDP limits. | Metadata is confidence-prioritized under 60,000 bytes. |
| 39 | Pose overlay IDs changed with detection order. | Insight pose metadata uses PersonTracker IDs. |
| 40 | Metadata send failures were silent. | Send failures are counted and exposed. |
| 41 | Tracking values in YAML were ignored. | All identity/event/retention thresholds now drive runtime objects. |
| 42 | Invalid YAML values reached PyNeat. | Source, FPS, score, channel, ratio, and duration checks fail fast. |
| 43 | Stale metrics could report dead perception as healthy. | Metrics expire and `perception_healthy` is explicit. |
| 44 | A GenAI PID was mistaken for model readiness. | Health requires the process and a verified model-ready marker. |
| 45 | PID reuse could preserve unrelated processes. | Supervisor commands are validated against `/proc` command lines. |
| 46 | Orphan Recall processes escaped PID files. | Module-based process discovery adopts or stops them safely. |
| 47 | Synthetic/hardware mode changes did not restart the API. | Command-aware supervision replaces mismatched processes. |
| 48 | GenAI replacement raced MLA model unloading. | Coordinated restart quiesces all workers and resets appcomplex. |
| 49 | Fresh installations could not reproduce speech setup. | A reviewed, pinned `setup_tts_devkit.sh` path is included. |
| 50 | UI polling, recording, and status lacked repeated-use safeguards. | Fetch timeouts, overlap prevention, busy states, URL validation, last-seen data, speech readiness, and recorder errors are implemented. |

## Verification

- Pure logic/API/worker suite: `31 passed` after these passes.
- JavaScript parsed with `node --check`; all shell launchers passed `bash -n`.
- Concurrent VLM question + summary completed without MLA contention.
- Coordinated restart removed an orphan server and loaded Gemma + Whisper once.
- Piper asynchronously prewarmed; post-ready speech returned in 1.15-1.63 seconds.
- Exact YOLO26 recorded-stream gates remain blocked on authenticated model assets.
