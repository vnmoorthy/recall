# Recall: Product Audit Passes 51-100

This second round continued the same fault-first process. Each pass either changed
the product, added a regression boundary, verified a live operational property,
or recorded an external asset gate without claiming it was complete.

| # | Fault or risk inspected | Result |
|---:|---|---|
| 51 | Event-window reads could return the entire database. | SQL results are bounded, with API limits validated to 1-1000. |
| 52 | Per-object timelines were also unbounded. | Timeline reads now accept a capped limit. |
| 53 | Object-history queries lacked a supporting index. | Added `(object_id, ts)` indexing. |
| 54 | Person-oriented event joins lacked a supporting index. | Added `(person_id, ts)` indexing. |
| 55 | Events derived from one frame committed separately. | A frame's event batch now commits once. |
| 56 | A failed event batch could leave partial work pending. | The batch uses SQLite transaction rollback semantics. |
| 57 | Unknown object timelines looked like valid empty histories. | They now return HTTP 404; invalid IDs return 400. |
| 58 | Timeline events omitted object and person names. | Timeline SQL now returns the same joined names as the event feed. |
| 59 | Liveness was the only startup probe. | Added `/ready` for perception, GenAI/fallback, and Piper. |
| 60 | Operators could not distinguish degraded from ready. | Readiness returns HTTP 503 with component states. |
| 61 | A full speech queue failed silently. | Queue drops are counted and exposed in `/status`. |
| 62 | Speech worker exceptions disappeared. | Playback/synthesis errors are counted. |
| 63 | A nonzero `aplay` exit was treated as success. | Nonzero exits now enter the speech error counter. |
| 64 | Shutdown could close Piper during active playback. | The join window now covers the configured playback timeout. |
| 65 | Repeated memory shutdown raised on a closed connection. | `Memory.close()` is idempotent. |
| 66 | WAL setup could contend before the busy timeout existed. | The timeout is set before journal-mode negotiation. |
| 67 | The last pose was reused forever if the pose branch stalled. | Person detections expire after a bounded freshness interval. |
| 68 | Pose records accepted invalid or non-finite boxes. | Decode now rejects malformed pose boxes. |
| 69 | NaN/Inf keypoints could break JSON overlay generation. | Non-finite keypoints are emitted as zero-confidence points. |
| 70 | Keypoint confidence could leave the documented range. | Overlay confidence is clamped to 0-1. |
| 71 | Truncated NV12 tensors failed as opaque reshape errors. | Payload size is checked with a precise error. |
| 72 | Truncated I420 tensors had the same ambiguity. | Both YUV420 paths share the payload check. |
| 73 | Odd YUV420 dimensions could reach OpenCV. | Invalid dimensions fail before conversion. |
| 74 | Insight UDP ports were not range checked. | Video and metadata ports must be 1-65535. |
| 75 | Background crop naming was absent from metrics. | Naming calls, failures, and median latency are exported. |
| 76 | Naming latency samples grew for process lifetime. | Samples use a 512-entry bounded deque. |
| 77 | Answer latency samples also grew without bound. | API latency samples use the same bound. |
| 78 | A large historical inventory could overfill the VLM prompt. | Context includes the 100 most recent inventory rows. |
| 79 | Summary evidence could repeat the same image. | Summary snapshots are path-deduplicated. |
| 80 | Object IDs restarted at 1 after perception restart. | New IDs continue above the SQLite maximum. |
| 81 | Person IDs had the same overwrite risk. | Person tracking also resumes above persisted IDs. |
| 82 | Frame-writer shutdown had too little drain time. | The drain window was extended to 30 seconds. |
| 83 | Track persistence had the same close race. | TrackWriter receives the extended drain window. |
| 84 | `stop_devkit.sh` returned while workers could still run. | It waits, then force-stops only validated Recall processes. |
| 85 | The launcher returned as soon as the API socket opened. | With Piper installed it now waits for `/ready`. |
| 86 | The readiness failure contract was absent from OpenAPI. | HTTP 503 is declared in the route schema. |
| 87 | Inventory cards did not expose their event history. | Selecting a card loads its object timeline. |
| 88 | Selected history had no subject context. | The timeline heading changes to the object name. |
| 89 | Returning to the room-wide feed required a reload. | Added an `All events` control. |
| 90 | Selection state was visually ambiguous. | The active inventory item is highlighted and uses `aria-pressed`. |
| 91 | UI polling did not request an explicit event bound. | It requests at most 500 events. |
| 92 | Older browsers could throw while probing `AbortSignal.timeout`. | The global is checked before access. |
| 93 | A click during an active poll could be ignored for three seconds. | Overlapping refresh requests now queue one follow-up. |
| 94 | The VLM call total omitted automatic naming. | Status exposes answer, naming, and combined totals. |
| 95 | The mode badge omitted track and speech failures. | Its diagnostic tooltip now includes both. |
| 96 | Preview availability was inferred only from a PID. | Insight confirmed live H.264 ingress on channel 0. |
| 97 | No attached browser could be mistaken for failed ingress. | Ingest stats show valid H.264 while accurately reporting no WebRTC peer. |
| 98 | Exact YOLO26/Qwen execution was still assumed by the brief. | It remains explicitly gated on authenticated SiMa assets. |
| 99 | New boundaries had no regression tests. | The suite grew from 31 to 40 passing tests. |
| 100 | Changes were not yet proven as a deployed product. | The DevKit was restarted and typed, summary, speech, ASR, history, and readiness paths passed. |

## End-State Verification

- `/ready`: HTTP 200 with perception, GenAI, and Piper ready.
- Gemma evidence answer: 2.67 seconds with validated event IDs and one snapshot.
- Three-sentence summary: 6.23 seconds with two distinct snapshots.
- Piper: 0.54 seconds for a valid mono 16-bit 22.05 kHz WAV.
- Whisper plus evidence answer: 3.77 seconds with the phrase transcribed correctly.
- Insight: active H.264 channel 0, SPS/PPS/IDR present, no malformed packets or estimated sequence gaps.
- Tests: `40 passed, 1 skipped`; the skip is the exact-model recorded pickup gate.

The post-ledger smoke also caught a Gemma response that used valid evidence IDs
but invented a clock time. Model answers now pass a timestamp-grounding check;
an unknown time causes deterministic evidence rendering instead of user-visible
prose.
