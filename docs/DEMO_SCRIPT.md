# Recall Demo Script

## Deliverables

- Recorded walkthrough: [`demo/recall-product-demo.mp4`](../demo/recall-product-demo.mp4)
- Reusable recorder: [`tools/record_demo.py`](../tools/record_demo.py)
- Public interactive demo: <https://vnmoorthy.github.io/recall/>
- Runtime UI: `http://127.0.0.1:8080`

The included recording is an honest walkthrough of the offline `SIMULATION`
mode. It demonstrates the complete UI workflow, but it is not evidence that the
currently disconnected Modalix perception models are running. Record the final
hardware cut only when the header shows a live hardware link.

## 90-Second Talk Track

| Time | Operator action | Presenter words | Expected screen result |
|---|---|---|---|
| 0:00 | Show the live view and status strip. | "Recall is Ctrl+F for the physical world: private visual memory that runs locally on a SiMa Modalix device." | Camera or local replay is moving; mode is clearly labeled. |
| 0:10 | Ask **What's on the table?** | "Recall continuously builds an object inventory. Naming happens off the frame path, so tracking never waits for the language model." | Five named objects and one evidence frame appear. |
| 0:25 | Move the mug out of frame, then ask **Where is the red mug?** | "The mug disappeared while a tracked wrist was nearby. Recall preserved who moved it, when it happened, and the exit direction." | Mug changes to `MISSING`; answer says it moved right; pickup frame appears. |
| 0:42 | Have a person take the laptop; ask **Who took my laptop?** | "The same event engine attributes the laptop pickup using pose proximity. People are described only by clothing, never by identity." | Laptop changes to `MISSING`; clothing description and snapshot appear. |
| 0:58 | Disconnect upstream Ethernet and repeat the laptop query. | "The behavior is unchanged because video, memory, language, speech, and evidence stay on the local device." | Query still returns locally; no cloud connection is required. |
| 1:12 | Select **Mission summary**. | "Recall turns ten minutes of room activity into three factual sentences and the two most important evidence frames." | Summary and two snapshots appear. |
| 1:25 | Scroll to the object index and select the mug. | "Every answer is auditable. Selecting an object filters its event history instead of returning an unsupported guess." | Timeline filters to the mug's events. |

Closing line:

> Recall makes continuous visual memory practical in homes, hospitals, labs,
> workshops, and stockrooms because the video never has to leave the building.

## Recording The Offline Walkthrough

Start the UI first:

```bash
cd /workspace/recall
./run_mac.sh
```

In another terminal, install the recorder once and generate the MP4:

```bash
cd /workspace/recall
python -m venv .venv-demo
. .venv-demo/bin/activate
pip install -r requirements-demo.txt
playwright install chromium
python tools/record_demo.py --output demo/recall-product-demo.mp4
```

The recorder creates a 1280x720 silent MP4 with title, beat, and closing
captions. It drives the actual browser controls and fails if the page reports a
JavaScript error. Record narration separately from the talk track when a spoken
submission is required.

## Final Hardware Cut

1. Confirm the header reports the hardware perception mode, not `SIMULATION`.
2. Frame the mug, laptop, bottle, phone, and book together for ten seconds.
3. Perform all five beats without cuts so the offline claim is visible.
4. Keep the status strip in frame when unplugging upstream Ethernet.
5. Show the evidence snapshots and object timeline before ending.
6. Do not claim exact YOLO26 or Qwen operation until those authenticated model
   packages are installed and their hardware gates have passed.

## Acceptance Checklist

- The UI and text remain legible at 1280x720.
- Every query visibly changes the response and evidence panels.
- Mug and laptop states change from stationary to missing.
- The event log grows and object selection filters it.
- The summary shows two evidence frames.
- Simulation and hardware footage are never presented as each other.
- No passwords, tokens, terminal history, or private network details appear.
