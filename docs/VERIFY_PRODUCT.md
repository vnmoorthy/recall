# Verify Recall

Recall has two deliberately separate operating modes. Use these checks to avoid
mistaking the public replay for live Modalix inference.

## Public Product Demo

Open <https://vnmoorthy.github.io/recall/app.html> (the site root at <https://vnmoorthy.github.io/recall/> is the product page; its **Launch the live demo** button opens the app).

1. Confirm the header says `SIMULATION` and the first panel says
   `RECORDED PERCEPTION`.
2. Wait for the four-stage scenario to advance through `INDEX`, `MUG`,
   `LAPTOP`, and `SUMMARY`.
3. Select **Locate mug** and confirm the response includes `moved right` with
   one evidence frame.
4. Select the checkmark button in the header.

Expected system-check result:

```text
3/3 LOCAL CHECKS PASS
LOCAL DEMO READY // HARDWARE OFFLINE
```

The public site proves the interface, local replay, inventory, event timeline,
queries, and evidence rendering. It does not prove that Modalix is connected.

## Live Modalix Verification

The laptop running Recall must be physically connected to the powered DevKit.
First verify the API from that laptop:

```bash
curl --fail --max-time 3 http://10.42.0.232:8090/health
curl --fail --max-time 3 http://10.42.0.232:8090/status | python -m json.tool
```

Live perception requires all of these status conditions:

```text
perception_mode = hardware
seg_fps > 0
pose_fps > 0
```

Start the local UI and open it on the same laptop:

```bash
cd /workspace/recall
./run_mac.sh
```

Open `http://127.0.0.1:8080`, then run the header system check. A passing live
system reports `LIVE PERCEPTION ONLINE`; the first panel changes to
`LIVE PERCEPTION`, and the feed state changes to `LIVE LINK`.

## If Live Perception Is Offline

| Result | Meaning | Required action |
|---|---|---|
| API curl cannot connect | DevKit or Ethernet link is unavailable | Power and reconnect the DevKit; verify its address. |
| API responds, mode is `synthetic` | Preview/fallback process is active | Confirm both staged YOLO26 archives exist and restart with `./run_devkit.sh --restart`. |
| Hardware mode, zero FPS | Model or camera pipeline is not producing samples | Check DevKit logs, model paths, and camera/RTSP source. |
| Video works, queries fail | Perception is active but GenAI is unavailable | Verify the resident Qwen/Gemma server and Recall API logs. |
| Public site says hardware offline | Expected for GitHub Pages | Use the local UI on the directly connected laptop. |

The exact segmentation and pose packages are already staged in the shared
`models/` directory. The next `./run_devkit.sh --restart` automatically selects
hardware perception; no second download or source change is needed. Qwen still
requires `llima pull` after the DevKit reconnects. Never paste credentials into
an issue or chat.
