# Hackathon submission copy

## Title (37/50)

Recall: Ctrl+F for the Physical World

## Short description (238/255)

A camera on a SiMa.ai Modalix chip that remembers everything in a room and answers out loud: "Where's my red mug?" "Who took the laptop?" Segmentation, pose, tracking and a vision-language model on one sub-10 W chip, with no cloud at all.

## Long description (354 words)

Recall is Ctrl+F for the physical world. A single camera on a SiMa.ai Modalix DevKit remembers everything that happens in a room and answers questions out loud, with the snapshot and the moment: "What's on the table?" "Where's my red mug?" "Who took the laptop?" "What happened here in the last ten minutes?"

The thesis is simple. Continuous visual memory of a home, a hospital ward, a lab, or a stockroom cannot exist in the cloud, because nobody will stream that video to a server. It can only exist if it runs on the device. Recall runs entirely on the Modalix MLSoC under 10 watts. The runtime has no cloud client, the UI loads nothing from a CDN, and you can pull the internet cable mid-demo and it keeps answering.

Every model on the chip has a job no other model can do. YOLO26 instance segmentation runs on every frame and gives each object a pixel mask and a stable identity. YOLO26 pose runs at 10 fps and gives us wrists: when an object moves or vanishes while a person's wrist was within 60 pixels of its mask in the previous 1.5 seconds, that pickup is attributed to that person. A tracker keeps object and person IDs through occlusion and re-entry. Qwen3-VL, served by SiMa's LLiMa, names each object once in five words including its color, and describes people by clothing only, never identity. The same model in text mode answers questions over a compact evidence JSON pulled from SQLite. Every answer cites event IDs and passes a timestamp-grounding check, so the model cannot invent facts the camera never saw. Whisper Small handles spoken questions on the MLA and Piper speaks the answers. The vision-language model is never on the frame path: stop it, and inventory and events keep flowing.

Measured on the board: custody questions with snapshots answered in 2.7 seconds, three-sentence summaries in 6.2 seconds, spoken question to spoken answer in under 4 seconds, and 40 passing tests. Homes, hospitals, labs, workshops, and retail back rooms are the first markets, and the privacy property that stops cloud AI is exactly Recall's moat.
