#!/usr/bin/env python3
"""Record the Recall browser walkthrough and optionally transcode it to MP4."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

try:
    from playwright.sync_api import Page, sync_playwright
except ImportError as exc:
    raise SystemExit(
        "Playwright is required. Run: pip install -r requirements-demo.txt && "
        "playwright install chromium"
    ) from exc


TITLE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><style>
*{box-sizing:border-box}body{margin:0;background:#050505;color:#f5f7f8;font-family:Arial,sans-serif}
main{height:100vh;display:flex;flex-direction:column;justify-content:center;padding:0 92px;border-top:3px solid #4ba3ff}
.mark{width:46px;height:28px;border:3px solid #fff;border-right:0;transform:skewX(-18deg);margin-bottom:34px}
.eyebrow{color:#4ba3ff;font-size:15px;font-weight:800;margin-bottom:18px}
h1{font-size:82px;line-height:.9;margin:0;font-weight:900}p{font-size:22px;color:#a9b0b5;margin:24px 0 0}
.meta{position:absolute;left:92px;right:92px;bottom:58px;display:flex;justify-content:space-between;padding-top:16px;border-top:1px solid #292d31;color:#8b9298;font-size:12px;font-weight:800}
</style></head><body><main><div class="mark"></div><div class="eyebrow">SIMA MODALIX // PRODUCT WALKTHROUGH</div><h1>RECALL</h1><p>Ctrl+F for the physical world.</p><div class="meta"><span>EDGE VISUAL MEMORY</span><span>SIMULATION MODE</span></div></main></body></html>"""


def set_caption(page: Page, kicker: str, message: str) -> None:
    page.evaluate(
        """({kicker, message}) => {
          let node = document.getElementById('recording-caption');
          if (!node) {
            node = document.createElement('div');
            node.id = 'recording-caption';
            node.style.cssText = 'position:fixed;z-index:1000;left:0;right:0;bottom:0;min-height:66px;padding:11px 28px 10px;background:rgba(5,5,5,.96);border-top:2px solid #4ba3ff;display:grid;align-content:center;gap:5px;pointer-events:none;font-family:Arial,sans-serif';
            node.innerHTML = '<span></span><strong></strong>';
            node.querySelector('span').style.cssText = 'color:#4ba3ff;font-size:9px;font-weight:900';
            node.querySelector('strong').style.cssText = 'color:#f5f7f8;font-size:14px;font-weight:800';
            document.body.appendChild(node);
          }
          node.querySelector('span').textContent = kicker;
          node.querySelector('strong').textContent = message;
        }""",
        {"kicker": kicker, "message": message},
    )


def show_end_slate(page: Page) -> None:
    page.evaluate(
        """() => {
          const node = document.createElement('div');
          node.style.cssText = 'position:fixed;z-index:2000;inset:0;background:#050505;color:#f5f7f8;display:flex;flex-direction:column;justify-content:center;padding:0 92px;font-family:Arial,sans-serif;border-top:3px solid #54d68a';
          node.innerHTML = '<span style="color:#54d68a;font-size:14px;font-weight:900;margin-bottom:20px">WALKTHROUGH COMPLETE</span><strong style="font-size:54px;line-height:1;font-weight:900">VISUAL MEMORY.<br>LOCAL BY DESIGN.</strong><p style="color:#a9b0b5;font-size:18px;margin-top:24px">5 object tracks // 2 custody events // evidence-linked answers</p><small style="position:absolute;left:92px;bottom:54px;color:#8b9298;font-size:11px;font-weight:800">SIMULATION SHOWN // CONNECT MODALIX FOR LIVE PERCEPTION AND SPEECH</small>';
          document.body.appendChild(node);
        }"""
    )


def hover_click(page: Page, selector: str) -> None:
    target = page.locator(selector)
    target.scroll_into_view_if_needed()
    target.hover()
    page.wait_for_timeout(300)
    target.click()


def record(url: str, output: Path, browser_path: str | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    page_errors: list[str] = []

    with tempfile.TemporaryDirectory(prefix="recall-recording-") as temp_dir:
        with sync_playwright() as playwright:
            launch_options = {
                "headless": False,
                "args": ["--headless=new", "--no-sandbox", "--disable-gpu", "--window-size=1280,720"],
            }
            if browser_path:
                launch_options["executable_path"] = browser_path
            browser = playwright.chromium.launch(**launch_options)
            context = browser.new_context(
                viewport={"width": 1280, "height": 720},
                record_video_dir=temp_dir,
                record_video_size={"width": 1280, "height": 720},
            )
            page = context.new_page()
            page.on("pageerror", lambda error: page_errors.append(str(error)))

            page.set_content(TITLE_HTML)
            page.wait_for_timeout(2200)
            page.goto(url, wait_until="domcontentloaded", timeout=15_000)
            page.add_style_tag(content="html{scrollbar-width:none}::-webkit-scrollbar{display:none}")
            page.wait_for_selector("#scenario-button")
            set_caption(page, "SYSTEM // LOCAL REPLAY", "The product remains interactive while the SiMa link is offline.")
            page.wait_for_timeout(1300)

            if page.locator("#scenario-action").inner_text() == "STOP SCENARIO":
                hover_click(page, "#scenario-button")
            set_caption(page, "BEAT 01 // INVENTORY", "Recall indexes the objects in view with stable identities.")
            hover_click(page, '[data-question="What\'s on the table?"]')
            page.wait_for_timeout(1600)

            set_caption(page, "BEAT 02 // OBJECT MOVEMENT", "The red mug leaves frame to the right; Recall preserves the moment.")
            hover_click(page, "#scenario-button")
            page.wait_for_function("document.querySelector('#scenario-message').textContent.includes('RED CERAMIC MUG')", timeout=6_000)
            page.wait_for_timeout(1000)

            set_caption(page, "BEAT 03 // CUSTODY", "A wrist-proximity event attributes the laptop pickup to the nearby person.")
            page.wait_for_function("document.querySelector('#scenario-message').textContent.includes('SILVER LAPTOP')", timeout=6_000)
            page.wait_for_timeout(1100)

            set_caption(page, "BEAT 04 // OFFLINE", "The same local memory remains available without an upstream network.")
            page.wait_for_function("document.querySelector('#scenario-message').textContent.includes('2 EVENTS')", timeout=6_000)
            page.wait_for_timeout(1300)

            set_caption(page, "QUERY // LOCATE MUG", "A natural-language answer returns the direction and evidence frame.")
            hover_click(page, '[data-question="Where is the red mug?"]')
            page.wait_for_timeout(1300)

            set_caption(page, "QUERY // CUSTODY", "Recall answers who took the laptop and links the pickup snapshot.")
            hover_click(page, '[data-question="Who took my laptop?"]')
            page.wait_for_timeout(1300)

            set_caption(page, "BEAT 05 // SUMMARY", "The mission summary condenses the important events and evidence.")
            hover_click(page, "#summary-button")
            page.wait_for_timeout(1500)

            set_caption(page, "EVIDENCE // AUDIT TRAIL", "Every tracked object keeps a filterable event timeline.")
            page.locator(".memory-band").scroll_into_view_if_needed()
            page.wait_for_timeout(900)
            hover_click(page, '[data-object-id="1"]')
            page.wait_for_timeout(1600)

            show_end_slate(page)
            page.wait_for_timeout(2600)

            video = page.video
            context.close()
            raw_video = Path(video.path())
            browser.close()

        if page_errors:
            raise RuntimeError(f"Browser errors during recording: {page_errors}")

        if output.suffix.lower() == ".mp4":
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise RuntimeError("ffmpeg is required for MP4 output; use a .webm output instead")
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(raw_video),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output),
                ],
                check=True,
            )
        else:
            shutil.copy2(raw_video, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="served Recall UI URL")
    parser.add_argument("--output", type=Path, default=Path("demo/recall-product-demo.mp4"))
    parser.add_argument("--browser", default=os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE"))
    args = parser.parse_args()
    record(args.url, args.output.resolve(), args.browser)
    print(f"Recorded {args.output.resolve()}")


if __name__ == "__main__":
    main()
