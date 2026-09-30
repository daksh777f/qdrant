# ruff: noqa: E501  (caption strings are prose)
"""Scripted walkthrough of the mission-control UI: records a video and checks the story.

    python scripts/record_demo.py           # records docs/assets/edge-demo.webm (about 2 minutes)
    python scripts/record_demo.py --fast    # no video, no pauses: just runs the story's assertions

The UI runs with --no-autosync so every sync in the story is the one the narration says.
Requires Playwright and a Chromium (set CHROMIUM=/path/to/chrome if it is not auto-detected).
The data is synthetic; a caption bar in the video says so. If any step of the story does not
behave as narrated, the script fails, so a recording can never show something untrue.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets" / "edge-demo.webm"

CAPTION_JS = """
(text) => {
  let el = document.getElementById('__cap');
  if (!el) {
    el = document.createElement('div');
    el.id = '__cap';
    el.style.cssText = 'position:fixed;left:0;right:0;bottom:0;z-index:99999;padding:14px 24px;' +
      'background:rgba(8,12,18,.94);color:#fff;font:600 20px/1.35 system-ui,sans-serif;' +
      'border-top:2px solid #4cc2ff;text-align:center;pointer-events:none';
    document.body.appendChild(el);
  }
  el.textContent = text;
}
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def find_chromium() -> str | None:
    for cand in (os.environ.get("CHROMIUM"), "/opt/pw-browsers/chromium"):
        if cand and Path(cand).exists():
            return cand
    return shutil.which("chromium") or shutil.which("google-chrome")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="no video, no pauses")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    port = free_port()
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "loci.edge.ui",
            "--port",
            str(port),
            "--no-autosync",
            "--nodes",
            "2",
            "--node-interval",
            "0.7",
        ],  # fmt: skip
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "LOCI_NODES_DIR": tempfile.mkdtemp(prefix="loci-demo-nodes-"),
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        else:
            print("server did not start")
            return 2

        tmp_video = ROOT / ".demo-video-tmp"
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=find_chromium(), args=["--no-sandbox"])
            ctx = browser.new_context(
                viewport={"width": 1280, "height": 900},
                **({} if args.fast else {"record_video_dir": str(tmp_video),
                                         "record_video_size": {"width": 1280, "height": 900}}),
            )  # fmt: skip
            pg = ctx.new_page()
            errors: list[str] = []
            pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            pg.on("pageerror", lambda e: errors.append(str(e)))

            def pause(sec: float) -> None:
                pg.wait_for_timeout(200 if args.fast else int(sec * 1000))

            def cap(text: str, sec: float = 3.0) -> None:
                pg.evaluate(CAPTION_JS, text)
                pause(sec)

            def show(selector: str) -> None:
                """Scroll a panel to the middle of the view so the video shows what is narrated."""
                pg.evaluate(
                    "s => document.querySelector(s).scrollIntoView({block: 'center'})", selector
                )
                pg.wait_for_timeout(300)

            def top() -> None:
                pg.evaluate("window.scrollTo(0, 0)")
                pg.wait_for_timeout(300)

            def tab(name: str) -> None:
                pg.click(f"role=tab[name=/{name}/]")
                pg.wait_for_timeout(500)

            def set_net(up: bool) -> None:
                on = pg.evaluate("document.querySelector('#net').classList.contains('on')")
                if on != up:
                    pg.click("#net")
                pg.wait_for_function(
                    f"document.querySelector('#net').classList.contains('{'on' if up else 'off'}')"
                )

            def click(text: str, sec: float = 1.5) -> None:
                pg.click(f"button:has-text('{text}')")
                pause(sec)

            def ask(q: str) -> str:
                pg.fill("#q", q)
                pg.press("#q", "Enter")
                pg.wait_for_function(
                    "document.querySelector('#route').innerText.length > 0", timeout=8000
                )
                pg.wait_for_timeout(500)
                return str(pg.inner_text("#route"))

            def expect(cond: bool, what: str) -> None:
                if not cond:
                    raise AssertionError(f"story broke: {what}")

            pg.goto(f"http://127.0.0.1:{port}/")
            pg.wait_for_selector("#main:not([hidden])")
            cap(
                "LOCI Edge: a robot fleet on a flaky network, built on Qdrant Edge (synthetic sensors)",
                3.5,
            )
            pg.wait_for_function(
                "document.querySelectorAll('#tabs .tag.proc').length >= 2", timeout=30000
            )
            show("#tabs")
            cap(
                "Four devices: two scripted simulators and two real OS processes, each with its own Qdrant Edge shards.",
                4,
            )
            tab("robot-c")
            show("#remotectl")
            expect("pid" in pg.inner_text("#remote"), "process device should show its pid")
            cap(
                "robot-c is a separate process: its PID, RAM, CPU, disk and Qdrant call latencies are live telemetry.",
                4,
            )

            cap("1/6  Robot A patrols online. Repeats stay home; only surprise crosses the wire.")
            tab("robot-a")
            click("Patrol 50 steps", 2)
            click("Sync now", 2.5)
            show("#feed")
            cap(
                "Most observations were deduplicated: same thing, same place. Look at the decision feed."
            )

            cap("2/6  The network dies. Robot A keeps remembering, deciding and searching.")
            set_net(False)
            click("Patrol 10 steps", 1)
            click("Report oil spill", 1)
            click("Private note", 1)
            pause(2.5)
            txt = ask("oil spill")
            show("#route")
            expect("ANSWERED LOCALLY" in txt, f"offline answer should be local, got: {txt[:80]}")
            expect("OFFLINE" in pg.inner_text("#smeta"), "search should say it ran offline")
            cap("Offline answer, from this device only. The private note will never leave it.", 3.5)

            cap("3/6  Reconnect: the outbox drains and the diff shows convergence with the cloud.")
            set_net(True)
            click("Sync now", 2.5)
            show("#diffhead")
            expect("converged" in pg.inner_text("#diffhead"), "diff should read converged")
            pause(1.5)
            top()
            cap(
                "Converged. Sent bytes vs. naive sync are in the header; the private note stayed local.",
                3.5,
            )

            cap(
                "4/6  Robot B never saw the spill, but the cloud knows. Low local confidence: escalate."
            )
            tab("robot-b")
            txt = ask("oil spill")
            show("#route")
            expect("ESCALATED TO CLOUD" in txt, f"expected escalation, got: {txt[:80]}")
            pause(2.5)
            set_net(False)
            txt = ask("oil spill")
            expect("ANSWERED LOCALLY" in txt, "second ask should be answered locally after caching")
            cap("The answer was cached into Robot B's mirror: now it works offline, too.", 3.5)
            set_net(True)

            cap(
                "5/6  Both robots see the toolbox while offline. The cloud reconciles by place and time."
            )
            set_net(False)
            tab("robot-a")
            set_net(False)
            click("Both robots see the toolbox", 1.5)
            set_net(True)
            tab("robot-b")
            set_net(True)
            for name in ("robot-a", "robot-b", "robot-a", "robot-b"):
                tab(name)
                click("Sync now", 1.2)
            pause(1.5)
            show("#inbox")
            inbox = pg.inner_text("#inbox")
            expect("MERGED" in inbox, f"duplicate sighting should be merged; inbox: {inbox[:120]}")
            cap("Merged: same thing, same place, same time window. Every change is audited.", 3.5)

            cap(
                "6/6  A blurry view is similar but not certain. The cloud never guesses: a human decides."
            )
            click("Blurry toolbox view", 1.2)
            for name in ("robot-a", "robot-b"):
                tab(name)
                click("Sync now", 1.2)
            pause(1.5)
            show("#inbox")
            expect("REVIEW" in pg.inner_text("#inbox"), "ambiguous match should await review")
            pg.click("text=Same object: merge")
            pause(2.5)
            cap("Approved by the operator: merged.", 2.5)

            top()
            show("#features")
            n_calls = pg.inner_text("#qsub")
            expect("calls" in n_calls, "inspector should count Qdrant calls")
            cap(
                "Every Qdrant Edge call is recorded as it happens: dense HNSW, BM25, filters, decay formulas, facets.",
                4,
            )
            show("#ops")
            pause(2)

            cap(
                "A real outage: robot-c's uplink drops. It keeps patrolling offline; its heartbeat goes silent.",
                3,
            )
            pg.request.post(
                f"http://127.0.0.1:{port}/api/devices/robot-c/command", data={"outage_s": 9}
            )
            tab("robot-c")
            show("#remotectl")
            pg.wait_for_function(
                "document.querySelector('#remname').innerText.includes('no heartbeat')",
                timeout=30000,
            )
            cap(
                "Offline: no heartbeat. Its observations are safe in its own outbox on its own disk.",
                3.5,
            )
            pg.wait_for_function(
                "document.querySelector('#remname').innerText.includes('online')", timeout=40000
            )
            cap("Uplink back: the device syncs its backlog by itself. Nothing was lost.", 3.5)
            tab("robot-a")

            txt = ask("banana submarine")
            show("#route")
            expect("ABSTAINED" in txt, f"junk question should be refused, got: {txt[:80]}")
            cap("And when nobody is confident, the robot says nothing rather than guessing.", 3.5)
            show("#evidence")
            expect(
                "REAL DATA" in pg.inner_text("#evidence"),
                "evidence panel should show real-data results",
            )
            cap(
                "Evidence, not claims: results on public human-labelled datasets and a 12-check stress test, one command each.",
                5,
            )

            expect(not errors, f"console errors: {errors}")
            video = pg.video
            ctx.close()  # finalises the recording; it must be saved before the browser closes
            if not args.fast and video is not None:
                OUT.parent.mkdir(parents=True, exist_ok=True)
                video.save_as(str(OUT))
                shutil.rmtree(tmp_video, ignore_errors=True)
                print(f"recorded {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
            browser.close()
        print("story OK")
        return 0
    except AssertionError as exc:
        print(f"FAIL: {exc}")
        return 1
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    sys.exit(main())
