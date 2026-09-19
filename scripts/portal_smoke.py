"""Click through every page of a LIVE portal (real data) with Playwright and report JS errors,
hangs, raw template text, and per-page timings. Complements the hermetic e2e suite.

    python scripts/portal_smoke.py --url http://127.0.0.1:8765 [--generate]
"""

from __future__ import annotations

import argparse
import sys
import time

from playwright.sync_api import sync_playwright

BAD = ("divThis", "undefined", "NaN", "[object Object]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--generate", action="store_true", help="also load the newest checkpoint into slot A (CPU) and generate")
    a = ap.parse_args()
    problems: list[str] = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
        pg.on("console", lambda m: problems.append(f"console.error: {m.text[:200]}") if m.type == "error" else None)

        def settle(ms=500):
            pg.wait_for_timeout(ms)
            t = time.time()
            assert pg.evaluate("1+1") == 2
            if time.time() - t > 2:
                problems.append("page unresponsive (>2 s to evaluate)")
            body = pg.inner_text("main")
            for bad in BAD:
                if bad in body:
                    problems.append(f"raw text {bad!r} at {pg.url}: {body[:120]!r}")

        def click_all(skip=()):
            seen = set()
            for _ in range(80):
                target = None
                for btn in pg.locator("main button").all():
                    try:
                        label = btn.inner_text().strip()
                    except Exception:  # noqa: BLE001
                        continue
                    if label in seen or label in skip or not btn.is_visible() or not btn.is_enabled():
                        continue
                    target = (btn, label)
                    break
                if not target:
                    break
                btn, label = target
                seen.add(label)
                t0 = time.time()
                btn.click(timeout=3000)
                settle()
                if time.time() - t0 > 4:
                    problems.append(f"slow action {label!r} on {pg.url}: {time.time() - t0:.1f}s")
            return seen

        def cycle_selects(limit=3):
            for sel in pg.locator("main select").all():
                if not sel.is_visible():
                    continue
                for o in sel.locator("option").all()[:limit]:
                    v = o.get_attribute("value")
                    if v:
                        sel.select_option(v)
                        settle()

        t0 = time.time()
        pg.goto(a.url + "/#/")
        settle(800)
        print(f"home ok ({time.time() - t0:.1f}s)")
        rows = pg.locator("tr.click").all()
        print(f"runs: {len(rows)} rows")
        for i in range(len(rows)):
            pg.goto(a.url + "/#/")
            settle(600)
            pg.locator("tr.click").nth(i).click()
            settle(1200)
            name = pg.locator("h1").inner_text().split("\n")[0]
            clicked = click_all(skip=("log y",))
            pg.get_by_role("button", name="charts").click()
            settle()
            for x in ("update", "time", "tokens", "log y"):
                pg.get_by_role("button", name=x, exact=True).click()
                settle()
            cycle_selects()
            print(f"  run {name!r}: clicked {sorted(clicked)}")
        for page in ("data", "tokenizer", "arch", "inference"):  # overview already exercised above
            t0 = time.time()
            pg.goto(a.url + "/#/" + page)
            settle(1200)
            if page == "data":
                clicked = set()
                for tab in ("sources", "documents"):
                    pg.get_by_role("button", name=tab, exact=True).click()
                    settle(800)
                    clicked |= click_all(skip=("‹", "›", "‹ prev", "next ›", "‹ all recipes", "recipes", "sources", "documents"))
                    cycle_selects(2)
            elif page == "inference" and a.generate:
                sel = pg.locator("main select").first
                opts = [o.get_attribute("value") for o in sel.locator("option").all() if o.get_attribute("value")]
                if opts:
                    sel.select_option(opts[0])
                    pg.locator("main select").nth(1).select_option("cpu")
                    pg.get_by_role("button", name="load").first.click()
                    pg.wait_for_function("document.querySelector('main').innerText.includes('cpu/float32')", timeout=120000)
                    pg.locator("input[type=number]").nth(3).fill("16")
                    pg.get_by_role("button", name="generate").click()
                    pg.wait_for_function("document.querySelectorAll('.rawout span').length > 3", timeout=120000)
                    settle()
                clicked = {"load", "generate"}
            else:
                clicked = click_all(skip=("generate", "load", "cancel", "score prompt (teacher-forced)", "unload"))
                cycle_selects(2)
            print(f"  {page}: clicked {sorted(clicked)} ({time.time() - t0:.1f}s)")
        b.close()
    print("\nPROBLEMS:" if problems else "\nno problems found")
    for x in problems:
        print(" -", x)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
