"""Every dashboard chart goes through components/chart.js; Peter's rules for them (2026-10-10): hover values in a card
beside the cursor (not in the legend), no colour boxes in the legend, no horizontal crosshair line."""

import re
from pathlib import Path

STATIC = Path("slm/portal/static")


def test_chart_component_rules():
    src = (STATIC / "components" / "chart.js").read_text(encoding="utf-8")
    assert "legend: { show: false }" in src, "uPlot's live legend (values in the series, colour boxes) is off"
    assert re.search(r"cursor: \{ y: false", src), "no horizontal crosshair line"
    assert not re.search(r"\bsync:\s*\{", src),"no cursor sync: the vertical line shows only in the chart under the pointer"
    assert "chart-card" in src and "setCursor" in src, "hover values go to the card beside the cursor"
    assert "border-top:2px" in src, "the legend shows a line in the series colour, not a box"
    assert "card.remove()" in src, "the card leaves the page with its chart"
    assert "rows.sort((a, b) =>" in src and "highest value on top" in src, "card rows sorted by value, highest first"


def test_no_page_builds_its_own_uplot():
    for f in STATIC.rglob("*.js"):
        if "vendor" in f.parts or f.name == "chart.js":
            continue
        assert "new uPlot" not in f.read_text(encoding="utf-8"), f"{f} bypasses components/chart.js"
