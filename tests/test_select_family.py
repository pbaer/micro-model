"""The RL `select` family reads the pool slm.rl.synth_select writes, and its prompts carry their own ask."""

import json

from slm.rl.tasks import POOLED, select_pool


def test_select_pool_reads_prompts_with_gold_and_no_answer_suffix(tmp_path):
    p = tmp_path / "select_pool.jsonl"
    rows = [{"prompt": "What is 2+2?\n\nSeveral attempts...\n- Answer: 4 (agreed by 3 attempts; computed with code in 2)\n\nDecide.", "gold": "4",
             "source": "gsm8k", "has_correct": True},
            {"prompt": "How many legs?\n\n- Answer: 7 (agreed by 1 attempt; not computed with code)", "gold": 8, "source": "svamp", "has_correct": False}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    tasks = select_pool(p)
    assert [t.answer for t in tasks] == ["4", "8"], "gold is a string, whatever the JSON held"
    assert all(t.task == "select" and t.meta["answer_style"] == "free" for t in tasks), "the selection prompt is complete as written"
    assert tasks[1].meta["has_correct"] is False, "pools with no right candidate stay in: the model must then solve it"
    assert "select" in POOLED


def test_select_pool_is_empty_without_the_file(tmp_path):
    assert select_pool(tmp_path / "missing.jsonl") == []
