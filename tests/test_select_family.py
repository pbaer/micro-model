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


def test_groups_round_trip_through_the_prompt_and_think_opens_with_the_fixed_prefix():
    from slm.rl.synth_select import THINK_PREFIX, groups_from_prompt, think_line
    from slm.swarm import Candidate, answer_key, collapse, selector_messages

    def cand(i, parsed, from_tool=False):
        return Candidate(idx=i, think="t", answer=f"#### {parsed}", parsed=parsed, key=answer_key(parsed), terminated=True,
                         n_calls=int(from_tool), n_errors=0, calls=[["1", parsed]] if from_tool else [], from_tool=from_tool, n_tokens=5)
    groups = collapse([cand(0, "10"), cand(1, "10"), cand(2, "12", True), cand(3, "12"), cand(4, "7")])
    prompt = selector_messages("How many?", groups, tok=None)[0]["content"]
    back = groups_from_prompt(prompt)
    assert [(g.answer, g.support, g.verified) for g in back] == [(g.answer, g.support, g.verified) for g in groups]
    for gold in ("12", "10", "7", "99"):
        line = think_line(back, answer_key(gold), 5)
        assert line.startswith(THINK_PREFIX), "the first think token is fixed so greedy decoding opens the span"
        assert (gold in line) or gold == "99"


def test_pair_rebuild_is_balanced_and_disjoint(tmp_path):
    from slm.rl.synth_select import rebuild_pairs
    from slm.rl.tasks import POOLED, pair_pool
    from slm.swarm import Candidate, answer_key, collapse, selector_messages

    def cand(i, parsed, from_tool=False):
        return Candidate(idx=i, think=f"think {parsed}", answer=f"#### {parsed}", parsed=parsed, key=answer_key(parsed), terminated=True,
                         n_calls=int(from_tool), n_errors=0, calls=[["1", parsed]] if from_tool else [], from_tool=from_tool, n_tokens=5)
    rows = []
    for q, gold, others in (("How many?", "12", ["10", "7"]), ("What is 5+5?", "10", ["11", "9", "8"]), ("Twice 4?", "8", ["6"]), ("Unsolved?", "99", ["1", "2"])):
        groups = collapse([cand(0, gold, True)] + [cand(i + 1, o) for i, o in enumerate(others)]) if q != "Unsolved?" else collapse([cand(i, o) for i, o in enumerate(others)])
        rows.append({"prompt": selector_messages(q, groups, tok=None)[0]["content"], "gold": gold, "source": "t", "split": "train", "has_correct": q != "Unsolved?"})
    pool = tmp_path / "select_pool.jsonl"
    pool.write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\n", encoding="utf-8")
    import slm.rl.synth_select as ss
    ss.SFT_DIR = tmp_path / "sft"
    m = rebuild_pairs(str(pool), "pairs", r"C:\slm-data\tokenizer\v1", sft_permille=500)
    total = m["counts"].get("sft:train", 0) + m["counts"].get("rl_pool", 0)
    assert total == 2 + 3 + 1, "one pair per wrong group; the unsolved pool gives none"
    assert m["counts"]["no_correct"] == 1
    tasks = pair_pool(tmp_path / "sft" / "v1" / "pairs" / "pair_pool_rl.jsonl")
    assert all(t.answer in ("A", "B") and t.task == "pair" and t.meta["answer_style"] == "free" for t in tasks)
    assert all("Answer A:" in t.prompt and "Answer B:" in t.prompt and "Reasoning: think" in t.prompt for t in tasks), "rationales are carried into the pair prompt"
    assert "pair" in POOLED
