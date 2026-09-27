"""The swarm's pure parts: grouping, evidence, majority rules and the selection prompt's budget (no model)."""

from slm.swarm import Candidate, answer_key, collapse, display_answer, majority, selector_messages, verified_majority


def cand(i, parsed, think="reasoning " * 20, from_tool=False, errors=0, calls=None):
    return Candidate(idx=i, think=think, answer=f"#### {parsed}" if parsed is not None else "no idea", parsed=parsed,
                     key=answer_key(parsed), terminated=True, n_calls=len(calls or []), n_errors=errors,
                     calls=calls or [], from_tool=from_tool, n_tokens=50)


def test_answer_key_groups_numbers_by_value_and_text_by_case():
    assert answer_key("42") == answer_key("42.0") == answer_key("42.00") == "42"
    assert answer_key("2.5") == answer_key("2.50")
    assert answer_key("Paris.") == answer_key("paris") == "paris"
    assert answer_key(None) is None and answer_key("   ") is None
    huge = "9" * 400  # a runaway sample; float() overflows, and the eval must not die on it
    assert answer_key(huge) == huge and display_answer(huge) == huge


def test_collapse_orders_verified_support_first_and_picks_a_verified_rationale():
    cs = [cand(0, "10"), cand(1, "10"), cand(2, "10"),                      # popular, never computed
          cand(3, "12", from_tool=True, calls=[["3*4", "12"]], think="short"),  # computed, verified
          cand(4, "12"), cand(5, "7", from_tool=True, errors=1, calls=[["x", "error: y"]]),  # a call that errored is not verified
          cand(6, None)]
    groups = collapse(cs)
    assert [g.key for g in groups] == ["12", "10", "7"], "verified support outranks raw support"
    twelve = groups[0]
    assert twelve.support == 2 and twelve.verified == 1 and twelve.rationale == "short"
    assert groups[2].verified == 0
    assert majority(groups) == "10" and verified_majority(groups) == "12"
    assert all(6 not in g.members for g in groups), "an unparsable candidate joins no group"


def test_verified_majority_falls_back_to_majority_without_evidence():
    groups = collapse([cand(0, "3"), cand(1, "3"), cand(2, "5")])
    assert verified_majority(groups) == "3"


def test_selector_prompt_fits_its_budget_and_keeps_the_strongest_groups():
    long_think = "step " * 400
    cs = [cand(i, str(i), think=long_think) for i in range(30)]
    cs += [cand(100 + j, "99", think=long_think, from_tool=True, calls=[["9*11", "99"]]) for j in range(3)]
    groups = collapse(cs)
    assert groups[0].key == "99"
    msgs = selector_messages("What is 9 times 11?", groups, tok=None, budget_tokens=300, max_groups=12)
    content = msgs[0]["content"]
    assert len(content) // 4 <= 300
    assert "- Answer: 99 (agreed by 3 attempts; computed with code in 3)" in content
    assert content.startswith("What is 9 times 11?") and content.rstrip().endswith("'#### <answer>'.")
    assert content.count("- Answer:") >= 3, "answers are trimmed before they are dropped, and never below three"


def test_pair_prompt_and_pick_parsing():
    from slm.swarm import pair_messages, parse_pick

    a = collapse([cand(0, "10"), cand(1, "10", think="ten " * 300)])[0]
    b = collapse([cand(2, "12", from_tool=True, calls=[["3*4", "12"]], think="short")])[0]
    content = pair_messages("What is 3 times 4?", a, b, tok=None, budget_tokens=120)[0]["content"]
    assert content.startswith("What is 3 times 4?") and "Answer A: 10 (agreed by 2 attempts; not computed with code)" in content
    assert "Answer B: 12 (agreed by 1 attempt; computed with code in 1)" in content and content.rstrip().endswith("'#### A' or '#### B'.")
    assert len(content) // 4 <= 120 or "Reasoning: " in content, "rationales shrink to fit; the two answers are never dropped"
    assert parse_pick("#### A") == 0 and parse_pick("#### b") == 1 and parse_pick("#### B.") == 1
    assert parse_pick("#### 12") is None and parse_pick("no marker") is None and parse_pick("#### Answer B is right") is None


def test_tournament_bracket_seeds_swaps_and_falls_back_to_evidence():
    from slm.swarm import seed_pairs, tournament

    gs = collapse([cand(0, "1", from_tool=True, calls=[["1", "1"]]), cand(1, "2"), cand(2, "2"), cand(3, "3"), cand(4, "4"), cand(5, "5")])
    assert [g.key for g in gs] == ["1", "2", "3", "4", "5"]
    pairs, byes = seed_pairs(gs)
    assert [(a.key, b.key) for a, b in pairs] == [("1", "5"), ("2", "4")] and [g.key for g in byes] == ["3"]
    seen = []

    def compare(oriented):  # picks the larger number, and gives no pick for the pair holding "4"
        seen.append([(x.key, y.key) for x, y in oriented])
        return [None if "4" in (x.key, y.key) else (0 if float(x.key) > float(y.key) else 1) for x, y in oriented]

    champ, rounds = tournament(None, None, "q", gs, compare=compare)
    assert seen[0] == [("1", "5"), ("4", "2")], "second pair is presented swapped"
    r1 = rounds[0]
    assert r1[0]["winner"] == "5" and r1[1]["pick"] is None and r1[1]["winner"] == "2", "no pick -> the evidence order (support 2 beats 1)"
    assert [g for g in [rounds[0][0]["swapped"], rounds[0][1]["swapped"]]] == [False, True]
    assert champ.key == "5" and len(rounds) == 3, "5 beats 3, then the bye (2)"
    assert tournament(None, None, "q", [], compare=compare) == (None, [])
    solo, rounds = tournament(None, None, "q", gs[:1], compare=compare)
    assert solo.key == "1" and rounds == []
