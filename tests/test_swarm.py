"""The swarm's pure parts: grouping, evidence, majority rules and the selection prompt's budget (no model)."""

from slm.swarm import Candidate, answer_key, collapse, majority, selector_messages, verified_majority


def cand(i, parsed, think="reasoning " * 20, from_tool=False, errors=0, calls=None):
    return Candidate(idx=i, think=think, answer=f"#### {parsed}" if parsed is not None else "no idea", parsed=parsed,
                     key=answer_key(parsed), terminated=True, n_calls=len(calls or []), n_errors=errors,
                     calls=calls or [], from_tool=from_tool, n_tokens=50)


def test_answer_key_groups_numbers_by_value_and_text_by_case():
    assert answer_key("42") == answer_key("42.0") == answer_key("42.00") == "42"
    assert answer_key("2.5") == answer_key("2.50")
    assert answer_key("Paris.") == answer_key("paris") == "paris"
    assert answer_key(None) is None and answer_key("   ") is None


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
