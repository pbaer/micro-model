# Judged quality: an LLM-scored prompt suite over training

Validation loss says how well the model predicts held-out text. It does not say whether the model can name the
capital of France, finish `def fibonacci(n):`, or continue "Monday, Tuesday," without looping. This eval does:
a fixed suite of 35 prompts, run greedily on every saved checkpoint, scored blind by a judge model on three
rubrics, and charted per checkpoint on the run page. It is a *tracking* signal (does quality rise steadily, does
a stage regress something), not a benchmark to compare against other people's models.

Code: `slm/eval/quality_suite.py` (prompts + rubric), `slm/eval/quality.py` (generate / pack / ingest /
summary), portal endpoints `/api/runs/{run}/quality[/{tokens}]`, run page chart + `quality` tab.

## The suite (`SUITE_VERSION = v1`)

35 prompts in 9 categories, chosen to be things a few-hundred-million-parameter model can realistically do well
once fully trained but has not saturated early: `facts` (8), `prose` (4), `python` (6), `bash` (3), `arithmetic`
(4), `pattern` (4: list and sequence continuation), `definition` (2), `narrative` (2), `qa` (2, "why" questions).
Every prompt has two forms:

- `completion` for base checkpoints: the model continues the text (`The capital of France is`).
- `chat` for SFT / reasoning / RL checkpoints: a user turn (`What is the capital of France?`) through the chat
  format, with a forced think span when the stage trains one. The answer span is what gets judged.

The stage comes from `run.json` (`slm.utils.stage.run_stage`). A checkpoint trained to call the Python tool
(`slm.utils.stage.run_tools`: RL `tools: true`, or an SFT mixture of `*-tools` sets) answers through the tool loop
(`slm.tools.loop.sample_with_tools`, one `PySession` per prompt, up to 8 calls), so its `<|python_call|>` gets a
real result inserted; without that the model derails on an empty result and the score measures the harness, not
the model. Tool spans render as `<<code=result>>` in the stored think text, and the record carries `tool_calls`,
`tool_errors` and `termination`. Each prompt carries `expect`, a short description
of what a good answer contains, and its own `max_new_tokens` (24 for a fact, 96 for code or prose). Decoding is
greedy, so a checkpoint's outputs are reproducible. The three prompts the trainer has sampled since M1 (`rome`,
`fib`, `cap_france`) are also averaged separately as `legacy3`.

Changing any prompt or expectation bumps `SUITE_VERSION`; item ids include it, so old scores never mix with new.

## Rubrics (`RUBRIC_VERSION = v1`)

Three integers, 1-5 each, from `JUDGE_INSTRUCTIONS` in `quality_suite.py` (the packet carries the full text):

| rubric | 5 | 3 | 1 |
|---|---|---|---|
| correctness | factually / logically / syntactically right where it matters | partly right, or right with a real error | wrong, empty, nonsense |
| coherence | fluent, on topic, no repetition | readable but drifts or repeats | word salad or an immediate loop |
| task | does exactly what the prompt implies, natural format | related but incomplete or wrong format | ignores the task |

`overall` is the mean of the three. Rules the judge is given: for short-answer completions the first sentence
decides correctness and the rest only affects coherence/task (a right answer followed by a loop is 5 / 1-2 / 3-4);
code is mentally executed; length is not rewarded; the judge does not know which checkpoint an item comes from.

## Protocol

```bash
# 1. outputs: every snapshot + final.pt of a run, on CPU beside a training job (~1 min per 336M checkpoint)
python -m slm.eval.quality generate --run m8_base_stable_336m --device cpu --threads 8
# 2. blind packets for the judge: unjudged items across all checkpoints, shuffled, 40 per packet
python -m slm.eval.quality pack --run m8_base_stable_336m
# 3. judge each packet (see below) -> a json list of {item_id, correctness, coherence, task, note}
# 4. validate + store, rebuild summary.json (the portal reads it on the next page load)
python -m slm.eval.quality ingest --run m8_base_stable_336m --scores scores_01.json scores_02.json --judge claude-sonnet-4.5
python -m slm.eval.quality status --run m8_base_stable_336m
```

**Judging.** A packet (`runs/<run>/quality/packets/<run>-NN.json`) is self-contained: `instructions` (the rubric
text), then `items` with `item_id`, `category`, `mode`, `prompt`, `output`, `expect`. The judge is a Claude
subagent running a cheaper model (Sonnet), given the packet file path and told to return only the JSON array
the instructions describe. One subagent per packet, packets in parallel. The judge name goes into every stored
score (`--judge`), so a judge change is visible in `summary.json["judges"]` and can be re-run with `--replace`.
`ingest` rejects unknown ids and out-of-range or missing rubrics and reports what it rejected.

Why blind, shuffled packets: a judge that sees one prompt's answers ordered by checkpoint will assume later is
better. Item ids are `sha1(run|tokens|prompt_id|suite)[:12]`, and the packet carries nothing else about origin.

**CPU beside training.** `generate --device cpu` pins `--threads` (default 8 of 32) and drops the process to
below-normal priority; measured at 34 tok/s for the 336M model with training at full throughput (29.9K tok/s
before and during). Loading is fp32 on CPU; the numbers differ from the trainer's bf16 GPU samples in the last
digits of the logits, occasionally in a greedy token, never in what the judge is scoring.

**During a run.** Phase 2 of the second base generates the suite on the GPU at every milestone
(`eval.quality_suite: true`; a few seconds), writing the same `outputs/<tokens>.jsonl`, so only judging remains.

## Files

```
runs/<run>/quality/
  outputs/<tokens:012d>.jsonl   header {run, tokens, checkpoint, stage, suite, device, seconds} + one line per prompt
  packets/<run>-NN.json         blind judge packets (rewritten by every `pack`)
  scores.jsonl                  {item_id, tokens, prompt_id, scores{correctness,coherence,task}, note, judge, suite, rubric, judged_at}
  summary.json                  {checkpoints: [{tokens, checkpoint, stage, n_items, n_scored, overall, <rubrics>, legacy3, categories{...}}], judges, ...}
```

## Reading the chart

The run page shows `overall` (black), the three rubrics, and `legacy3` (grey, dashed) against tokens, plus a
second chart of `overall` per category. Things to expect: `facts` and `pattern` move first and saturate;
`python` and `prose` climb slowly; `coherence` lags `correctness` on base checkpoints because the right answer
arrives long before the model learns to stop (repetition loops after a correct first sentence are the dominant
failure mode of a base model at this size, and the rubric is designed to show exactly that split). SFT should
lift `task` and `coherence` sharply at a nearly flat `correctness`; RL on math should move `arithmetic` and
little else. A category that drops across a stage boundary is the regression signal this exists for.
