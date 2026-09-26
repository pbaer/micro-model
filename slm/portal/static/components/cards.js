import { h } from "preact";
import htm from "htm";

const html = htm.bind(h);

/** The explanation behind every "?" in the command center, keyed by concept (components/info.js renders them).
 *  One card per concept, reused wherever the concept appears. Pattern: what it is, then how to read it (what good
 *  and bad look like), then a muted line on how it connects to the rest of the pipeline. Concepts, not numbers:
 *  the numbers belong to the page, and docs/results.md says what they have been. */
export const CARDS = {
  // ------------------------------------------------------------------ overview and shared run metrics
  gpu_tile: { t: "GPU status", b: html`
    <p>The one GPU everything shares, read from <code>nvidia-smi</code> every 5 s: memory used / total, utilization,
      power and temperature, and the runs whose metrics are still being written.</p>
    <ul><li>During training, utilization near 100% is normal; lower means the GPU is waiting (evals, data, another process).</li>
      <li>On Windows, memory past 16 GiB does not fail: it spills into host RAM and runs ~60x slower. Above ~14.5 GiB, treat it as a failure.</li>
      <li>Above 80 °C the trainers log a <code>warn</code> event, because sustained heat lowers clocks and throughput.</li></ul>
    <p class="see">While a run is live, the Inference page loads models on the CPU so it does not compete for this memory.</p>` },
  pipeline: { t: "The training pipeline", b: html`
    <p>A model is built in stages, and each stage starts from the weights of the one before it (<code>init_from</code>):</p>
    <ul><li><b>Pretraining</b>: next-token prediction on web prose, math and Python, from random weights to a base model.</li>
      <li><b>Instruction SFT</b>: chat-formatted conversations; only the assistant's tokens are trained.</li>
      <li><b>Reasoning SFT</b>: a <code>&lt;|think|&gt;</code> span before every answer, with Python tool calls inside it.</li>
      <li><b>RL (GRPO)</b>: the model samples its own answers and a program checks them; no human labels or reward model.</li>
      <li><b>Context extension</b>: RoPE scaling plus training on long documents.</li></ul>
    <p class="see">"from …" under a run name is its parent checkpoint. The Data page's <i>chain</i> view adds up everything a checkpoint has seen.</p>` },
  run_status: { t: "Run status", b: html`
    <p>Read from the run's <code>metrics.jsonl</code>, not from the process:</p>
    <ul><li><b>running</b>: a record arrived in the last 15 minutes.</li>
      <li><b>finished</b>: the run reached its token (or step) budget and wrote <code>final.pt</code>.</li>
      <li><b>stopped</b>: it ended on purpose: Ctrl-C, a <code>STOP</code> file, or a guard. <code>latest.pt</code> was saved.</li>
      <li><b>stale</b>: silent for 15 minutes without a stop record. The process died (crash, reboot); rerunning the same command resumes from <code>latest.pt</code>.</li>
      <li><b>empty</b>: no records yet.</li></ul>` },
  tokens_seen: { t: "Tokens: this run and in total", b: html`
    <p>Tokens are the unit of training: every token in a training row is one prediction the model is graded on.
      The first number is what this run processed (RL runs count steps instead). "seen" is the cumulative count
      along the <code>init_from</code> chain: every token these weights have trained on since random init.</p>
    <p>Training compute is roughly 6 × parameters × tokens, so tokens are also the fair axis for comparing
      checkpoints. Post-training stages are tiny next to pretraining; almost all of a model's knowledge comes from the base.</p>` },
  loss_col: { t: "Loss", b: html`
    <p>The latest training loss: mean cross-entropy per predicted token, in nats. It is −ln of the probability the
      model gave the token that actually came next. A uniform guess over the 32K vocabulary scores ln 32768 ≈ 10.4.</p>
    <p>Only compare losses measured on the same data: an SFT loss (assistant tokens only) and a pretraining loss are
      different quantities. RL runs show the mean rollout reward instead, because their loss is ≈0 by construction.</p>` },
  result_col: { t: "Result", b: html`
    <p>The number each kind of run is judged by. Pretraining and SFT: the <b>best validation loss</b>, the lowest
      loss on held-out text so far. RL: <b>held-out accuracy</b>, greedy answers to problems the policy never trained on.</p>
    <p class="see"><code>best.pt</code> is the checkpoint at this value; it is often the one the next stage starts from.</p>` },
  quality_col: { t: "Judged quality (1-5)", b: html`
    <p>The mean score of a fixed 35-prompt suite, answered greedily by the latest judged checkpoint and graded blind
      by an LLM judge on correctness, coherence and task (1-5 each). It catches what loss cannot: whether answers are
      right, readable and do what was asked.</p>
    <p>It is a tracking signal, not a benchmark. Neighbouring checkpoints wobble by about ±0.25; a drop across a
      stage boundary is the thing to look for.</p>` },
  tok_s: { t: "Throughput (tokens/second)", b: html`
    <p>Training tokens processed per second over the last log window; <b>ema</b> is a smoothed version (it drives the ETA).
      It is set by model size, sequence length, microbatch size, <code>torch.compile</code> (a ~6x win here) and the
      attention kernel.</p>
    <p><b>Flat</b> is healthy. <b>Dips</b> at regular intervals are evals and sample generation. A <b>lasting drop</b>
      means something changed: thermal throttling, VRAM spilling into host memory, or another process on the GPU.
      RL runs do not report it; their time goes into sampling.</p>` },
  data_readiness: { t: "Data readiness", b: html`
    <p>What is prepared for training, per tokenizer version: pretraining sources converted to token-id shards, and
      chat-formatted SFT sets with their loss masks. A run can only train on data tokenized with its own tokenizer:
      the same text under another tokenizer is a different sequence of ids.</p>` },

  // ------------------------------------------------------------------ run page: tiles and charts
  eta: { t: "ETA", b: html`
    <p>Remaining tokens divided by the smoothed throughput (RL: remaining steps × the last step time). Evals and
      snapshots are not modelled separately, so it jumps a little around them and settles back.</p>` },
  elapsed: { t: "Elapsed vs initial estimate", b: html`
    <p>Wall-clock training time so far. The initial estimate is the whole run's duration predicted from the first
      window after warmup and stored in <code>run.json</code>. If elapsed runs well past it, the run slowed down:
      check the throughput, VRAM and GPU charts.</p>` },
  train_loss: { t: "Training loss", b: html`
    <p>Cross-entropy on the batches being trained, averaged per target token over the log window. exp(loss) is the
      perplexity: the model is as unsure as a uniform choice among that many tokens.</p>
    <p><b>Typical shape</b>: a steep fall over the first few hundred million tokens, then a slow power-law decline, and a
      visible extra drop while the learning rate decays. <b>Spikes</b> point to instability; check the gradient norm at
      the same x. In SFT only assistant tokens count.</p>` },
  val_loss: { t: "Validation loss", b: html`
    <p>The same loss on a fixed set of held-out windows: about 1% of documents, chosen by a hash of their text and never
      trained on. Measured every <code>eval.every_tokens</code>. <b>ppl</b> = exp(val loss); <b>best</b> is the lowest
      so far, saved as <code>best.pt</code>.</p>
    <p>In pretraining most data is read once, so train and val stay close. <b>Val flat or rising while train keeps
      falling</b> means memorization: repeated data (see epochs on the Data page) or too many SFT epochs.</p>
    <p class="see">Comparable only between runs with the same validation mixture and tokenizer.</p>` },
  loss_chart: { t: "Train and validation loss", b: html`
    <p>The thin line is training loss at every log step; the points are validation loss at every eval. Training loss
      is measured on batches the model has not been updated on yet, so in single-pass pretraining it is already an
      estimate of generalization. Validation uses a fixed set of held-out windows, which makes it less noisy and
      comparable from one eval to the next.</p>
    <p>Turn on <b>log y</b> to see late progress. A widening gap with val above train is the memorization signal.
      A val jump at a phase boundary usually means the data changed, not that the model got worse.</p>` },
  val_drift: { t: "Task validation vs pretraining drift", b: html`
    <p>Stages after pretraining track two validation losses. <b>val</b> is the stage's own task, e.g. assistant tokens
      of held-out conversations. <b>pretrain val</b> is held-out pretraining text (<code>extra_val_mixture</code>),
      which measures how much general language modelling the stage is giving up.</p>
    <p>Task val should fall. A small, levelling rise in pretrain val is the normal price of specializing; a steep or
      still-climbing one means forgetting: too high a learning rate, too many epochs, or too little rehearsal data.
      For context-extension runs the drift line is short-context loss, the "≤ 5% cost" check.</p>` },
  needle: { t: "Needle retrieval and effective context", b: html`
    <p>A short fact (the needle) is hidden in real text at several depths of a prompt of length L, and the model is asked
      for it. Each (length, depth) cell is an accuracy. <b>Effective context</b> is the longest length at which even the
      worst depth reaches 80%.</p>
    <p>The configured context (the RoPE table size) says what the model can accept, not what it can use. Retrieval
      typically fails first at the far edge of the trained window, where long-range dependencies were rarest in training.</p>
    <p class="see">Project rule: long context is claimed only where this gate holds on real text with n ≥ 16.</p>` },
  needle_chart: { t: "Needle retrieval by context length", b: html`
    <p>One colour per prompt length. <b>Solid</b> = accuracy averaged over depths, <b>dashed</b> = the worst depth,
      which is what the gate uses: a model that cannot find the fact at one depth does not really have that context.</p>
    <p><b>Heavy lines</b> are a sweep: every milestone snapshot re-measured afterwards at one sample count on one
      fixed haystack. In-run evals use small n, and small n is noisy: at n=16 a true 90% cell reads below 80% about a
      fifth of the time; at n=64 about 1%.</p>
    <p>Watch the longest length's dashed line: it moves first, especially while the learning rate decays.</p>` },
  quality_chart: { t: "Judged quality over training", b: html`
    <p>Every checkpoint answers the same 35 prompts greedily; an LLM judge scores each answer 1-5 without knowing which
      checkpoint wrote it. <b>overall</b> (black) is the mean of the three rubrics over all categories except bash
      (shell was dropped from the data and allowed to fade); the thin lines are the rubrics; <b>legacy 3</b> (grey) is
      the three prompts the trainer has sampled since the first run.</p>
    <ul><li>Base models: coherence lags correctness, because the right answer arrives long before the model learns to stop (repetition loops).</li>
      <li>SFT should lift task and coherence at nearly flat correctness.</li>
      <li>Expect ±0.25 between neighbouring checkpoints; trust trends and stage-boundary jumps.</li></ul>` },
  quality_cat_chart: { t: "Judged quality by category", b: html`
    <p>The overall score split by prompt category: facts, prose, python, arithmetic, pattern (list and sequence
      continuation), definition, narrative, qa ("why" questions), and bash (still scored, excluded from overall).</p>
    <p>Categories hold 2-8 prompts each, so one answer moves a line a lot. Facts and pattern move first and saturate;
      python and prose climb slowly. The useful signal is a category that drops across a stage boundary, e.g. when
      tool SFT made every category that emitted a Python call regress.</p>` },
  tool_misfire: { t: "Tool misfire", b: html`
    <p>The share of suite prompts where running code cannot help (all but arithmetic, and bash, which is excluded) on which the model
      reached for the Python sandbox anyway: a call inside the think span, or tool markup in the answer. It needs no
      judge, so it is measured on every checkpoint as soon as outputs exist. Target: 0; the tool-SFT gate was ≤ 0.10.</p>
    <p>Why it exists: a model trained on tool conversations learns to route on surface form, not need. One checkpoint
      answered "What is the capital of France?" with an invented Python call. The fix was data that shows when <i>not</i>
      to use the tool.</p>
    <p class="see">No suite prompt needs a tool, so this measures only misuse; correct use is measured by <code>slm.eval.reasoning --tools</code>.</p>` },
  lr: { t: "Learning rate", b: html`
    <p>The step size of each AdamW update, set before every update from a schedule indexed by tokens: a linear warmup,
      then flat (WSD), cosine or constant, usually ending in a decay.</p>
    <p>Too high shows up as loss spikes or divergence; too low as a curve that flattens early. Loss typically drops
      noticeably during the final decay, because the model settles into a minimum it was bouncing around.
      RL runs use a constant, much smaller rate.</p>
    <p class="see">The Architecture page's hparams tab draws any config's schedule with the trainer's own function.</p>` },
  grad_norm: { t: "Gradient norm", b: html`
    <p>The size (L2 norm) of the whole gradient before clipping. Updates are clipped to norm 1.0, so a single bad batch
      cannot throw the weights far.</p>
    <p><b>Healthy</b>: high during warmup, then settling and drifting slowly. <b>Spikes</b> that coincide with loss spikes
      point to instability (learning rate too high, bad data). <b>Sitting above the clip</b> for long stretches means
      every update is clipped, so the effective step is smaller than the schedule says.</p>` },
  step_time: { t: "Step time", b: html`
    <p>Wall-clock time per optimizer update, averaged over the log window.</p>
    <ul><li><b>fwd / bwd</b>: forward and backward are timed together with CUDA events and shown as a 1:2 split. They
        interleave per microbatch, so the split is an assumption, not a measurement.</li>
      <li><b>opt</b>: gradient clipping plus the AdamW step.</li>
      <li><b>data</b>: host time waiting for batches. It should be near zero, since the loader prefetches on a thread.</li></ul>
    <p>In RL runs "fwd" is rollout collection (sampling, scoring, reference log-probs) and "bwd" is the optimization passes.</p>` },
  vram: { t: "VRAM: peak allocated and reserved", b: html`
    <p><b>peak</b> is the most memory live tensors used at once: weights, gradients, Adam state, activations, logits.
      <b>reserved</b> is what PyTorch's caching allocator holds from the driver, including gaps it cannot reuse.</p>
    <p>On Windows (WDDM) the GPU does not raise out-of-memory at 16 GiB. It spills into host RAM and runs ~60x slower,
      and <i>reserved</i> is what crosses the line. Keep peak around 11 GiB and treat &gt; ~14.5 GiB as a failure. A
      generation eval can raise reserved for good (a KV cache needs large contiguous blocks); the mild form is a steady
      ~8% slowdown at full utilization, which is why trainers free the cache after evals.</p>` },
  gpu_temp: { t: "GPU temperature and power", b: html`
    <p>Sampled from <code>nvidia-smi</code> every 2 s by the trainer and stored in every train record. Above 80 °C the
      trainer logs a <code>warn</code> event (at most one per 5 minutes).</p>
    <p>Sustained heat makes the card lower its clocks, which shows up as lower throughput. Power falling while
      utilization stays at 100% is the signature of throttling, or of the GPU waiting on memory (e.g. a VRAM spill).</p>` },

  // ------------------------------------------------------------------ run page: RL
  rl_reward: { t: "Reward and success rate", b: html`
    <p>The mean reward over every rollout of a step (prompts per step × group size), and the share a verifier judged
      correct. Reward depends on the scheme: 1/0 for binary, partial credit for constraint tasks, 1 vs 0.5 for
      correct-with-a-tool vs correct-without.</p>
    <p>Per-step values come from a few dozen samples, so read the trend. <b>Reward rising while held-out accuracy stays
      flat</b> means the policy is fitting its training prompts or exploiting the reward scheme; read the rollouts.</p>` },
  heldout_acc: { t: "Held-out accuracy", b: html`
    <p>Greedy answers to prompts from the held-out split (10% of prompts, chosen by a hash of their text, never sampled
      in training), checked by the verifier. This is what RL is judged by: <code>best.pt</code> is the best held-out
      step, and an RL run's "val loss" is 1 − held-out accuracy.</p>
    <p><b>train</b> is the same measurement on an equal-sized sample of training prompts. Train pulling away from held-out
      means memorization. Held-out sets are small (one or two prompts are several points), so do not read one eval.</p>` },
  kl_ref: { t: "KL to the reference policy", b: html`
    <p>How far the policy has moved from the frozen checkpoint it started from, per completion token. It is estimated
      with k3 = e<sup>ref−new</sup> − (ref − new) − 1, which is never negative. The objective subtracts
      <code>kl_coef</code> × KL, and <code>kl_stop</code> ends the run when it goes over budget.</p>
    <p><b>Near zero</b>: the policy has not moved, so it cannot have learned (or collapsed). One run used 0.2% of its
      budget and its reward stayed flat until the learning rate went up. <b>A steady climb</b> is learning. <b>A sudden jump
      together with rising entropy</b> is collapse.</p>
    <p class="see">The KL budget, not the learning rate, is RL's safety mechanism.</p>` },
  kl_entropy: { t: "KL to reference and entropy", b: html`
    <p><b>KL</b>: how far the policy has drifted from the frozen starting checkpoint (k3 estimator, per token).
      <b>entropy</b>: the mean uncertainty of the policy's next-token distribution over its own completions, in nats.
      Low = confident; RL normally lowers it slowly.</p>
    <p><b>Collapse</b> looks like entropy shooting up and staying up while KL climbs and completions turn to garbage
      (the m6 try-1 run: entropy 0.8 → 5, KL 0 → 0.28). The guards <code>entropy_stop</code> / <code>kl_stop</code>
      test the mean over the last <code>guard_window</code> steps, because single steps are noisy: a one-step entropy
      spike once stopped a healthy run.</p>` },
  completion_len: { t: "Completion length", b: html`
    <p>Mean tokens per sampled completion. <b>malformed</b> is the share scoring 0 for form alone: no think span when one
      is required, no answer line, or a tool call outside the think span. <b>no-signal groups</b> is the share of
      prompts whose rollouts all scored the same, so they teach nothing (next card over).</p>` },
  length_chart: { t: "Completion length", b: html`
    <p>Tokens per rollout: mean, and the means over correct and wrong ones. Wrong answers are often longer: they ramble,
      loop, or hit the token limit.</p>
    <p>A steady decline under RL is the policy becoming terser. That is fine for arithmetic, and a warning when it spreads
      to answers that should be prose. Lengths pinned at <code>max_new_tokens</code> mean rollouts are being cut off
      (see length-terminated).</p>` },
  rl_health: { t: "Malformed, length-terminated, no-signal", b: html`
    <ul><li><b>malformed</b>: failed on form (no think span, no parsable answer, a tool call outside the think span) and scored 0.
        It should fall early in RL; the policy learns the format first.</li>
      <li><b>length-term</b>: hit <code>max_new_tokens</code> without closing the turn.</li>
      <li><b>no-signal</b>: groups whose rollouts all got the same reward. With group-relative advantages they contribute
        exactly nothing. The share is high when tasks are too easy (all right) or too hard (all wrong); useful tasks sit in between.</li></ul>` },
  rl_update: { t: "Clip fraction, |advantage|, group reward std", b: html`
    <p><b>clip</b>: share of tokens whose probability ratio new/old left [1−ε, 1+ε] and was clipped. The ratio starts at 1
      each step, so a rising clip fraction means updates are big enough for the trust region to be doing the limiting.</p>
    <p><b>|adv|</b>: mean size of the advantages, the per-rollout learning signal. <b>group std</b>: mean spread of rewards
      within a group. With std-normalized advantages, |adv| looks similar whenever a group has any spread at all, so
      group std is the better gauge of how much the rewards actually differ.</p>` },
  policy_objective: { t: "Policy objective", b: html`
    <p>The clipped GRPO surrogate −Σ min(ρA, clip(ρ, 1±ε)A), summed over completion tokens and divided by the
      token count. Advantages are centred within each group per <em>sequence</em>, but the sum is per <em>token</em>,
      so sequences with more tokens weigh more.</p>
    <p>That is why it is usually not zero. A wrong rollout tends to run to the length limit while a right one stops
      early, so negative-advantage tokens outnumber positive ones and the value sits above zero -- around +0.2 to
      +0.5 in the M9 runs. A drop toward zero or below means wrong answers are getting shorter or right ones
      longer, not that the policy is worse. Ratio clipping only matters after the first optimisation pass.</p>
    <p>It is not a progress measure; do not read it like a loss. Progress is reward, held-out accuracy and KL.</p>` },

  // ------------------------------------------------------------------ run page: tabs
  tab_charts: { t: "Charts, x axis and log y", b: html`
    <p>Every chart shares one cursor; hover a legend entry to highlight that series.</p>
    <ul><li><b>tokens</b> (default): the schedule, evals and milestones are all indexed by tokens, so runs with different batch sizes line up.</li>
      <li><b>update</b>: optimizer steps (for RL, GRPO steps).</li>
      <li><b>time</b>: wall-clock since the first record, including pauses such as evals.</li>
      <li><b>log y</b>: loss curves are close to power laws, so late progress is only visible on a log scale.</li></ul>` },
  tab_milestones: { t: "Milestones", b: html`
    <p>Every <code>milestone_tokens</code> (100M in most runs) the trainer saves a bf16 weights snapshot
      (<code>snap_&lt;tokens&gt;.pt</code>), refreshes <code>report.html</code>, and logs how long that segment took.
      Segment tok/s makes slow stretches easy to find.</p>
    <p class="see">These snapshots are what the judged suite and the needle sweep re-evaluate after the run.</p>` },
  tab_samples: { t: "Samples", b: html`
    <p>A few fixed prompts completed at a regular token interval during training. <b>greedy</b> (temperature 0) is the
      single most likely continuation and is reproducible; <b>sampled</b> draws from the distribution and shows its range.</p>
    <p>The same prompt over time is the most direct view of what training buys: babble, then grammar, then on-topic, then
      correct. Greedy repetition loops are typical of small base models and fade with SFT.</p>` },
  tab_quality: { t: "The judged quality suite", b: html`
    <p>35 prompts in 9 categories. Each has a completion form for base checkpoints ("The capital of France is") and a chat
      form for SFT/RL ones. Answers are greedy, so they are reproducible; tool-trained checkpoints answer through the real
      tool loop.</p>
    <p>Answers go to an LLM judge in shuffled packets with opaque ids, so it cannot tell which checkpoint wrote what and
      cannot assume later is better. Each answer gets correctness, coherence and task (1-5) and a short note.</p>
    <p class="see">Protocol and rubric: docs/quality_eval.md.</p>` },
  correctness: { t: "Rubric: correctness", b: html`
    <p>Is it factually, logically or syntactically right where it matters? <b>5</b> right; <b>3</b> partly right, or right
      with a real error; <b>1</b> wrong, empty or nonsense. For short-answer completions the first sentence decides it; code
      is mentally executed.</p>` },
  coherence: { t: "Rubric: coherence", b: html`
    <p>Is it fluent and on topic, without repetition? <b>5</b> fluent; <b>3</b> readable but drifts or repeats; <b>1</b> word
      salad or an immediate loop. A right answer followed by a loop scores high correctness and low coherence, the
      signature failure of a small base model.</p>` },
  task: { t: "Rubric: task", b: html`
    <p>Does it do what the prompt implies, in a natural format? <b>5</b> exactly; <b>3</b> related but incomplete or in the
      wrong format; <b>1</b> ignores the task. SFT is what moves this one most.</p>` },
  tab_checkpoints: { t: "Checkpoints", b: html`
    <ul><li><b>latest.pt</b>: everything needed to resume exactly: fp32 weights, AdamW state, the data loader's per-source
        positions, counters and RNG. Written every few minutes and on stop.</li>
      <li><b>snap_&lt;tokens&gt;.pt</b>: bf16 weights only, at every milestone.</li>
      <li><b>best.pt</b>: bf16 weights at the best validation loss (RL: best held-out accuracy).</li>
      <li><b>final.pt</b>: bf16 weights at the end. Weights only: no optimizer, no data positions.</li>
      <li><b>step_&lt;n&gt;.pt</b>: RL training snapshots.</li></ul>
    <p class="see">A continuation that should not re-read its parent's data must point <code>init_loader_from</code> at <code>latest.pt</code>, not <code>final.pt</code>.</p>` },
  tab_events: { t: "Events", b: html`
    <p>The run's log, newest first: <b>start</b>, <b>resume</b> (restarted from <code>latest.pt</code>), <b>checkpoint</b>
      (<code>latest.pt</code> written, with per-source data positions), <b>warn</b> (GPU temperature), <b>stop</b> (Ctrl-C,
      STOP file or a guard, with the reason) and <b>finish</b>. Gaps between stop and resume are excluded from elapsed time.</p>` },
  tab_config: { t: "Config", b: html`
    <p>The training config exactly as the run started with it (every hyperparameter is config-driven), the model config,
      and the environment (git commit, tokenizer hash, versions).</p>
    <p>A run that continues another inherits three things through three separate switches: <code>init_from</code>
      (weights), <code>init_optimizer</code> (AdamW moments) and <code>init_loader_from</code> (data positions). Miss the
      last one and every source restarts at token 0: one phase spent most of its first 311M tokens re-reading its parent's data.</p>` },

  // ------------------------------------------------------------------ data page
  recipe: { t: "Recipe", b: html`
    <p>The data side of one training job. A <b>plan</b> is the YAML config; a <b>run</b> is what that job actually started
      with (from its <code>run.json</code>). Pretraining and SFT recipes are mixtures of sources with weights; an RL recipe is
      a list of task generators, because RL trains on the model's own samples.</p>` },
  seq_len: { t: "Sequence length", b: html`
    <p>Tokens per training row. The model only ever attends within a row, so this is the longest dependency it can learn
      from. Attention cost grows quadratically with it and activation memory linearly, so bulk pretraining uses 2K rows and
      later phases 4K or 8K.</p>` },
  init_from: { t: "init_from: where the weights start", b: html`
    <p>"random init" trains from scratch; otherwise the checkpoint of another run. Continuing is three separate switches:
      <code>init_from</code> (weights), <code>init_optimizer</code> (AdamW state) and <code>init_loader_from</code> (the
      data positions, from the parent's <code>latest.pt</code>). Without the last, every source restarts at token 0 and the
      run re-reads what the parent already trained on.</p>` },
  plan_differs: { t: "Plan vs run", b: html`
    <p>Compares the YAML file as it is today with the config the run recorded at launch. "differs" usually means the file
      was edited for a later phase. The run's own record is the truth; this page and the run page show that.</p>` },
  mix_weight: { t: "Mixture weight", b: html`
    <p>The probability that a training row is drawn from this source. The loader picks a source per row with a seeded
      RNG, then takes that source's next window.</p>
    <p>Weights are <b>token</b> shares, not document or conversation shares. A source of short conversations gets far more
      conversations per token than one of long documents: once, a 33% token share of short tool conversations was 34
      tool conversations for every chat conversation.</p>` },
  planned: { t: "Planned tokens", b: html`<p>The recipe's total tokens × this source's weight: how much of it the run expects to read.</p>` },
  available: { t: "Available tokens", b: html`<p>Tokens in this source's training split on disk, for the recipe's tokenizer.</p>` },
  epochs: { t: "Epochs", b: html`
    <p>Planned ÷ available: how many times the run reads this source. Below 1, part of it is never seen. Above 1, the model
      sees the same text again. A few passes over a small, high-quality set are normal; heavy repetition is memorized
      (training loss keeps falling, validation does not). Marked red above 1.5.</p>` },
  consumed: { t: "Consumed", b: html`
    <p>The loader's own count for this source at the run's last checkpoint (tokens, and fractional epochs). This is what
      actually happened, not the plan. These positions are what <code>init_loader_from</code> hands to a continuation.</p>` },
  val_tokens: { t: "Validation tokens", b: html`
    <p>The held-out split: about 1% of documents, chosen by a hash of their text so the choice is stable and can never
      leak into training. Validation loss is computed on a fixed set of windows from it.</p>` },
  provenance: { t: "Prepared from", b: html`
    <p>How the tokenized set was produced: from which raw source and by which step (<code>prepare</code> for web/code
      text, <code>sft</code> for chat sets, <code>sft_to_pretrain</code> for conversations re-laid as pretraining text,
      <code>rl.synth</code> for templated traces, <code>synth_retrieval</code> for retrieval documents).</p>` },
  inspector_views: { t: "raw → prepared → row", b: html`
    <p>Follow data through the pipeline:</p>
    <ul><li><b>raw</b>: a row as downloaded. <i>trace</i> re-runs the preparation filters (language, character ratios,
        length) and reports whether it was kept and which split it landed in.</li>
      <li><b>prepared</b>: the stored document with its <code>&lt;|bos|&gt;</code>/<code>&lt;|eos|&gt;</code>, or the stored SFT
        example with its loss mask.</li>
      <li><b>row</b>: exactly what the model trains on: <code>seq_len</code> + 1 consecutive tokens (inputs, and targets
        shifted by one). A row usually starts mid-document; red marks document boundaries.</li></ul>` },
  show_as: { t: "text / tokens / ids", b: html`
    <p><b>text</b>: decoded. <b>tokens</b>: one chip per BPE token (· is a space, ↵ a newline), which shows how the text is
      split. <b>ids</b>: the integers the embedding layer actually receives. Where there is a loss mask, green tokens are
      trained on and grey ones are only read.</p>` },
  extra_val: { t: "extra_val_mixture (drift set)", b: html`
    <p>A second validation set, taken from the <i>pretraining</i> sources' validation splits and evaluated next to the
      stage's own validation loss. Nothing is trained on it. It is the "pretrain val" line on the run page: if it rises,
      the stage is costing general language modelling.</p>` },
  rl_recipe: { t: "RL: prompts and reward", b: html`
    <p>RL has no token mixture. Each step draws prompts from task generators, samples several answers per prompt from the
      current model, scores them with a program, and pushes up the answers that beat their group's average. The rows
      below are the knobs of that loop.</p>` },
  rl_tasks: { t: "Tasks", b: html`
    <p>Task families (arithmetic, algebra, word problems, GSM8K, Python-tool families, writing constraints), generated
      deterministically from the seed, with the share of prompts drawn from each. A task teaches only where the model
      sometimes succeeds: if every rollout of a prompt scores the same, it contributes nothing.</p>` },
  rl_prompts: { t: "Train and held-out prompts", b: html`
    <p>Prompts are split by a hash of their text, 10% held out. The policy never samples a held-out prompt, so held-out
      accuracy measures generalization to new problems rather than memory of old ones.</p>` },
  rl_rollouts: { t: "Rollouts", b: html`
    <p>Each step takes P prompts and samples G completions of each (a group) at the given temperature and top-p, up to
      <code>max_new_tokens</code>. The group is the baseline: an answer's advantage is its reward minus the group's mean.
      A bigger G gives a better baseline at the cost of more sampling.</p>` },
  rl_format: { t: "Format", b: html`
    <p>Whether every answer must open with a think span, and whether the Python tool is available (and how many calls).
      Tool results are written by the environment, so they are never policy targets and never enter the KL term.</p>` },
  rl_reward_scheme: { t: "Reward scheme", b: html`
    <ul><li><b>binary</b>: 1 if the verifier accepts the answer, else 0.</li>
      <li><b>tool</b>: 1 for a correct answer produced by a Python call, 0.5 for a correct answer computed in the head.
        <b>tool_strict</b> pays 0.25 for the latter.</li>
      <li><b>fraction</b>: partial credit, the share of writing constraints satisfied.</li>
      <li><b>plain</b>: 1 for an ordinary chat answer with no tool call; the anchor against tool-and-template habits.</li>
      <li>A malformed answer scores 0 under every scheme. Families can override the default.</li></ul>
    <p>The scheme is the whole specification of "good", and the policy optimizes it literally: paying for tool use made
      one policy reach for the sandbox on questions that needed none.</p>` },
  rl_objective: { t: "Objective (GRPO)", b: html`
    <p>Advantage = reward − the group's mean (optionally ÷ its std): no value network, the group is the baseline. Each
      token's update is weighted by the ratio ρ of new to old probability, clipped to [1−ε, 1+ε] so one step cannot move
      far. <code>kl_coef</code> × KL to the frozen starting model pulls it back toward where it began. <code>lr</code> is
      the optimizer step.</p>` },
  rl_guards: { t: "Collapse guards", b: html`
    <p><code>entropy_stop</code> and <code>kl_stop</code> end the run when the mean over the last <code>guard_window</code>
      steps passes the limit. They are windowed because a step's statistics come from a few dozen samples: a single-step
      entropy spike once stopped a healthy run whose KL was 0.0003.</p>
    <p>A real collapse is sustained: entropy climbing and staying up, KL rising, completions turning into garbage tokens.</p>` },
  rl_prompt_sample: { t: "Prompt sample", b: html`
    <p>The exact prompt the rollouts start from, <code>&lt;|bos|&gt;&lt;|user|&gt;</code>question + answer-format
      instruction<code>&lt;|end|&gt;&lt;|assistant|&gt;</code> (and <code>&lt;|think|&gt;</code> when required), and the
      gold answer the verifier compares against. Every gold is produced with the task, so tasks are correct by construction.</p>` },
  rl_rollouts_view: { t: "Stored rollouts", b: html`
    <p>Every completion the policy sampled at a step, with the reward it earned. The completion tokens are what the
      gradient acts on; the prompt and any tool results are not.</p>
    <p>Read these to catch reward hacking: high reward with nonsense reasoning, a templated answer that happens to parse,
      a tool call that just prints the answer.</p>` },
  parsed: { t: "Parsed answer", b: html`
    <p>What the verifier extracted: the last <code>#### …</code> line (a number for math, the whole span for exact answers;
      for writing tasks the answer is the text itself). <b>malformed</b>: no think span where required, no answer line, or
      a tool call after the think span; it scores 0.</p>` },
  chain: { t: "Data seen by these weights", b: html`
    <p>Walks <code>init_from</code> back to random init and adds up, per source, what each stage trained on. The totals are
      the true exposure of the final weights, e.g. how many epochs of a source they have seen across all stages.</p>
    <p><b>actual</b> rows use the loader's own counters; <b>expected</b> rows (older runs without counters) use tokens ×
      weight. RL stages train on their own samples, so they have no sources.</p>` },
  source_kind: { t: "Kind", b: html`
    <p>What a source is (prose, math, code, chat, math_cot, math_qa, or derived from another set). It decides the
      preparation filters: English-only where a language column exists, and different character-ratio thresholds for
      prose and code.</p>` },
  tokenizer_tag: { t: "Tokenizer tag", b: html`
    <p>Which tokenizer produced these ids. Shards are only meaningful to a model trained with the same tokenizer (its
      sha256 is recorded in every checkpoint), which is why the tokenizer is frozen once a model has been trained on it.</p>` },
  loss_targets: { t: "Loss targets", b: html`
    <p>The share of tokens the model is trained to predict: assistant content and its <code>&lt;|end|&gt;</code>. Prompts,
      system text, tool results, declared functions and <code>&lt;|eos|&gt;</code> are only read. A low share means long
      prompts and short answers: little learning signal per token of compute.</p>` },
  max_len: { t: "Max length", b: html`
    <p>Examples longer than this are <b>dropped, not truncated</b>. A truncated answer would teach the model to stop
      mid-thought.</p>` },
  think_span: { t: "Think span", b: html`
    <p>"mandatory": every assistant turn is <code>&lt;|think|&gt;</code>reasoning<code>&lt;|/think|&gt;</code>answer. Python
      calls live inside the span. The prompt ends with <code>&lt;|think|&gt;</code>, so the model always reasons before it answers.</p>` },
  dropped: { t: "Dropped / too long", b: html`
    <p>Rows lost in conversion: <b>dropped</b> had no usable answer (e.g. a tool set's row without any computation to turn
      into a call); <b>too long</b> exceeded the max length.</p>` },
  prepared_artifacts: { t: "Prepared artifacts", b: html`
    <p>What this source became, per tokenizer tag: shards of uint16 token ids with a document index (and a loss mask for
      SFT sets), a train and a validation split, and the step that made them.</p>` },
  browse_views: { t: "doc / window / stats", b: html`
    <p><b>doc</b>: one stored document. <b>window</b>: a contiguous slice of the token stream at a chosen length, which is
      what a training row looks like; it ignores document boundaries (red). <b>stats</b>: the length distribution of this
      shard.</p>` },
  doc_lengths: { t: "Document length", b: html`
    <p>Percentiles of tokens per document, and how many documents are long. Long-context training needs text where the
      beginning actually matters at the end. Rows packed from short documents rarely contain such dependencies, which is
      why retrieval fails first at the edge of the trained window and why long-document sources exist.</p>` },
  doc_hist: { t: "Documents vs tokens by length", b: html`
    <p>The first histogram counts documents per length bin; the second counts the tokens in them. A few long documents can
      hold a large share of the tokens. The token view is the one that matters for training: the loader reads a token
      stream, not documents.</p>` },

  // ------------------------------------------------------------------ tokenizer
  tokenizer: { t: "The tokenizer", b: html`
    <p>Byte-level BPE trained on a sample of the pretraining mix: 32,704 learned pieces plus 64 reserved special tokens.
      Pre-tokenization follows GPT-4's pattern, except that every digit is its own token (easier arithmetic for a small model).</p>
    <p>Special tokens (<code>&lt;|user|&gt;</code>, <code>&lt;|think|&gt;</code>, …) can never come from raw text; only the
      chat formatter inserts them, which is why the sample's literal "&lt;|user|&gt;" splits into ordinary pieces. The
      tokenizer is frozen, and its sha256 is recorded with every checkpoint.</p>` },
  tok_modes: { t: "raw / document / chat", b: html`
    <p><b>raw</b>: the text as-is. <b>document</b>: wrapped in <code>&lt;|bos|&gt;</code> … <code>&lt;|eos|&gt;</code>, as in
      pretraining shards. <b>chat</b>: messages through the chat formatter, with the loss mask: green tokens are trained on
      (assistant content and <code>&lt;|end|&gt;</code>), grey ones are only read.</p>` },
  chars_per_token: { t: "Compression (characters per token)", b: html`
    <p>How much text one token carries. Higher is better: more text fits in the context and each character costs less
      compute. Common English words are one token with their leading space; code, rare words and numbers (one token per
      digit) compress worse.</p>` },
  vocab: { t: "Vocabulary", b: html`
    <p>The learned pieces, searchable. Ids start with the 256 byte values, then follow merge order, so lower ids are the
      more frequent merges. "·" marks a leading space: most word pieces carry the space before them.</p>` },

  // ------------------------------------------------------------------ inference
  worker: { t: "Inference worker", b: html`
    <p>Models run in a separate process so the portal server stays free of torch and a crash cannot take it down. While a
      training run is live, loads default to the CPU so the portal does not take VRAM from it ("force cuda" overrides).
      "release GPU" stops the worker and frees its memory.</p>` },
  slots: { t: "Checkpoint slots", b: html`
    <p>Two checkpoints loaded side by side (A and B) to compare on the same prompt. The badge is the stage it came from:
      <b>base</b> has never seen chat tokens and should be used in completion mode; <b>sft</b> chats; <b>reasoning</b> and
      <b>rl</b> open every answer with a think span. For RL runs prefer <code>best.pt</code> over training snapshots.</p>` },
  sampling: { t: "Sampling", b: html`
    <ul><li><b>temperature</b> scales the logits: 0 is greedy (always the most likely token, deterministic), 1 samples the
        model's own distribution, higher flattens it.</li>
      <li><b>top-p</b> keeps the smallest set of tokens whose probabilities add up to p; <b>top-k</b> keeps the k most likely (0 = off).</li>
      <li><b>seed</b> makes a sampled run reproducible.</li></ul>
    <p>Small models loop under greedy decoding and derail at high temperature. Reasoning and RL checkpoints default to
      greedy, because at 0.8 they often fail to close the turn. The trainer, RL and evals use the same sampler.</p>` },
  force_think: { t: "Force <|think|>", b: html`
    <p>Appends <code>&lt;|think|&gt;</code> after <code>&lt;|assistant|&gt;</code>, so the model must reason before it
      answers, as reasoning and RL checkpoints were trained to. Leave it off for an instruct checkpoint.</p>` },
  python_tool: { t: "Python tool", b: html`
    <p>Inside the think span the model can write <code>&lt;|python_call|&gt;</code>code<code>&lt;|/python_call|&gt;</code>.
      Generation pauses, the code runs in a sandboxed interpreter for a Python subset (no imports, no I/O, never
      exec/eval), and the output is inserted as a <code>&lt;|python_result|&gt;</code> span. Inserted tokens are blue and
      have no log-prob, because the model did not sample them.</p>
    <p>One session per conversation, so variables persist across calls and turns. "new conversation" resets it.</p>` },
  declared_functions: { t: "Declared functions", b: html`
    <p>Capabilities the model cannot write for itself (a price list, a lookup), declared before the first turn as masked
      <code>&lt;|python_def|&gt;</code> blocks: a body-less <code>def</code> line and a plain-language comment. The model
      calls them through the ordinary Python tool. Use this to see how the model reads and reacts to a declaration.</p>` },
  score: { t: "Teacher-forced scoring", b: html`
    <p>Instead of generating, feed a given text and read the probability the model assigned to each actual next token,
      exactly what the training loss measures. <b>mean logprob</b> averages ln p over the loss-target tokens (in chat, the
      assistant's); <b>perplexity</b> = e<sup>−mean logprob</sup>.</p>
    <p>Use it to compare checkpoints on a text you care about: lower perplexity means the text is less surprising to the model.</p>` },
  token_view: { t: "Reading the output", b: html`
    <p>Each generated token carries its <b>log-prob</b> (ln of the probability the model gave it), its <b>rank</b> among all
      tokens, and the top-k alternatives (hover). In the tokens view colour runs from red (surprised) to green (confident).</p>
    <p><b>mean entropy</b> (nats) is the model's average uncertainty over the next token: high means many plausible
      continuations. A run of red tokens is usually where sampling took the model somewhere it did not expect.</p>` },

  // ------------------------------------------------------------------ architecture
  arch_shape: { t: "Model shape", b: html`
    <ul><li><b>params</b>: all weights; <b>non-embedding</b> excludes the token embedding and sets the compute per token.</li>
      <li><b>layers</b>: identical Transformer blocks. <b>d_model</b>: width of the residual stream.</li>
      <li><b>q/kv heads × head_dim</b>: attention heads; fewer key/value heads than query heads is GQA.</li>
      <li><b>d_ff</b>: hidden width of the SwiGLU MLP. <b>vocab</b>: token ids.</li>
      <li><b>max ctx</b>: size of the RoPE table (configured context, not proven effective context). <b>rope θ</b>: the RoPE base.</li>
      <li><b>qk-norm</b>: RMSNorm on queries and keys, which keeps attention logits bounded at high learning rates.
        <b>tied</b>: the output layer reuses the embedding matrix.</li></ul>` },
  arch_bt: { t: "B and T", b: html`
    <p><b>B</b>: sequences per micro-batch. <b>T</b>: tokens per sequence. They change the activation shapes, activation
      and logit memory, and the attention FLOPs (which grow with T). They do not change the parameters.</p>` },
  grad_ckpt: { t: "Gradient checkpointing", b: html`
    <p>Instead of keeping every block's activations for the backward pass, keep only each block's input and recompute the
      rest during backward. Activation memory drops sharply for about one extra forward pass (~1/3 more compute). Used for
      16K+ rows.</p>` },
  loss_chunk: { t: "Loss chunk", b: html`
    <p>The logits for a batch are [B·T, 32K] floats: at long T they can be the largest tensor in training. Chunking computes
      them and their loss a slice of tokens at a time, so the full tensor never exists. The result is identical.</p>` },
  module_graph: { t: "The module graph", b: html`
    <p>A decoder-only Transformer: token ids → embedding → N identical blocks → final RMSNorm → LM head (the embedding matrix
      again) → logits over the vocabulary. Each block is pre-norm and residual:</p>
    <p><code>x = x + Attn(RMSNorm(x))</code><br /><code>x = x + SwiGLU(RMSNorm(x))</code></p>
    <p>The residual stream [B, T, d_model] runs through the whole model; each block reads it and adds its result back.
      Shapes, parameters and FLOPs here come from the real module tree.</p>` },
  gqa: { t: "Grouped-query attention", b: html`
    <p>Every query head keeps its own projection, but key and value heads are shared by a group of query heads. That cuts
      the KV cache and the k/v projections by the group size, with little quality loss. It is standard in Llama-3-style
      models, and the fused attention kernel handles the sharing without copying heads.</p>` },
  param_family: { t: "Parameters by family", b: html`
    <p>Where the weights live. The MLP (gate, up and down projections, ~3 × d_model × d_ff per layer) holds most of them,
      attention most of the rest, and norms almost nothing. The embedding is counted once because the output layer reuses
      it; at this size it is still a large share, which is why an untied head was not worth it.</p>` },
  kv_cache: { t: "KV cache", b: html`
    <p>During generation the keys and values of earlier tokens are kept, so each new token attends to them without
      recomputing them. Size = 2 × layers × KV heads × head_dim × tokens × 2 bytes (bf16). It grows linearly with context,
      and GQA divides it by the group size.</p>` },
  flops: { t: "Training FLOPs per token", b: html`
    <p>A matrix multiply costs 2 FLOPs per weight per token forward and about 4 backward, hence the "6 × parameters" rule
      for the linear layers and the LM head. Attention scores add 12 × layers × width × T, the only term that grows with
      context length. Total training compute ≈ FLOPs per token × tokens.</p>
    <p>The PaLM convention counts the dense score matrix; causal kernels skip the masked half, hence the second figure.</p>
    <p class="see">MFU divides achieved FLOPs/s by the GPU's peak.</p>` },
  memory: { t: "Training memory", b: html`
    <ul><li><b>weights</b> fp32 (4 B/param) and <b>grads</b> fp32 (4 B/param): mixed precision computes in bf16 but keeps fp32 master weights.</li>
      <li><b>AdamW m, v</b>: two fp32 moments per parameter (8 B/param), the largest fixed cost.</li>
      <li><b>activations</b>: what the forward pass keeps for backward. Grows with B × T × layers; gradient checkpointing trades it for compute.</li>
      <li><b>logits</b>: B·T × vocab in fp32 plus its gradient, unless the loss is chunked.</li></ul>
    <p>This is an estimate. On this Windows GPU, anything near 16 GiB spills into host memory (60x slower), so configs aim
      for ~11 GiB; the measured table is the ground truth.</p>` },
  bench: { t: "Measured throughput", b: html`
    <p>A real benchmark of forward, backward and optimizer steps per (sequence length, microbatch).</p>
    <p><b>MFU</b> (model FLOPs utilization) = achieved training FLOPs/s ÷ the GPU's bf16 peak (104.5 TFLOPS). The causal
      column counts only attention work that is actually executed. <code>torch.compile</code> is a ~6x win here. A larger
      microbatch raises utilization until memory runs out; training configs take their microbatch from this table.</p>` },
  lr_schedule: { t: "Learning-rate schedule", b: html`
    <p><b>Warmup</b> ramps linearly from 0 to the peak, because early gradients are large and Adam's statistics are still
      settling. Then:</p>
    <ul><li><b>WSD</b> (warmup-stable-decay): flat at the peak, then a linear decay over the last fraction. A stable phase
        can be continued or branched, with the decay done as a separate phase.</li>
      <li><b>cosine</b>: a smooth fall to <code>min_lr</code>. <b>constant</b>: no decay.</li></ul>
    <p>Indexed by tokens, not steps, so changing the batch size does not silently stretch the schedule.</p>` },
  batch: { t: "Batch and gradient accumulation", b: html`
    <p>A micro-batch is what fits in memory; <b>tokens per update</b> is what the optimizer sees. Gradients from several
      micro-batches are added up before one AdamW step. Each micro-batch returns (loss sum, token count) and the trainer
      divides by the update's total count, so accumulation gives exactly the same update as one big batch.</p>` },
  rope: { t: "RoPE (rotary position embedding)", b: html`
    <p>Positions are encoded by rotating each pair of query/key dimensions by position × θ<sub>i</sub>, with
      θ<sub>i</sub> = base<sup>−2i/D</sup>. The attention score then depends only on the distance between two tokens.</p>
    <p>Early pairs rotate fast and resolve local order; late pairs rotate slowly and carry long-range position. A larger
      base (θ) slows them all, which suits longer contexts.</p>` },
  rope_wavelength: { t: "RoPE wavelengths", b: html`
    <p>Tokens per full rotation for each frequency pair (log scale). Pairs above the red line never complete a turn within
      the context, so they act as slow position counters. The model has only seen their angles up to the trained length,
      which is why a longer context needs RoPE scaling plus training rather than just a bigger table.</p>` },
  rope_scaling: { t: "RoPE scaling for context extension", b: html`
    <ul><li><b>linear</b>: divide positions by the factor. Every frequency is compressed, which blurs local order.</li>
      <li><b>NTK</b>: raise the base. High frequencies stay intact and low ones stretch.</li>
      <li><b>YaRN</b>: per frequency, keep the fast pairs, interpolate the slow ones and blend in between, plus an
        attention temperature (mscale). Used for this project's context extension.</li></ul>` },
  cadence: { t: "Cadence", b: html`
    <p>How often a run evaluates, samples, snapshots and saves a resumable checkpoint. Often enough to see a problem early
      and lose little on a crash; rarely enough that evaluation does not eat the throughput.</p>` },
};
