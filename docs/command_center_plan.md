# Command Center Portal — Implementation Plan

Drafted 2026-09-12 by a planning agent; reviewed by Claude. Status: proposed, build after M1 is running.
The portal is a supplement; the chat session stays the primary interface for the project.

## Summary of decisions

- FastAPI + uvicorn backend (torch-free server process); no-build single-page frontend: vendored ESM
  Preact + htm, uPlot for time series, hand-written SVG for diagrams. SSE for streaming.
- All torch work (checkpoint loading, generation, scoring, diagnostics, architecture shape hooks) runs
  in a lazily spawned worker subprocess with a GPU guard: never touch CUDA while a training run is
  live unless forced; CPU fallback; idle auto-stop; explicit "Release GPU".
- Stage plugin registry (pretrain, sft, reasoning_sft, grpo, context_ext) drives navigation, chart
  specs, hyper-parameter illustrations, and architecture "lenses" so new stages slot in.
- Architecture diagrams come from a JSON module graph derived by walking the real nn.Module tree with
  shape hooks plus per-module "expanders" for teaching-level ops (split/reshape/RoPE/SDPA/GQA groups).
  Tests reconcile the graph's params/FLOPs with Transformer.num_params() and profiling.flops_per_token.
- Illustrations call the real functions (lr_at, rope_inv_freq) so they cannot drift from training.

## Pages (priority)

1. Home: GPU tile, live runs, checkpoints, data readiness. (P0)
2. Runs: list, detail with live uPlot charts via SSE tail, samples timeline, checkpoints, events,
   config; compare runs (P1); STOP control (P2).
3. Data: sources overview + mixture/epochs-per-source (P0); tokenized shard/doc/window browser with
   token colorization and doc boundaries (P0); per-source stats from idx files (P0); raw parquet
   browser, sampled search job, exact batch replay inspector (P1).
4. Tokenizer: playground with colored chips and offsets, document/chat modes, vocab explorer. (P0)
5. Model harness: slots A/B, completion + chat modes, streaming tokens with per-token logprob and
   top-k alternatives, seed, cancel (P0); prompt scoring, A/B side-by-side with divergence marker
   (P1); diagnostics runner/viewer (P2).
6. Architecture: interactive expandable SVG of the exact model from any config, symbolic + numeric
   shapes with B/T sliders, params/FLOPs/memory tables next to measured bench/run numbers, KV-cache
   panel, LR/batch/RoPE/mixture/cadence illustrations (P0); animations (P1); stage lenses (P2).

## Layout

slm/portal/{__main__,app,settings,cache,jobs,sse}.py, api/, services/ (runs, datasets, batches,
tokenizer, hparams, worker, harness), stages/, static/ (index.html, pages/, components/, vendor/).
Plus slm/model/introspect.py, slm/eval/sampling.py, slm/data/chat.py, slm/utils/metrics.py.

## Risks

- VRAM contention with training: worker isolation + GPU guard + CPU fallback + idle stop.
- Windows file locking vs os.replace in checkpoint saves: retry loop in checkpoint.py (done),
  worker loads with mmap=False and closes, portal prefers immutable snap_*.pt for live runs.
- Large shard/parquet scanning: row-group random access, idx-only stats, sampled cancellable search,
  (path, mtime, size)-keyed disk cache.

## Milestones (focused days)

M0 skeleton (1) -> M1 runs (2) -> M2 tokenizer + tokenized data (2) -> M3 model harness (3) ->
M4 architecture (3) -> M5 data/harness P1 (2) -> M6 plugins + diagnostics (ongoing).
