"""External comparison models: similarly sized open-weight LMs, run locally through our own eval harness.

    python -m slm.eval.external list                           # the registry, and which weights are on disk
    python -m slm.eval.external download --all                 # one-time acquisition (the only networked step)
    python -m slm.eval.external model-json smollm2-360m-instruct   # writes runs/ext_<name>/model.json
    bash scripts/measure_external.sh smollm2-360m-instruct     # every applicable eval, same limits as ours

Each eval takes `--external <short-name>` and runs the same items, limits, prompts, verifier and decoding settings
through `HfChatModel`, writing the same JSON shape with `"checkpoint": "external:<name>"` and a `"model"` block.
Results go to `runs/ext_<name>/` under the file names our runs use; the Evals tab shows them as a separate group.

Rules (docs/design.md §8):
- **Purely local at run time.** Importing this module forces the Hugging Face offline switches on (and flips the
  already-imported libraries' cached copies of them), and weights load from `<data root>/models/<name>/` with
  `local_files_only`. The one networked step is `download`, which runs `snapshot_download` in a child process
  whose environment has the switches removed, so nothing in this process ever goes online.
- **Each model uses its own tokenizer and chat template.** Base models get completion prompts only, and are n/a
  on chat-only evals: a base model is never given a chat template, even when its tokenizer ships one.
- **Our decoding, not theirs.** Generation is our own loop around the HF forward pass with `slm.eval.sampling.
  sample_next`, so temperature / top-p / top-k mean exactly what they mean for our checkpoints and no
  `generation_config.json` default (Qwen2.5-Instruct ships repetition_penalty 1.1, top_k 20, ...) leaks in.
  Greedy = temperature 0 (or top_k 1).
- Anything that depends on our tool protocol (Python calls, sandbox verification, the selector) is n/a.
"""

from __future__ import annotations

import os
import sys

# ------------------------------------------------------------------------------------------------ offline switches
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}


def ensure_offline() -> None:
    """Force offline mode for this process. The env vars cover everything imported later; huggingface_hub and
    datasets read them once at import, so a copy already imported (lm_eval pulls in datasets) is flipped too."""
    os.environ.update(OFFLINE_ENV)
    hub = sys.modules.get("huggingface_hub.constants")
    if hub is not None:
        hub.HF_HUB_OFFLINE = True
    dcfg = sys.modules.get("datasets.config")
    if dcfg is not None:
        dcfg.HF_HUB_OFFLINE = True
        dcfg.HF_DATASETS_OFFLINE = True


ensure_offline()

import json  # noqa: E402
import subprocess  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

DATA_ROOT = Path(os.environ.get("SLM_DATA_ROOT", r"C:\slm-data"))
MODELS_ROOT = DATA_ROOT / "models"
# weights, configs and tokenizer files only: no ONNX / TF / Flax exports, no duplicate pytorch_model.bin
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "LICENSE*", "README.md"]
IGNORE_PATTERNS = ["*/*"]  # sub-folders (onnx/, runs/, ...) never hold what we load


@dataclass(frozen=True)
class ExternalModel:
    name: str
    hf_id: str
    params: int                 # sum of model.parameters() (tied embeddings counted once; gpt2's causal-mask buffers excluded)
    license: str
    is_chat: bool               # instruction-tuned with a chat template; base models get completion prompts only
    max_positions: int          # config max_position_embeddings / n_positions: needle lengths above it are n/a
    train_tokens: int | None    # pretraining tokens as published by the authors (None = not published)
    notes: str = ""
    local_dir: Path = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.local_dir is None:
            object.__setattr__(self, "local_dir", MODELS_ROOT / self.name)

    def block(self) -> dict:
        """The `"model"` block every external result file carries (and runs/ext_<name>/model.json)."""
        return {"name": self.name, "hf_id": self.hf_id, "local_dir": str(self.local_dir), "params": self.params, "license": self.license,
                "is_chat": self.is_chat, "max_positions": self.max_positions, "train_tokens": self.train_tokens, "notes": self.notes}

    def available(self) -> bool:
        return (self.local_dir / "config.json").is_file() and any(self.local_dir.glob("*.safetensors"))


_T = 10**12
EXTERNAL_MODELS: dict[str, ExternalModel] = {m.name: m for m in [
    ExternalModel("smollm2-135m", "HuggingFaceTB/SmolLM2-135M", 134_515_008, "Apache-2.0", False, 8192, 2 * _T,
                  "SmolLM2 base; 2T tokens (FineWeb-Edu, DCLM, The Stack, FineMath, ...), context 8K"),
    ExternalModel("smollm2-135m-instruct", "HuggingFaceTB/SmolLM2-135M-Instruct", 134_515_008, "Apache-2.0", True, 8192, 2 * _T,
                  "SmolLM2-135M + SFT (SmolTalk) + DPO; ChatML template with a default system prompt"),
    ExternalModel("smollm2-360m", "HuggingFaceTB/SmolLM2-360M", 361_821_120, "Apache-2.0", False, 8192, 4 * _T,
                  "SmolLM2 base; 4T tokens, context 8K"),
    ExternalModel("smollm2-360m-instruct", "HuggingFaceTB/SmolLM2-360M-Instruct", 361_821_120, "Apache-2.0", True, 8192, 4 * _T,
                  "SmolLM2-360M + SFT (SmolTalk) + DPO; ChatML template with a default system prompt"),
    ExternalModel("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B", 494_032_768, "Apache-2.0", False, 32768, 18 * _T,
                  "Qwen2.5 base; up to 18T tokens (series-level figure), 151K-token vocabulary (136M of the params are embeddings), context 32K"),
    ExternalModel("qwen2.5-0.5b-instruct", "Qwen/Qwen2.5-0.5B-Instruct", 494_032_768, "Apache-2.0", True, 32768, 18 * _T,
                  "Qwen2.5-0.5B + SFT + RL; ChatML template with a default system prompt"),
    ExternalModel("gpt2-medium", "openai-community/gpt2-medium", 354_823_168, "MIT", False, 1024, None,
                  "GPT-2 (2019) medium; WebText (~40 GB of text, token count not published), context 1024, no chat model; fp32 weights run in bf16"),
]}


def get(name: str) -> ExternalModel:
    if name not in EXTERNAL_MODELS:
        raise SystemExit(f"unknown external model {name!r}; known: {', '.join(EXTERNAL_MODELS)}")
    return EXTERNAL_MODELS[name]


def run_dir(name: str, runs_root: str | Path = "runs") -> Path:
    return Path(runs_root) / f"ext_{name}"


def checkpoint_label(name: str) -> str:
    return f"external:{name}"


def write_model_json(name: str, runs_root: str | Path = "runs") -> Path:
    p = run_dir(name, runs_root) / "model.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(get(name).block(), indent=1), encoding="utf-8")
    return p


def chat_only(name: str, eval_name: str) -> ExternalModel:
    """The registry entry, or a clean exit for a base model on a chat-only eval (n/a, not approximated)."""
    m = get(name)
    if not m.is_chat:
        raise SystemExit(f"{eval_name}: n/a for {name} (a base model; this eval needs a chat model and we never invent a template)")
    return m


# ------------------------------------------------------------------------------------------------ the wrapper
@dataclass
class Generation:
    text: str            # decoded generated tokens, special tokens and the stop token removed, cut at a stop string
    ids: list[int]       # generated ids including the stop token when one was reached
    terminated: bool     # ended on a stop token / stop string (False = ran out of max_new_tokens or the window)
    n_tokens: int


class HfChatModel:
    """A local HF causal LM behind the small interface our evals need. Not thread-safe; one per process."""

    def __init__(self, entry: ExternalModel, model, tokenizer, device: str) -> None:
        self.entry = entry
        self.name = entry.name
        self.is_chat = entry.is_chat
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        cfg = model.config
        self.max_positions = int(getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", None) or entry.max_positions)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.eos_ids = self._eos_ids()

    def _eos_ids(self) -> set[int]:
        ids = set()
        if self.tokenizer.eos_token_id is not None:
            ids.add(int(self.tokenizer.eos_token_id))
        gc = getattr(self.model, "generation_config", None)
        e = getattr(gc, "eos_token_id", None) if gc is not None else None
        for x in (e if isinstance(e, (list, tuple)) else [e]):
            if x is not None:
                ids.add(int(x))
        if self.is_chat:  # ChatML end of turn (both chat families) and end of document (SmolLM2-Instruct's config omits it)
            vocab = self.tokenizer.get_vocab()
            ids.update(vocab[t] for t in ("<|im_end|>", "<|endoftext|>") if t in vocab)
        return ids

    # -------------------------------------------------------------------- encoding
    def encode_text(self, text: str) -> list[int]:
        """A completion prompt, encoded the way the model's tokenizer does by default (`add_special_tokens=True`:
        none of the registered tokenizers adds a BOS, which is also what lm-eval's hf backend does)."""
        return list(self.tokenizer(text, add_special_tokens=True)["input_ids"])

    def encode_plain(self, text: str) -> list[int]:
        """Raw text with no special tokens (haystack filler, needle, question)."""
        return list(self.tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])

    def prefix_ids(self) -> list[int]:
        """What the tokenizer puts in front of a document by default (empty for every registered model)."""
        return self.encode_text("")

    def render_chat(self, messages: list[dict]) -> str:
        if not self.is_chat:
            raise ValueError(f"{self.name} is a base model: no chat template is applied to it")
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def encode_chat(self, messages: list[dict]) -> list[int]:
        return list(self.tokenizer(self.render_chat(messages), add_special_tokens=False)["input_ids"])

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    # -------------------------------------------------------------------- generation
    def generate_ids(self, prompts: list[list[int]], max_new_tokens: int, temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0,
                     seed: int = 0, stop: list[str] | None = None, stop_ids: set[int] | None = None, batch_size: int = 32) -> list[Generation]:
        """Generate for every prompt (left-padded batches, sorted by length). One torch.Generator seeded once per call,
        consumed batch by batch, so a call is reproducible for a fixed prompt list and batch size."""
        import torch

        from slm.utils.sdpa import sdpa_context

        stop_ids = self.eos_ids if stop_ids is None else stop_ids
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        out: list[Generation | None] = [None] * len(prompts)
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        with torch.no_grad(), sdpa_context("decode" if self.device != "cpu" else None):
            for b in range(0, len(order), max(1, batch_size)):
                idx = order[b : b + batch_size]
                for i, g in zip(idx, self._generate_batch([prompts[i] for i in idx], max_new_tokens, temperature, top_p, top_k, gen, stop, stop_ids)):
                    out[i] = g
                if self.device != "cpu":
                    torch.cuda.empty_cache()  # a KV cache segment the allocator keeps is reserved VRAM (CLAUDE.md, WDDM)
        return out  # type: ignore[return-value]

    def _generate_batch(self, prompts, max_new_tokens, temperature, top_p, top_k, gen, stop, stop_ids) -> list[Generation]:
        import torch

        from slm.eval.sampling import sample_next

        dev = self.device
        B, P = len(prompts), max(len(p) for p in prompts)
        budget = max(0, min(int(max_new_tokens), self.max_positions - P))
        if budget == 0 or min(len(p) for p in prompts) == 0:
            return [Generation("", [], False, 0) for _ in prompts]
        ids = torch.full((B, P), self.pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros((B, P), dtype=torch.long, device=dev)
        for i, p in enumerate(prompts):
            ids[i, P - len(p) :] = torch.tensor(p, dtype=torch.long, device=dev)
            mask[i, P - len(p) :] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        o = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True, logits_to_keep=1)
        past, logits = o.past_key_values, o.logits[:, -1, :].float()
        nxt_pos = pos[:, -1:] + 1
        comps: list[list[int]] = [[] for _ in range(B)]
        done = [False] * B
        term = [False] * B
        for step in range(budget):
            nxt = sample_next(logits, temperature, top_p, top_k, gen)
            for i, t in enumerate(nxt.tolist()):
                if done[i]:
                    continue
                comps[i].append(t)
                if t in stop_ids:
                    done[i] = term[i] = True
                elif stop and any(s in self.decode(comps[i]) for s in stop):
                    done[i] = term[i] = True
            if all(done) or step == budget - 1:
                break
            mask = torch.cat([mask, torch.ones((B, 1), dtype=mask.dtype, device=dev)], dim=1)
            o = self.model(input_ids=nxt[:, None], attention_mask=mask, position_ids=nxt_pos, past_key_values=past, use_cache=True)
            past, logits = o.past_key_values, o.logits[:, -1, :].float()
            nxt_pos = nxt_pos + 1
        res = []
        for c, t in zip(comps, term):
            body = c[:-1] if c and c[-1] in stop_ids else c
            text = self.decode(body)
            if stop:
                for s in stop:
                    if s in text:
                        text = text.split(s)[0]
            res.append(Generation(text, c, t, len(c)))
        return res

    def stream_ids(self, prompt: list[int], max_new_tokens: int, temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0,
                   seed: int = 0, stop_ids: set[int] | None = None, logprobs_topk: int = 0):
        """One prompt, one token at a time (the portal's streaming path). Yields per sampled token {id, logprob, rank,
        entropy, topk: [(id, logprob)]} under the model's own distribution (before temperature / top-p). Same sampler,
        seeding, budget and stop rule as `generate_ids` with one prompt, so the streamed ids equal
        `generate_ids([prompt], ...)[0].ids`; the stop token is the last item yielded. Close the generator to stop early."""
        import torch

        from slm.eval.sampling import sample_next

        stop_ids = self.eos_ids if stop_ids is None else stop_ids
        budget = max(0, min(int(max_new_tokens), self.max_positions - len(prompt)))
        if budget == 0 or not prompt:
            return
        dev = self.device
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed))
        try:
            with torch.no_grad():
                ids = torch.tensor([prompt], dtype=torch.long, device=dev)
                mask = torch.ones_like(ids)
                pos = torch.arange(len(prompt), device=dev)[None]
                o = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True, logits_to_keep=1)
                past, logits = o.past_key_values, o.logits[:, -1, :].float()
                nxt_pos = pos[:, -1:] + 1
            for step in range(budget):
                with torch.no_grad():
                    nxt = sample_next(logits, temperature, top_p, top_k, gen)
                    t = int(nxt[0])
                    logp = torch.log_softmax(logits[0], dim=-1)
                    ev = {"id": t, "logprob": float(logp[t]), "rank": int((logp > logp[t]).sum()),
                          "entropy": float(-(logp.exp() * logp).nan_to_num().sum()), "topk": []}
                    if logprobs_topk > 0:
                        top = torch.topk(logp, min(int(logprobs_topk), logp.shape[-1]))
                        ev["topk"] = [(int(i), float(v)) for v, i in zip(top.values, top.indices)]
                yield ev
                if t in stop_ids or step == budget - 1:
                    return
                with torch.no_grad():
                    mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype, device=dev)], dim=1)
                    o = self.model(input_ids=nxt[:, None], attention_mask=mask, position_ids=nxt_pos, past_key_values=past, use_cache=True)
                    past, logits = o.past_key_values, o.logits[:, -1, :].float()
                    nxt_pos = nxt_pos + 1
        finally:
            if dev != "cpu":
                torch.cuda.empty_cache()  # the KV cache's segments stay reserved otherwise (CLAUDE.md, WDDM)

    def generate_text(self, prompt_text: str, max_new_tokens: int = 128, temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0,
                      seed: int = 0, stop: list[str] | None = None) -> str:
        """Completion prompt -> continuation (stops at the tokenizer's EOS or a stop string)."""
        return self.generate_ids([self.encode_text(prompt_text)], max_new_tokens, temperature, top_p, top_k, seed, stop)[0].text

    def batch_generate_text(self, prompt_texts: list[str], max_new_tokens: int = 128, temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0,
                            seed: int = 0, stop: list[str] | None = None, batch_size: int = 32) -> list[Generation]:
        return self.generate_ids([self.encode_text(t) for t in prompt_texts], max_new_tokens, temperature, top_p, top_k, seed, stop,
                                 batch_size=batch_size)

    def generate_chat(self, messages: list[dict], max_new_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0,
                      seed: int = 0, stop: list[str] | None = None) -> str:
        """Messages -> the assistant's reply, through the model's own chat template (add_generation_prompt=True)."""
        return self.batch_generate_chat([messages], max_new_tokens, temperature, top_p, top_k, seed, stop)[0].text

    def batch_generate_chat(self, list_of_messages: list[list[dict]], max_new_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0,
                            top_k: int = 0, seed: int = 0, stop: list[str] | None = None, batch_size: int = 32) -> list[Generation]:
        """Many conversations (or k copies of one, for parallel sampling) in left-padded batches."""
        return self.generate_ids([self.encode_chat(m) for m in list_of_messages], max_new_tokens, temperature, top_p, top_k, seed, stop,
                                 batch_size=batch_size)


def load_external(name: str, device: str = "cuda", dtype=None) -> HfChatModel:
    """Load a registered model from its local directory (never the network). dtype defaults to bfloat16."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ensure_offline()
    m = get(name)
    if not m.available():
        raise SystemExit(f"{name}: no weights under {m.local_dir}; run `python -m slm.eval.external download {name}` once (docs/setup.md)")
    dtype = torch.bfloat16 if dtype is None else dtype
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[external] CUDA not available: {name} runs on the CPU", flush=True)
        device = "cpu"
    tok = AutoTokenizer.from_pretrained(str(m.local_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(m.local_dir), local_files_only=True, dtype=dtype, attn_implementation="sdpa")
    model = model.to(device).eval()
    return HfChatModel(m, model, tok, device)


def result_header(name: str) -> dict:
    """The fields every external result file starts with."""
    return {"checkpoint": checkpoint_label(name), "model": get(name).block()}


# ------------------------------------------------------------------------------------------------ swarm (scripts/swarm_eval.py)
def swarm_external(name: str, gsm8k: int, svamp: int, k: int, temperature: float, max_new: int, seed: int, out: str | None,
                   top_p: float = 0.95) -> dict:
    """The sampling half of the swarm eval for an external chat model: greedy, majority vote over k samples and the
    pass@k oracle, from the same prompts (task + SUFFIX), the same k / temperature / top-p / budget and the same
    grouping (slm.swarm.collapse). No sandbox and no trained selector, so verified_majority, oracle_verified,
    selector and in_prompt are null (n/a), not approximated."""
    import statistics
    import time

    from slm.data.answers import SUFFIX
    from slm.eval.reasoning import gsm8k_tasks, svamp_tasks
    from slm.rl.rewards import parse_final_span, verify_answer
    from slm.swarm import Candidate, answer_key, collapse, majority

    chat_only(name, "swarm_eval")
    model = load_external(name)
    sets = {}
    if gsm8k:
        sets["gsm8k"] = gsm8k_tasks(gsm8k)
    if svamp:
        sets["svamp"] = svamp_tasks(svamp)

    def ok(answer: str | None, gold: str) -> bool:
        return answer is not None and bool(verify_answer(f"#### {answer}", gold, "auto").correct)

    na = ("verified_majority", "selector", "tournament", "oracle_verified", "in_prompt", "selector_called_tool")
    results = {}
    for set_name, tasks in sets.items():
        t0 = time.time()
        rows = []
        for i, t in enumerate(tasks):
            msgs = [{"role": "user", "content": t.prompt + SUFFIX}]
            t1 = time.time()
            greedy = parse_final_span(model.generate_chat(msgs, max_new, temperature=0.0))
            gens = model.batch_generate_chat([msgs] * k, max_new, temperature=temperature, top_p=top_p, top_k=0, seed=seed * 100003 + i, batch_size=k)
            cands = []
            for j, g in enumerate(gens):
                parsed = parse_final_span(g.text)
                cands.append(Candidate(idx=j, think=None, answer=g.text, parsed=parsed, key=answer_key(parsed), terminated=g.terminated,
                                       n_calls=0, n_errors=0, calls=[], from_tool=False, n_tokens=g.n_tokens))
            groups = collapse(cands)
            maj = majority(groups)
            rows.append({"id": t.id, "gold": t.answer, "seconds": round(time.time() - t1, 2), "greedy": ok(greedy, t.answer), "majority": ok(maj, t.answer),
                         **{key: None for key in na}, "oracle": any(ok(c.parsed, t.answer) for c in cands),
                         "n_groups": len(groups), "n_verified": None, "selector_final": None, "majority_answer": maj})
            if (i + 1) % 10 == 0:
                m = lambda key: statistics.fmean(r[key] for r in rows)  # noqa: E731
                print(f"  {set_name} {i + 1}/{len(tasks)}: greedy {m('greedy'):.2f} majority {m('majority'):.2f} | oracle {m('oracle'):.2f} [{time.time() - t0:.0f}s]", flush=True)
        summary = {key: round(statistics.fmean(r[key] for r in rows), 3) for key in ("greedy", "majority", "oracle")}
        summary.update({key: None for key in na})
        summary.update({"n": len(rows), "k": k, "mean_groups": round(statistics.fmean(r["n_groups"] for r in rows), 1), "mean_verified": None,
                        "seconds": round(time.time() - t0, 1)})
        results[set_name] = {"summary": summary, "problems": rows}
        print(f"\n{set_name} (n={len(rows)}, k={k}): greedy {summary['greedy']}  majority {summary['majority']}  oracle {summary['oracle']}  "
              f"(verified_majority / selector / oracle_verified / in_prompt: n/a for an external model)", flush=True)
    res = {**result_header(name), "k": k, "temperature": temperature, "top_p": top_p, "tools": False, "results": results}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"wrote {out}")
    return res


# ------------------------------------------------------------------------------------------------ acquisition (the only networked step)
_DOWNLOAD_CHILD = """
import json, sys
from huggingface_hub import snapshot_download
a = json.loads(sys.argv[1])
p = snapshot_download(repo_id=a["repo_id"], local_dir=a["local_dir"], allow_patterns=a["allow"], ignore_patterns=a["ignore"])
print(p)
"""


def download(name: str) -> Path:
    """snapshot_download into <data root>/models/<name>/, in a child process with the offline switches removed: this
    process stays offline, and nothing else in the project ever downloads a model."""
    m = get(name)
    env = {k: v for k, v in os.environ.items() if k not in OFFLINE_ENV}
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    arg = json.dumps({"repo_id": m.hf_id, "local_dir": str(m.local_dir), "allow": ALLOW_PATTERNS, "ignore": IGNORE_PATTERNS})
    m.local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-c", _DOWNLOAD_CHILD, arg], env=env, check=True)
    return m.local_dir


def _dir_bytes(d: Path) -> int:
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file() and ".cache" not in p.parts) if d.exists() else 0


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the registry and what is on disk")
    d = sub.add_parser("download", help="one-time snapshot_download of the weights (network)")
    d.add_argument("names", nargs="*")
    d.add_argument("--all", action="store_true")
    j = sub.add_parser("model-json", help="write runs/ext_<name>/model.json (the Evals tab's registry entry)")
    j.add_argument("name")
    j.add_argument("--runs-root", default="runs")
    a = ap.parse_args()
    if a.cmd == "list":
        for m in EXTERNAL_MODELS.values():
            size = _dir_bytes(m.local_dir)
            print(f"{m.name:24s} {m.hf_id:40s} {m.params / 1e6:6.0f}M  {m.license:10s} {'chat' if m.is_chat else 'base':4s}  ctx {m.max_positions:<6d} "
                  f"{'on disk %.2f GB' % (size / 1e9) if m.available() else 'MISSING'}")
    elif a.cmd == "download":
        names = list(EXTERNAL_MODELS) if a.all else a.names
        if not names:
            raise SystemExit("name models or pass --all")
        for n in names:
            p = download(n)
            print(f"{n}: {p} ({_dir_bytes(p) / 1e9:.2f} GB)", flush=True)
    elif a.cmd == "model-json":
        print(write_model_json(a.name, a.runs_root))


if __name__ == "__main__":
    main()
