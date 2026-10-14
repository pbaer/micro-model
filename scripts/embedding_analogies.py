"""Word-vector analogies (king - man + woman ~ queen) on a checkpoint's token-embedding matrix. CPU only.

    python scripts/embedding_analogies.py --checkpoint runs/m10_rl_336m/checkpoints/final.pt \
        --out runs/embeddings/m10_analogies.json

Only the embedding rows are read (the 336M config ties input and output embeddings, so this is also the unembedding).
Each category is a list of (a, b) word pairs sharing one relation; every ordered pair of pairs gives one quadruple
a:b :: c:d, answered by the nearest neighbour of v(b) - v(a) + v(c) by cosine (3CosAdd) with a, b, c excluded.
Words are tested as single tokens in the leading-space form (" king"); the bare form ("king") is a secondary run.
Pairs where either word is not a single token are dropped and listed in the JSON.

Conditions: two candidate pools (the full BPE vocab minus specials; "word-like" = alphabetic tokens of the same form)
times three embedding treatments (raw rows unit-normalised; centred = minus the vocab mean, then unit-normalised;
abtt = centred with the top principal directions removed, Mu & Viswanath 2018). The "c-only" baseline answers each
quadruple with the nearest neighbour of v(c) alone: when it matches, the analogy offset added nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from slm.data.tokenizer import SlmTokenizer  # noqa: E402

CATEGORIES: dict[str, list[tuple[str, str]]] = {
    "gender": [
        ("king", "queen"), ("man", "woman"), ("prince", "princess"), ("actor", "actress"), ("father", "mother"),
        ("brother", "sister"), ("uncle", "aunt"), ("husband", "wife"), ("boy", "girl"), ("he", "she"),
        ("his", "her"), ("son", "daughter"), ("Mr", "Mrs"), ("lord", "lady"), ("male", "female"), ("men", "women"),
        ("boys", "girls"), ("grandfather", "grandmother"), ("nephew", "niece"), ("emperor", "empress"),
        ("god", "goddess"), ("waiter", "waitress"), ("himself", "herself"), ("dad", "mom"), ("kings", "queens"),
        ("sons", "daughters"), ("brothers", "sisters"), ("duke", "duchess"), ("hero", "heroine"), ("groom", "bride"),
    ],
    "capital": [
        ("France", "Paris"), ("Japan", "Tokyo"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
        ("England", "London"), ("Russia", "Moscow"), ("China", "Beijing"), ("Egypt", "Cairo"), ("Greece", "Athens"),
        ("Canada", "Ottawa"), ("Ireland", "Dublin"), ("Austria", "Vienna"), ("Poland", "Warsaw"),
        ("Portugal", "Lisbon"), ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Kenya", "Nairobi"),
        ("Cuba", "Havana"), ("Thailand", "Bangkok"), ("Turkey", "Ankara"), ("Iran", "Tehran"), ("Iraq", "Baghdad"),
        ("Peru", "Lima"), ("Hungary", "Budapest"), ("Denmark", "Copenhagen"), ("Finland", "Helsinki"),
        ("Belgium", "Brussels"), ("Korea", "Seoul"), ("Vietnam", "Hanoi"), ("Philippines", "Manila"),
        ("Syria", "Damascus"), ("Lebanon", "Beirut"), ("Afghanistan", "Kabul"), ("Australia", "Canberra"),
        ("Scotland", "Edinburgh"), ("Netherlands", "Amsterdam"), ("Chile", "Santiago"), ("Wales", "Cardiff"),
        ("India", "Delhi"), ("Israel", "Jerusalem"), ("Switzerland", "Bern"),
    ],
    "state_capital": [
        ("Massachusetts", "Boston"), ("Texas", "Austin"), ("California", "Sacramento"), ("Georgia", "Atlanta"),
        ("Virginia", "Richmond"), ("Ohio", "Columbus"), ("Colorado", "Denver"), ("Arizona", "Phoenix"),
        ("Oregon", "Salem"), ("Wisconsin", "Madison"), ("Mississippi", "Jackson"), ("Alabama", "Montgomery"),
        ("Nebraska", "Lincoln"), ("Tennessee", "Nashville"), ("Hawaii", "Honolulu"), ("Indiana", "Indianapolis"),
    ],
    "nationality": [
        ("France", "French"), ("Germany", "German"), ("Italy", "Italian"), ("Spain", "Spanish"), ("China", "Chinese"),
        ("Japan", "Japanese"), ("Russia", "Russian"), ("England", "English"), ("Greece", "Greek"),
        ("Poland", "Polish"), ("Sweden", "Swedish"), ("Ireland", "Irish"), ("India", "Indian"),
        ("Canada", "Canadian"), ("Mexico", "Mexican"), ("Egypt", "Egyptian"), ("Brazil", "Brazilian"),
        ("Turkey", "Turkish"), ("Korea", "Korean"), ("Norway", "Norwegian"), ("Denmark", "Danish"),
        ("Netherlands", "Dutch"), ("Australia", "Australian"), ("America", "American"), ("Israel", "Israeli"),
        ("Portugal", "Portuguese"), ("Scotland", "Scottish"), ("Wales", "Welsh"), ("Austria", "Austrian"),
        ("Switzerland", "Swiss"),
    ],
    "plural": [
        ("dog", "dogs"), ("cat", "cats"), ("car", "cars"), ("house", "houses"), ("tree", "trees"), ("book", "books"),
        ("child", "children"), ("woman", "women"), ("mouse", "mice"), ("foot", "feet"), ("tooth", "teeth"),
        ("city", "cities"), ("country", "countries"), ("box", "boxes"), ("bird", "birds"), ("horse", "horses"),
        ("apple", "apples"), ("day", "days"), ("year", "years"), ("eye", "eyes"), ("hand", "hands"),
        ("idea", "ideas"), ("problem", "problems"), ("person", "people"), ("knife", "knives"), ("leaf", "leaves"),
        ("story", "stories"), ("church", "churches"), ("student", "students"), ("window", "windows"),
    ],
    "comparative": [
        ("big", "bigger"), ("small", "smaller"), ("good", "better"), ("bad", "worse"), ("fast", "faster"),
        ("old", "older"), ("young", "younger"), ("large", "larger"), ("high", "higher"), ("long", "longer"),
        ("strong", "stronger"), ("easy", "easier"), ("hard", "harder"), ("cheap", "cheaper"), ("warm", "warmer"),
        ("cold", "colder"), ("happy", "happier"), ("short", "shorter"), ("tall", "taller"), ("low", "lower"),
        ("dark", "darker"), ("close", "closer"), ("great", "greater"), ("deep", "deeper"), ("wide", "wider"),
        ("slow", "slower"), ("heavy", "heavier"), ("rich", "richer"), ("safe", "safer"), ("simple", "simpler"),
    ],
    "superlative": [
        ("big", "biggest"), ("small", "smallest"), ("good", "best"), ("bad", "worst"), ("fast", "fastest"),
        ("old", "oldest"), ("young", "youngest"), ("large", "largest"), ("high", "highest"), ("long", "longest"),
        ("strong", "strongest"), ("easy", "easiest"), ("hard", "hardest"), ("cheap", "cheapest"),
        ("warm", "warmest"), ("cold", "coldest"), ("happy", "happiest"), ("short", "shortest"), ("tall", "tallest"),
        ("low", "lowest"), ("dark", "darkest"), ("close", "closest"), ("great", "greatest"), ("deep", "deepest"),
        ("wide", "widest"), ("slow", "slowest"), ("heavy", "heaviest"), ("rich", "richest"), ("safe", "safest"),
        ("simple", "simplest"),
    ],
    "past_tense": [
        ("walk", "walked"), ("go", "went"), ("see", "saw"), ("run", "ran"), ("eat", "ate"), ("take", "took"),
        ("give", "gave"), ("come", "came"), ("know", "knew"), ("write", "wrote"), ("think", "thought"),
        ("make", "made"), ("find", "found"), ("say", "said"), ("tell", "told"), ("play", "played"),
        ("look", "looked"), ("work", "worked"), ("sing", "sang"), ("swim", "swam"), ("speak", "spoke"),
        ("buy", "bought"), ("sit", "sat"), ("fall", "fell"), ("drive", "drove"), ("begin", "began"),
        ("grow", "grew"), ("fly", "flew"), ("jump", "jumped"), ("ask", "asked"), ("feel", "felt"),
        ("bring", "brought"), ("stand", "stood"), ("win", "won"), ("lose", "lost"), ("open", "opened"),
        ("learn", "learned"), ("try", "tried"),
    ],
    "gerund": [
        ("walk", "walking"), ("go", "going"), ("run", "running"), ("play", "playing"), ("eat", "eating"),
        ("swim", "swimming"), ("write", "writing"), ("read", "reading"), ("sing", "singing"), ("think", "thinking"),
        ("look", "looking"), ("work", "working"), ("sit", "sitting"), ("fly", "flying"), ("make", "making"),
        ("take", "taking"), ("speak", "speaking"), ("dance", "dancing"), ("jump", "jumping"), ("listen", "listening"),
    ],
    "antonym": [
        ("hot", "cold"), ("good", "bad"), ("big", "small"), ("up", "down"), ("black", "white"), ("day", "night"),
        ("love", "hate"), ("rich", "poor"), ("happy", "sad"), ("high", "low"), ("fast", "slow"), ("old", "new"),
        ("light", "dark"), ("left", "right"), ("early", "late"), ("hard", "soft"), ("top", "bottom"),
        ("inside", "outside"), ("always", "never"), ("before", "after"), ("first", "last"), ("wet", "dry"),
        ("strong", "weak"), ("alive", "dead"), ("friend", "enemy"), ("war", "peace"), ("north", "south"),
        ("east", "west"), ("buy", "sell"), ("win", "lose"),
    ],
    "negation_prefix": [
        ("possible", "impossible"), ("happy", "unhappy"), ("known", "unknown"), ("likely", "unlikely"),
        ("fair", "unfair"), ("certain", "uncertain"), ("able", "unable"), ("comfortable", "uncomfortable"),
        ("aware", "unaware"), ("honest", "dishonest"), ("legal", "illegal"), ("correct", "incorrect"),
        ("complete", "incomplete"), ("visible", "invisible"), ("usual", "unusual"), ("necessary", "unnecessary"),
        ("expected", "unexpected"), ("clear", "unclear"), ("lucky", "unlucky"), ("finished", "unfinished"),
    ],
    "code_inverse": [
        ("True", "False"), ("encode", "decode"), ("get", "set"), ("read", "write"), ("open", "close"),
        ("push", "pop"), ("load", "dump"), ("min", "max"), ("upper", "lower"), ("enable", "disable"),
        ("add", "remove"), ("start", "stop"), ("input", "output"), ("connect", "disconnect"), ("lock", "unlock"),
        ("show", "hide"), ("import", "export"), ("send", "receive"), ("serialize", "deserialize"),
        ("begin", "end"), ("enter", "exit"), ("acquire", "release"), ("insert", "delete"), ("encrypt", "decrypt"),
        ("compress", "decompress"), ("request", "response"), ("pack", "unpack"), ("append", "extend"),
    ],
    "code_abbrev": [
        ("string", "str"), ("integer", "int"), ("boolean", "bool"), ("dictionary", "dict"), ("number", "num"),
        ("argument", "arg"), ("parameter", "param"), ("message", "msg"), ("configuration", "config"),
        ("directory", "dir"), ("function", "func"), ("variable", "var"), ("index", "idx"), ("maximum", "max"),
        ("minimum", "min"), ("temporary", "temp"), ("source", "src"), ("destination", "dst"),
        ("previous", "prev"), ("length", "len"), ("character", "char"), ("object", "obj"), ("value", "val"),
        ("attribute", "attr"), ("image", "img"), ("error", "err"), ("document", "doc"), ("reference", "ref"),
        ("command", "cmd"), ("buffer", "buf"), ("context", "ctx"), ("request", "req"),
    ],
    "code_plural": [
        ("item", "items"), ("key", "keys"), ("value", "values"), ("arg", "args"), ("line", "lines"),
        ("file", "files"), ("row", "rows"), ("node", "nodes"), ("word", "words"), ("param", "params"),
        ("element", "elements"), ("result", "results"), ("token", "tokens"), ("name", "names"), ("field", "fields"),
        ("column", "columns"), ("user", "users"), ("label", "labels"), ("edge", "edges"), ("child", "children"),
    ],
}

NEIGHBOUR_WORDS = [
    " king", " queen", " Paris", " London", " dog", " happy", " three", " Monday", " January", " red", " water",
    " walked", " computer", " python", " Python", " def", " int", " list", " True", " append", " self", " she",
]

# Gender offsets compared head to head in the summary.
GENDER_SHOWCASE = [("king", "queen"), ("man", "woman"), ("prince", "princess"), ("father", "mother"), ("boy", "girl"),
                   ("he", "she")]


def load_embeddings(path: str) -> tuple[torch.Tensor, str, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd, cfg = ck["model"], ck.get("meta", {}).get("model_config") or ck.get("config") or {}
    key = "embed.weight" if "embed.weight" in sd else next(k for k, v in sd.items() if v.ndim == 2 and "emb" in k)
    return sd[key].float().clone(), key, {k: cfg.get(k) for k in ("d_model", "n_layers", "vocab_size", "tie_embeddings")}


def single_id(tok: SlmTokenizer, s: str) -> int | None:
    ids = tok.encode(s)
    return ids[0] if len(ids) == 1 else None


def treatments(E: torch.Tensor, abtt_dims: int) -> dict[str, torch.Tensor]:
    """Unit-normalised embedding variants: raw, centred, centred minus the top principal directions."""
    unit = lambda x: x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)  # noqa: E731
    C = E - E.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(C, full_matrices=False)
    top = Vh[:abtt_dims]
    return {"raw": unit(E), "centred": unit(C), "abtt": unit(C - (C @ top.T) @ top)}


def topk_excluding(Q: torch.Tensor, Eu: torch.Tensor, pool: torch.Tensor, excl: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k candidate ids by cosine for each query row; ids outside `pool` and the per-row `excl` ids are barred."""
    out = []
    for s in range(0, Q.shape[0], 512):
        S = Q[s : s + 512] @ Eu.T
        S[:, ~pool] = float("-inf")
        S.scatter_(1, excl[s : s + 512], float("-inf"))
        out.append(S.topk(k, dim=1).indices)
    return torch.cat(out) if out else torch.empty(0, k, dtype=torch.long)


def run_form(tok, Ev: dict[str, torch.Tensor], pools: dict[str, torch.Tensor], prefix: str, k: int = 5) -> dict:
    """Analogy accuracy for every category with words in one surface form (prefix " " or "")."""
    res = {}
    low = [tok.token_str(i).strip().lower() for i in range(tok.base_vocab)]
    for cat, pairs in CATEGORIES.items():
        kept, dropped = [], []
        for a, b in pairs:
            ia, ib = single_id(tok, prefix + a), single_id(tok, prefix + b)
            (kept if ia is not None and ib is not None else dropped).append((a, b, ia, ib))
        quads = [(p[2], p[3], q[2], q[3]) for p in kept for q in kept if p is not q and p[2] != q[2] and p[3] != q[3]]
        entry = {"pairs_total": len(pairs), "pairs_kept": len(kept), "quads": len(quads),
                 "kept": [f"{a}:{b}" for a, b, *_ in kept], "dropped": [f"{a}:{b}" for a, b, *_ in dropped],
                 "acc": {}, "examples": {}}
        if quads:
            T = torch.tensor(quads)
            a, b, c, d = T.unbind(1)
            excl = T[:, :3]
            for tname, Eu in Ev.items():
                Q = Eu[b] - Eu[a] + Eu[c]
                Q = Q / Q.norm(dim=1, keepdim=True)
                for pname, pool in pools.items():
                    top = topk_excluding(Q, Eu, pool, excl, k)
                    top_c = topk_excluding(Eu[c], Eu, pool, excl, 1)
                    hit1 = top[:, 0] == d
                    # A miss whose top-1 is c itself in another case (" Bad" for " bad") or c with another ending
                    # (" sees" for " see"): the query stayed next to c instead of moving along the relation.
                    norm = lambda i: low[int(i)]  # noqa: E731
                    miss = (~hit1).nonzero().squeeze(1).tolist()
                    near_c = sum(1 for i in miss if norm(top[i, 0])[:4] == norm(c[i])[:4])
                    # Lenient top-1: skip candidates that only differ from a, b or c in case or spacing.
                    ci_hits = 0
                    for i in range(len(quads)):
                        skip = {norm(a[i]), norm(b[i]), norm(c[i])}
                        first = next((int(j) for j in top[i] if norm(j) not in skip), None)
                        ci_hits += first == int(d[i])
                    entry["acc"][f"{pname}/{tname}"] = {
                        "top1": hit1.float().mean().item(),
                        "top5": (top == d[:, None]).any(1).float().mean().item(),
                        "c_only_top1": (top_c[:, 0] == d).float().mean().item(),
                        "top1_case_insensitive": ci_hits / len(quads),
                        "misses": len(miss),
                        "misses_top1_is_variant_of_c": near_c,
                    }
                    entry["examples"][f"{pname}/{tname}"] = [
                        {"q": f"{tok.token_str(int(a[i]))!r}:{tok.token_str(int(b[i]))!r} :: "
                              f"{tok.token_str(int(c[i]))!r}:{tok.token_str(int(d[i]))!r}",
                         "hit": bool(hit1[i]), "top5": [tok.token_str(int(j)) for j in top[i]]}
                        for i in range(len(quads))
                    ]
        res[cat] = entry
    return res


def offset_consistency(tok, Eu: torch.Tensor, wordlike_ids: torch.Tensor, seed: int = 0) -> dict:
    """Mean pairwise cosine between the pair offsets v(b)-v(a) of a category, against two baselines: the same
    category's words re-paired at random (keeps the class-level direction, breaks the pairing) and random word pairs."""
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)

    def mean_offdiag_cos(D: torch.Tensor) -> float:
        D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-8)
        M = D @ D.T
        n = M.shape[0]
        return ((M.sum() - M.diagonal().sum()) / (n * (n - 1))).item()

    out = {}
    for cat, pairs in CATEGORIES.items():
        ids = [(single_id(tok, " " + a), single_id(tok, " " + b)) for a, b in pairs]
        ids = [p for p in ids if None not in p]
        if len(ids) < 3:
            continue
        A, B = torch.tensor([p[0] for p in ids]), torch.tensor([p[1] for p in ids])
        shuf = []
        for _ in range(50):
            perm = list(range(len(ids)))
            while any(i == j for i, j in enumerate(perm)):
                rng.shuffle(perm)
            shuf.append(mean_offdiag_cos(Eu[B[perm]] - Eu[A]))
        out[cat] = {"n_pairs": len(ids), "matched": mean_offdiag_cos(Eu[B] - Eu[A]),
                    "shuffled": sum(shuf) / len(shuf)}
    R = wordlike_ids[torch.randint(len(wordlike_ids), (2, 2000), generator=g)]
    out["_random_word_pairs"] = {"n_pairs": 2000, "matched": mean_offdiag_cos(Eu[R[1]] - Eu[R[0]])}
    sc = [(single_id(tok, " " + a), single_id(tok, " " + b)) for a, b in GENDER_SHOWCASE]
    sc_ok = [(n, p) for n, p in zip(GENDER_SHOWCASE, sc) if None not in p]
    if sc_ok:
        D = torch.stack([Eu[q] - Eu[p] for _, (p, q) in sc_ok])
        D = D / D.norm(dim=1, keepdim=True)
        out["_gender_showcase"] = {"labels": [f"{a}-{b}" for (a, b), _ in sc_ok], "cos": (D @ D.T).tolist()}
    return out


def neighbours(tok, Ev: dict[str, torch.Tensor], pool: torch.Tensor, k: int = 8) -> dict:
    out = {}
    for w in NEIGHBOUR_WORDS:
        i = single_id(tok, w)
        if i is None:
            out[w] = None
            continue
        out[w] = {}
        for tname, Eu in Ev.items():
            S = Eu @ Eu[i]
            S[~pool] = float("-inf")
            S[i] = float("-inf")
            v, j = S.topk(k)
            out[w][tname] = [[tok.token_str(int(t)), round(float(s), 3)] for s, t in zip(v, j)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default="runs/m10_rl_336m/checkpoints/final.pt")
    ap.add_argument("--tokenizer", default=r"C:\slm-data\tokenizer\v1")
    ap.add_argument("--out", default="runs/embeddings/m10_analogies.json")
    ap.add_argument("--abtt-dims", type=int, default=10, help="principal directions removed in the abtt treatment")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads (the GPU is never used)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # token strings include U+FFFD byte fragments
    torch.set_num_threads(args.threads)

    tok = SlmTokenizer.load(args.tokenizer)
    E, key, cfg = load_embeddings(args.checkpoint)
    V = tok.base_vocab
    E = E[:V]  # drop the reserved special-token block at the top of the vocab
    strs = [tok.token_str(i) for i in range(V)]
    word_sp = torch.tensor([bool(re.fullmatch(r" [A-Za-z]+", s)) for s in strs])
    word_bare = torch.tensor([bool(re.fullmatch(r"[A-Za-z]+", s)) for s in strs])
    full = torch.ones(V, dtype=torch.bool)
    Ev = treatments(E, args.abtt_dims)

    norms = E.norm(dim=1)
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(V, (2, 20000), generator=g)
    aniso = {t: (Eu[idx[0]] * Eu[idx[1]]).sum(1).mean().item() for t, Eu in Ev.items()}
    stats = {"checkpoint": args.checkpoint, "embedding_key": key, "model_config": cfg, "bpe_vocab": V,
             "wordlike_space_tokens": int(word_sp.sum()), "wordlike_bare_tokens": int(word_bare.sum()),
             "mean_row_norm": norms.mean().item(), "mean_vector_norm": E.mean(0).norm().item(),
             "mean_cos_random_token_pairs": aniso}

    space = run_form(tok, Ev, {"full": full, "wordlike": word_sp}, " ")
    bare = run_form(tok, Ev, {"full": full, "wordlike": word_bare}, "")
    wl_ids = word_sp.nonzero().squeeze(1)
    offsets = {t: offset_consistency(tok, Eu, wl_ids) for t, Eu in Ev.items()}
    nn = neighbours(tok, {t: Ev[t] for t in ("raw", "centred")}, full)

    # ---- summary
    print(f"\n{args.checkpoint}  [{key} {tuple(E.shape)}; tied={cfg.get('tie_embeddings')}]")
    print(f"word-like pool: {stats['wordlike_space_tokens']} leading-space alphabetic tokens of {V}; "
          f"mean cos of random token pairs raw {aniso['raw']:.3f} / centred {aniso['centred']:.3f}; "
          f"|mean| / mean|row| = {stats['mean_vector_norm'] / stats['mean_row_norm']:.3f}")
    conds = [f"{p}/{t}" for p in ("full", "wordlike") for t in ("raw", "centred", "abtt")]
    for form, res in (("leading-space form", space), ("bare form", bare)):
        print(f"\nAnalogy accuracy, {form}: top1/top5 % (c-only = nearest neighbour of c alone, wordlike/centred)")
        print(f"{'category':16s} {'pairs':>6s} {'quads':>6s}  " + "  ".join(f"{c:>16s}" for c in conds) + f"  {'c-only':>6s}")
        tot = {c: [0.0, 0.0] for c in conds}
        nq = 0
        for cat, r in res.items():
            if not r["quads"]:
                print(f"{cat:16s} {r['pairs_kept']:>3d}/{r['pairs_total']:<2d} {0:>6d}")
                continue
            nq += r["quads"]
            cells = []
            for c in conds:
                a = r["acc"][c]
                tot[c][0] += a["top1"] * r["quads"]
                tot[c][1] += a["top5"] * r["quads"]
                cells.append(f"{100 * a['top1']:7.1f}/{100 * a['top5']:5.1f}  ")
            co = 100 * r["acc"]["wordlike/centred"]["c_only_top1"]
            print(f"{cat:16s} {r['pairs_kept']:>3d}/{r['pairs_total']:<2d} {r['quads']:>6d}  " + "".join(f"{x:>18s}" for x in cells) + f"{co:6.1f}")
        print(f"{'ALL (micro)':16s} {'':6s} {nq:>6d}  " + "".join(
            f"{f'{100 * tot[c][0] / nq:7.1f}/{100 * tot[c][1] / nq:5.1f}  ':>18s}" for c in conds))
        if form.startswith("leading"):
            m = [(cat, r["acc"]["wordlike/centred"]) for cat, r in res.items() if r["quads"]]
            print("top-1 misses (wordlike/centred) whose answer is a case/inflection variant of c (same first 4 letters): "
                  + ", ".join(f"{cat} {a['misses_top1_is_variant_of_c']}/{a['misses']}" for cat, a in m))
            nq = sum(res[cat]["quads"] for cat, _ in m)
            print("lenient top-1 (wordlike/centred; case/space variants of a, b, c skipped) %: "
                  + ", ".join(f"{cat} {100 * a['top1_case_insensitive']:.1f}" for cat, a in m)
                  + f"; ALL {100 * sum(a['top1_case_insensitive'] * res[cat]['quads'] for cat, a in m) / nq:.1f}")

    print("\nOffset consistency: mean pairwise cos of v(b)-v(a) within a category (leading-space form)")
    print(f"{'category':16s} {'pairs':>5s}  " + "  ".join(f"{t + ' match/shuf':>20s}" for t in Ev))
    for cat in offsets["raw"]:
        if cat.startswith("_"):
            continue
        print(f"{cat:16s} {offsets['raw'][cat]['n_pairs']:5d}  " + "  ".join(
            f"{offsets[t][cat]['matched']:9.3f}/{offsets[t][cat]['shuffled']:<10.3f}" for t in Ev))
    print(f"{'random words':16s} {2000:5d}  " + "  ".join(f"{offsets[t]['_random_word_pairs']['matched']:9.3f}{'':11s}" for t in Ev))
    sc = offsets["centred"].get("_gender_showcase")
    if sc:
        print("\nGender offsets head to head (centred cos):")
        print(" " * 16 + "".join(f"{lab:>16s}" for lab in sc["labels"]))
        for lab, row in zip(sc["labels"], sc["cos"]):
            print(f"{lab:16s}" + "".join(f"{x:16.3f}" for x in row))

    print("\nExamples (wordlike/centred, leading-space form): first two hits and misses per category")
    for cat, r in space.items():
        ex = r["examples"].get("wordlike/centred", [])
        for want in (True, False):
            for e in [e for e in ex if e["hit"] == want][:2]:
                print(f"  {cat:15s} {'HIT ' if want else 'miss'} {e['q']:52s} -> {e['top5']}")

    print("\nNearest neighbours (full vocab, centred; raw in JSON):")
    for w, r in nn.items():
        print(f"  {w!r:13s} " + ("(not a single token)" if r is None else ", ".join(repr(s) for s, _ in r["centred"])))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stats": stats, "analogies_space": space, "analogies_bare": bare,
                               "offset_consistency": offsets, "neighbours": nn}, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
