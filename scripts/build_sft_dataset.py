"""Build the TASK 005.1 SFT dataset (chat format, assistant-only loss labels).

Inputs (raw, git-ignored):
  data/sft/raw/aya_eng_fil_original.jsonl
  data/sft/raw/oasst1_en_human_messages.jsonl
  data/sft/raw/dolly_15k.jsonl
  data/sft/raw/taskmaster1_dialogs.jsonl

Processing:
  - normalization: UTF-8, Unicode NFC, whitespace cleanup (no lowercasing,
    no punctuation removal)
  - quality rejects: empty prompt/response, pathological repeated characters,
    corrupted Unicode
  - cross-source deduplication: exact normalized (prompt, response) pairs are
    removed across Aya/OASST/Dolly/Taskmaster (reported by source pair);
    exact-prompt-only duplicates are reported separately but NOT merged
  - tokenizer <unk> analysis: prompt unk rate / target unk rate; targets
    containing <unk> are rejected when practical
  - Aya: single-turn examples, deterministic 95/5 train/val split by stable
    hash of the record id (Aya test split is never used)
  - OASST: multi-turn paths (human-only context + final assistant target,
    rank==0 preferred), source-provided train/validation split; no
    message_tree_id crosses the split
  - Dolly: single-turn instruction(+context)->response, deterministic 95/5
    split by stable hash of the record id
  - Taskmaster-1: assistant turns become targets (up to 4 per conversation,
    chosen deterministically across the conversation); split by CONVERSATION
    ID (official train/dev/test dialog-ID CSVs, woz fallback deterministic
    bucket); a conversation never appears in both train and validation
  - chat format: <bos><user>U<assistant>A<eos> (multi-turn: earlier turns are
    context; ONLY the final assistant target + EOS is supervised)
  - CAUSAL next-token labels (TASK 005.2 fix): labels[i] = ids[i + 1] wherever
    the NEXT token is assistant content or the terminating EOS; the first
    assistant token is predicted from the <assistant> role marker, EOS is
    predicted from the LAST assistant content token, and the EOS input
    position is masked (-100). The same-position identity objective
    (labels[i] = ids[i]) is NEVER used.
  - context 256: keep the final user turn + complete target; keep the MOST
    RECENT complete turns and drop the OLDEST turns first; reject examples
    whose target itself cannot fit
  - evaluation probes from TASK 005 Part K/S and TASK 005.1 Part U are
    excluded from training
  - sampling weights: Filipino up to 4x (target ~15-25% effective share),
    any English source >50% of effective English tokens is deterministically
    down-weighted; weights are SAMPLING ONLY and do not count toward the
    unique-token gate

Outputs:
  data/sft/processed/sft_train.jsonl  (git-ignored)
  data/sft/processed/sft_val.jsonl    (git-ignored)
  data/sft/stats/sft_stats.json       (tracked)
  data/sft/stats/SFT_DATA_REPORT.md   (tracked)

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\build_sft_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tokenizer import SentencePieceTokenizer  # noqa: E402

RAW_DIR = ROOT / "data" / "sft" / "raw"
OUT_DIR = ROOT / "data" / "sft" / "processed"
STATS_DIR = ROOT / "data" / "sft" / "stats"
TOKENIZER_MODEL = ROOT / "data" / "tokenizer" / "tokenizer_v1.model"
TOKENIZER_META = ROOT / "data" / "tokenizer" / "tokenizer_v1_meta.json"

BLOCK_SIZE = 256
TRAIN_FRAC = 0.95

# TASK 005 Part K / Part S / TASK 005.1 Part U evaluation prompts + probes
# (NEVER in training data)
EVAL_PROBES = [
    # 20 single-turn prompts (10 Filipino + 10 English)
    "Kumusta ka?",
    "Bakit mahalaga ang tubig?",
    "Ano ang araw?",
    "Ipaliwanag nang simple kung ano ang gravity.",
    "Pagod ako ngayon. Ano ang pwede kong gawin?",
    "Ano ang pagkakaiba ng ilog at dagat?",
    "Gumawa ng maikling pangungusap tungkol sa mangga.",
    "Ano ang ibig sabihin ng kalayaan?",
    "Mahilig ako sa aso. Ano ang magandang pag-usapan natin?",
    "Ano ang 2 + 2?",
    "Hello, how are you?",
    "Why is water important?",
    "What is the Sun?",
    "Explain gravity simply.",
    "I'm tired today. What can I do?",
    "What is the difference between a river and an ocean?",
    "Write a short sentence about mangoes.",
    "What does freedom mean?",
    "I like dogs. What could we talk about?",
    "What is 2 + 2?",
    # Bruno multi-turn probes
    "Pangalan ng aso ko ay Bruno.",
    "Ano nga ang pangalan ng aso ko?",
    "My dog's name is Bruno.",
    "What is my dog's name?",
    # TASK 005.1 Part U NEW held-out multi-turn probes (never trained on)
    "Kumain ako ng saging kanina.",
    "Ano ang kinain ko?",
    "Ang paborito kong kulay ay berde.",
    "Anong kulay ang paborito ko?",
    "Nakatira ako sa Maynila.",
    "Saan ako nakatira?",
    "I visited Paris last summer.",
    "Where did I go last summer?",
    "My favorite food is pizza.",
    "What is my favorite food?",
    "I have two cats named Oreo and Luna.",
    "What are my cats' names?",
]

# Filipino effective-share target (Part G) and per-conversation cap (Part B)
FIL_TARGET_SHARE = 0.15
FIL_MAX_WEIGHT = 4.0
TM_MAX_TARGETS_PER_CONV = 4
SOURCE_BALANCE_MAX_SHARE = 0.5

PATHOLOGICAL_RUN = 15          # max run of the same character
PATHOLOGICAL_RATIO = 0.50      # max fraction of single repeated char

# Aya "eng" rows are occasionally mislabeled (Somali/Indonesian/Basque/German/
# Turkish content under language_code == "eng"). Deterministic English-check
# heuristic applied ONLY to aya lang=="eng" rows: non-Latin script, or a
# >=4-word prompt AND target both containing <2 of these function words.
ENGLISH_FUNCTION_WORDS = frozenset("""
    the a an of to in is was for on and with as are be by at from it or that
    this which what why who how does do did were will would can could should
    may might must have has had been being am not no nor but if then than so
    because although though when where there here all any both each either
    few many more most much several some such only own same very just too
    also again further
""".split())
NON_LATIN_RE = re.compile(
    r"[\u0400-\u04FF\u0600-\u06FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF\u0E00-\u0E7F]")


def norm_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_corrupted(text: str) -> bool:
    return "\ufffd" in text


def is_pathological(text: str) -> bool:
    if len(text) < 2:
        return False
    for ch in set(text):
        if ch.isspace():
            continue
        run = max(len(m.group(0)) for m in re.finditer(re.escape(ch) + "+", text))
        if run >= PATHOLOGICAL_RUN:
            return True
        if text.count(ch) / len(text) >= PATHOLOGICAL_RATIO and len(text) >= 10:
            return True
    return False


def _english_hits(text: str) -> int:
    padded = " " + text.lower() + " "
    return sum(1 for w in ENGLISH_FUNCTION_WORDS if f" {w} " in padded)


def aya_eng_looks_english(prompt: str, target: str) -> bool:
    """True if an Aya lang=="eng" example is plausibly English.

    Rejects: non-Latin script content, or (a) a >=4-word prompt AND target
    both containing almost no English function words, or (b) a >=6-word
    target AND prompt with almost no English function words — the second
    clause catches short mislabeled prompts (e.g. Somali) whose targets are
    full non-English paragraphs.
    """
    if NON_LATIN_RE.search(prompt + " " + target):
        return False
    p_hits = _english_hits(prompt)
    t_hits = _english_hits(target)
    if len(prompt.split()) >= 4 and p_hits < 2 and t_hits < 2:
        return False
    if len(target.split()) >= 6 and p_hits < 2 and t_hits < 2:
        return False
    return True


def stable_bucket(key: str, divisor: int) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % divisor


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_aya_examples(rows: List[dict]) -> List[dict]:
    examples = []
    for r in rows:
        prompt = norm_text(r["inputs"])
        target = norm_text(r["targets"])
        examples.append({
            "id": r["id"],
            "source": "aya",
            "lang": r["language_code"],
            "turns": [("user", prompt), ("assistant", target)],
            "split": "train" if stable_bucket(r["id"], 100) < TRAIN_FRAC * 100 else "val",
        })
    return examples


def build_oasst_examples(rows: List[dict]) -> Tuple[List[dict], dict]:
    by_id = {r["message_id"]: r for r in rows}
    # choose best assistant candidate per parent (rank 0 preferred, deterministic)
    best_assistant: Dict[str, dict] = {}
    for r in rows:
        if r["role"] != "assistant":
            continue
        parent = r["parent_id"]
        cur = best_assistant.get(parent)
        if cur is None:
            best_assistant[parent] = r
            continue
        rk, ck = r.get("rank"), cur.get("rank")
        if rk is not None and (ck is None or rk < ck):
            best_assistant[parent] = r
        elif rk is None and ck is not None:
            continue
        elif rk == ck and r["message_id"] < cur["message_id"]:
            best_assistant[parent] = r

    def norm_split(s: str) -> str:
        return "val" if s == "validation" else "train"

    examples = []
    skipped = Counter()
    for parent, tgt in best_assistant.items():
        path: List[dict] = []
        node = tgt
        while node is not None:
            path.append(node)
            node = by_id.get(node.get("parent_id"))
        path.reverse()
        if not path or path[0]["role"] != "prompter":
            skipped["root_not_prompter"] += 1
            continue
        turns = [(("user" if m["role"] == "prompter" else "assistant"), norm_text(m["text"]))
                 for m in path]
        if not turns or turns[-1][0] != "assistant":
            skipped["target_not_assistant"] += 1
            continue
        examples.append({
            "id": f"oasst-{tgt['message_id']}",
            "source": "oasst1",
            "lang": "en",
            "tree_id": tgt["message_tree_id"],
            "turns": turns,
            "split": norm_split(tgt["split"]),
        })
    return examples, skipped


def build_dolly_examples(rows: List[dict]) -> List[dict]:
    """Single-turn instruction(+context)->response examples (TASK 005.1)."""
    examples = []
    for r in rows:
        prompt = norm_text(r["instruction"])
        context = norm_text(r.get("context") or "")
        target = norm_text(r["response"])
        if not prompt or not target:
            continue
        if context:
            prompt = prompt + "\n\n" + context
        examples.append({
            "id": r["id"],
            "source": "dolly",
            "lang": "en",
            "category": r.get("category", ""),
            "turns": [("user", prompt), ("assistant", target)],
            "split": "train" if stable_bucket(r["id"], 100) < TRAIN_FRAC * 100 else "val",
        })
    return examples


def build_taskmaster_examples(rows: List[dict]) -> Tuple[List[dict], dict]:
    """Derive assistant-target examples from Taskmaster-1 conversations.

    Every accepted assistant turn is a candidate SFT target. Up to
    TM_MAX_TARGETS_PER_CONV targets per conversation are chosen deterministically
    across the conversation (start/middle/end coverage). A conversation is
    assigned to train or validation BEFORE deriving examples (official
    train/dev/test dialog-ID CSVs, woz fallback deterministic bucket), so a
    conversation never appears in both splits.
    """
    examples = []
    skipped = Counter()
    for conv in rows:
        utts = conv["utterances"]
        asst_idx = [i for i, u in enumerate(utts) if u["speaker"] == "assistant"]
        if len(asst_idx) <= TM_MAX_TARGETS_PER_CONV:
            chosen = asst_idx
            capped = 0
        else:
            # deterministic even coverage across the conversation
            idxs = sorted({round(k * (len(asst_idx) - 1) / (TM_MAX_TARGETS_PER_CONV - 1))
                           for k in range(TM_MAX_TARGETS_PER_CONV)})
            chosen = [asst_idx[i] for i in idxs]
            capped = len(asst_idx) - len(chosen)
        if capped:
            skipped["capped_candidate_turns"] += capped
        for target_i in chosen:
            path = utts[:target_i + 1]
            if not path or path[0]["speaker"] != "user":
                skipped["root_not_user"] += 1
                continue
            turns = [(("user" if u["speaker"] == "user" else "assistant"), norm_text(u["text"]))
                     for u in path]
            if not turns or turns[-1][0] != "assistant" or not turns[-1][1]:
                skipped["bad_turns"] += 1
                continue
            if any(not t for _, t in turns[:-1]):
                skipped["empty_context_turn"] += 1
                continue
            if any("\ufffd" in t for _, t in turns):
                skipped["corrupt_unicode"] += 1
                continue
            examples.append({
                "id": f"tm1-{conv['id']}-{target_i}",
                "source": "taskmaster1",
                "lang": "en",
                "domain": conv.get("domain", ""),
                "turns": turns,
                "split": conv["split"],
            })
    return examples, skipped


def compute_sampling_weights(train_rows: List[dict]) -> Tuple[dict, dict]:
    """Language/source-aware sampling weights (SAMPLING ONLY, not the gate).

    Filipino (Part G): a multiplier up to FIL_MAX_WEIGHT (4x) so the effective
    supervised-token share reaches FIL_TARGET_SHARE (15%) when possible;
    w = (share/(1-share)) * (english_tokens / fil_tokens), clamped to [1, 4].
    No fabrication or translation: this only changes how often existing
    Filipino examples are seen.

    English source balance (Part H): if any English source would exceed
    SOURCE_BALANCE_MAX_SHARE (50%) of the effective English target tokens, the
    OTHER English sources are deterministically up-weighted by
    dominant/others so the dominant source lands at 50%. Iterates until no
    source exceeds 50% (or 10 iterations).

    Returns (sampling_report, copies_by_source). copies = max(1, round(weight)).
    """
    fil_tok = sum(r["n_supervised"] for r in train_rows if r["lang"] == "fil")
    en_rows = [r for r in train_rows if r["lang"] != "fil"]
    en_sources = sorted({r["source"] for r in en_rows})

    w_en = {s: 1.0 for s in en_sources}
    for _ in range(10):
        src_tok = {s: sum(r["n_supervised"] for r in en_rows if r["source"] == s) * w_en[s]
                   for s in en_sources}
        total = sum(src_tok.values())
        dom = max(src_tok, key=src_tok.get)
        dom_share = src_tok[dom] / max(1.0, total)
        if dom_share <= SOURCE_BALANCE_MAX_SHARE:
            break
        others = total - src_tok[dom]
        m = src_tok[dom] / max(1.0, others)
        for s in en_sources:
            if s != dom:
                w_en[s] *= m

    eff_en_tokens = sum(sum(r["n_supervised"] for r in en_rows if r["source"] == s) * w_en[s]
                        for s in en_sources)
    fil_weight = 1.0
    if fil_tok > 0 and eff_en_tokens > 0:
        needed = (FIL_TARGET_SHARE / (1.0 - FIL_TARGET_SHARE)) * (eff_en_tokens / fil_tok)
        fil_weight = min(FIL_MAX_WEIGHT, max(1.0, needed))

    for r in train_rows:
        w = w_en.get(r["source"], 1.0)
        if r["lang"] == "fil":
            w *= fil_weight
        r["weight"] = round(w, 4)
        r["copies"] = max(1, int(round(w)))

    effective_examples = sum(r["copies"] for r in train_rows)
    effective_tokens = sum(r["n_supervised"] * r["copies"] for r in train_rows)
    effective_fil_tokens = sum(r["n_supervised"] * r["copies"]
                               for r in train_rows if r["lang"] == "fil")
    eff_fil_share = effective_fil_tokens / max(1, effective_tokens)
    eff_en = {s: sum(r["n_supervised"] * r["copies"] for r in train_rows
                     if r["lang"] != "fil" and r["source"] == s)
              for s in en_sources}
    return {
        "fil_weight": round(fil_weight, 4),
        "english_source_weights": {s: round(w, 4) for s, w in sorted(w_en.items())},
        "effective_examples": effective_examples,
        "effective_supervised_tokens": effective_tokens,
        "effective_fil_tokens": effective_fil_tokens,
        "effective_fil_share": round(eff_fil_share, 4),
        "effective_english_source_tokens": eff_en,
        "effective_english_source_shares": {
            s: round(v / max(1, sum(eff_en.values())), 4) for s, v in eff_en.items()
        },
    }, {r["source"]: r["copies"] for r in train_rows}


def tokenize_example(ex: dict, tok: SentencePieceTokenizer,
                     stats: Counter, block_size: int = BLOCK_SIZE) -> Optional[dict]:
    """Builds (ids, labels) in chat format with CAUSAL next-token supervision.

    Label convention (TASK 005.2 root-cause fix): labels[i] = ids[i + 1]
    whenever the NEXT token is part of the final assistant target or the
    terminating EOS; every other position is -100. So the model predicts the
    FIRST assistant token from the <assistant> role marker, each following
    assistant token from the previous one, and EOS from the LAST assistant
    content token. The EOS input position itself is masked (-100) because
    there is no next token. The same-position identity objective
    (labels[i] = ids[i]) is NEVER used.

    Multi-turn context: the most RECENT complete turns are preserved first;
    the OLDEST turns are dropped first when the context window is full.
    The final user turn and final assistant target are never dropped.
    """
    ids: List[int] = [tok.bos_id]

    # 1) final user turn + final assistant target must fit (always kept)
    core_turns = ex["turns"]
    final_user, final_assistant = core_turns[-2], core_turns[-1]
    core_ids = ([tok.user_id] + tok.encode(final_user[1])
                + [tok.assistant_id] + tok.encode(final_assistant[1]) + [tok.eos_id])
    if len(ids) + len(core_ids) > block_size:
        stats["rejected_target_too_long"] += 1
        return None

    # 2) prepend the most RECENT older complete turns while they fit
    #    (iterate newest -> oldest so the OLDEST turns are dropped first)
    budget = block_size - len(ids) - len(core_ids)
    kept_turns: List[List[int]] = []
    kept_len = 0
    for role, text in reversed(core_turns[:-2]):
        turn_ids = ([tok.user_id] + tok.encode(text)) if role == "user" else \
                   ([tok.assistant_id] + tok.encode(text))
        if kept_len + len(turn_ids) > budget:
            stats["dropped_oldest_turns"] += 1
            break
        kept_turns.append(turn_ids)
        kept_len += len(turn_ids)
    prefix: List[int] = []
    for turn_ids in reversed(kept_turns):
        prefix.extend(turn_ids)

    full_ids = ids + prefix + core_ids
    # position of the first assistant CONTENT token; causal supervision starts
    # at the <assistant> role marker (target_start - 1) and ends at the last
    # assistant content position (len - 2), predicting ids[i + 1].
    target_start = len(full_ids) - len(tok.encode(final_assistant[1])) - 1  # -1 = eos
    full_labels = [-100] * len(full_ids)
    for i in range(target_start - 1, len(full_ids) - 1):
        full_labels[i] = full_ids[i + 1]
    # the final input position (EOS) stays -100: there is no next token

    n_supervised = (len(full_ids) - 1) - (target_start - 1)
    return {"ids": full_ids, "labels": full_labels,
            "target_start": target_start, "n_supervised": n_supervised}


def cross_source_dedup(examples: List[dict]) -> Tuple[List[dict], Counter]:
    """Remove exact normalized (prompt, response) pairs across sources.

    Exact-prompt-only duplicates are reported but NOT merged. Returns
    (kept, stats) where stats includes rejected_duplicates and dup_pair_<src|src>
    counts for each source pair that collided.
    """
    stats = Counter()
    seen_pairs: dict = {}
    seen_prompts: dict = {}
    kept: List[dict] = []
    for ex in examples:
        prompt = ex["turns"][-2][1]
        target = ex["turns"][-1][1]
        pair_key = (prompt, target)
        if pair_key in seen_pairs:
            stats["rejected_duplicates"] += 1
            src_pair = "|".join(sorted({seen_pairs[pair_key], ex["source"]}))
            stats[f"dup_pair_{src_pair}"] += 1
            continue
        seen_pairs[pair_key] = ex["source"]
        if prompt in seen_prompts:
            stats["prompt_only_duplicates"] += 1
        seen_prompts[prompt] = ex["source"]
        kept.append(ex)
    return kept, stats


def exclude_eval_probes(examples: List[dict], probes: Optional[List[str]] = None) -> Tuple[List[dict], int]:
    """Exclude any example whose ANY user turn equals a held-out eval probe."""
    probe_set = {norm_text(p).lower() for p in (probes or EVAL_PROBES)}
    kept = []
    rejected = 0
    for ex in examples:
        user_turns = [t for r, t in ex["turns"] if r == "user"]
        if any(t.lower() in probe_set for t in user_turns):
            rejected += 1
            continue
        kept.append(ex)
    return kept, rejected


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the TASK 005.1 SFT dataset")
    ap.add_argument("--max-target-unk-rate", type=float, default=0.0,
                    help="reject examples whose target contains <unk> unless rate is below this threshold")
    args = ap.parse_args()

    tok = SentencePieceTokenizer(str(TOKENIZER_MODEL), str(TOKENIZER_META))
    special = tok.special_token_ids
    print(f"tokenizer: vocab={tok.vocab_size} unk={special['unk']} bos={special['bos']} "
          f"eos={special['eos']} user={special['user']} assistant={special['assistant']}")

    stats: Counter = Counter()
    aya_rows = load_jsonl(RAW_DIR / "aya_eng_fil_original.jsonl")
    oasst_rows = load_jsonl(RAW_DIR / "oasst1_en_human_messages.jsonl")
    dolly_rows = load_jsonl(RAW_DIR / "dolly_15k.jsonl")
    tm_rows = load_jsonl(RAW_DIR / "taskmaster1_dialogs.jsonl")
    for name, n in [("aya", len(aya_rows)), ("oasst", len(oasst_rows)),
                    ("dolly", len(dolly_rows)), ("taskmaster", len(tm_rows))]:
        stats[f"raw_{name}"] = n

    aya_examples = build_aya_examples(aya_rows)
    oasst_examples, oasst_skipped = build_oasst_examples(oasst_rows)
    dolly_examples = build_dolly_examples(dolly_rows)
    tm_examples, tm_skipped = build_taskmaster_examples(tm_rows)
    stats.update({f"oasst_skipped_{k}": v for k, v in oasst_skipped.items()})
    stats.update({f"tm_skipped_{k}": v for k, v in tm_skipped.items()})
    stats["aya_examples"] = len(aya_examples)
    stats["oasst_examples"] = len(oasst_examples)
    stats["dolly_examples"] = len(dolly_examples)
    stats["taskmaster_examples"] = len(tm_examples)

    # normalization + quality rejects (all sources)
    kept: List[dict] = []
    for ex in aya_examples + oasst_examples + dolly_examples + tm_examples:
        turns = [(r, norm_text(t)) for r, t in ex["turns"]]
        if any(not t for _, t in turns):
            stats["rejected_empty"] += 1
            continue
        if any(is_corrupted(t) for _, t in turns):
            stats["rejected_corrupted_unicode"] += 1
            continue
        if any(is_pathological(t) for _, t in turns):
            stats["rejected_pathological_repeats"] += 1
            continue
        if ex["source"] == "aya" and ex["lang"] == "eng" \
                and not aya_eng_looks_english(turns[-2][1], turns[-1][1]):
            stats["rejected_aya_eng_mislabel"] += 1
            continue
        ex["turns"] = turns
        kept.append(ex)
    stats["after_quality"] = len(kept)

# cross-source dedup: exact normalized (prompt, response) pairs removed,
    # reported by source pair; exact-prompt-only duplicates reported separately.
    deduped, dedup_stats = cross_source_dedup(kept)
    stats.update(dedup_stats)
    stats["after_dedup"] = len(deduped)

    # eval-probe exclusion (never train on any Part K/S/U prompt or probe turn)
    no_probe, n_probe_rejected = exclude_eval_probes(deduped)
    stats["rejected_eval_probes"] = n_probe_rejected
    stats["after_probe_exclusion"] = len(no_probe)

    # tokenization + <unk> analysis + target-unk rejection
    tokenized: List[dict] = []
    unk_stats = {"prompt_tokens": 0, "prompt_unk": 0, "target_tokens": 0, "target_unk": 0}
    for ex in no_probe:
        prompt_ids = tok.encode(ex["turns"][-2][1])
        target_ids = tok.encode(ex["turns"][-1][1])
        if special["unk"] in target_ids:
            stats["rejected_target_unk"] += 1
            continue
        built = tokenize_example(ex, tok, stats)
        if built is None:
            continue
        tokenized.append({**ex, "ids": built["ids"], "labels": built["labels"],
                          "target_start": built["target_start"],
                          "n_supervised": built["n_supervised"],
                          "label_convention": "causal_next_token"})
    stats["final_examples"] = len(tokenized)

    # <unk> rates on the FINAL accepted corpus (targets with <unk> were rejected).
    # prompt = everything before the first assistant CONTENT token (target_start);
    # target = assistant content tokens + EOS.
    # Also: label identity fraction = share of supervised positions where the
    # causal next-token label equals the CURRENT input token (labels[i]==ids[i]).
    # Under the old same-position objective this was ~1.0; causally it must be
    # small (only naturally repeated tokens).
    identity_same = 0
    identity_sup = 0
    for ex in tokenized:
        ids, labels = ex["ids"], ex["labels"]
        ts = ex["target_start"]
        unk_stats["prompt_tokens"] += ts
        unk_stats["prompt_unk"] += sum(1 for v in ids[:ts] if v == special["unk"])
        unk_stats["target_tokens"] += len(ids) - ts
        unk_stats["target_unk"] += sum(1 for v in ids[ts:] if v == special["unk"])
        for i, lab in enumerate(labels):
            if lab != -100:
                identity_sup += 1
                if lab == ids[i]:
                    identity_same += 1
    label_identity_fraction = round(identity_same / max(1, identity_sup), 6)

    # splits
    train_rows, val_rows = [], []
    for ex in tokenized:
        if ex["split"] == "val":
            val_rows.append(ex)
        else:
            train_rows.append(ex)

    # sampling weights (train only; SAMPLING ONLY, never counted toward the gate)
    sampling, _per_source_copies = compute_sampling_weights(train_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "sft_train.jsonl", "w", encoding="utf-8") as f:
        for ex in train_rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "sft_val.jsonl", "w", encoding="utf-8") as f:
        for ex in val_rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    def supervised_tokens(rows) -> int:
        return sum(r["n_supervised"] for r in rows)

    def lang_counts(rows) -> Counter:
        return Counter(r["lang"] for r in rows)

    def source_counts(rows) -> Counter:
        return Counter(r["source"] for r in rows)

    unique_total = supervised_tokens(train_rows) + supervised_tokens(val_rows)
    unique_fil = (sum(r["n_supervised"] for r in train_rows if r["lang"] == "fil")
                  + sum(r["n_supervised"] for r in val_rows if r["lang"] == "fil"))
    unique_en = unique_total - unique_fil

    sft_stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": str(TOKENIZER_MODEL),
        "block_size": BLOCK_SIZE,
        "label_convention": "causal_next_token (labels[i] = ids[i+1] over the final "
                            "assistant target + EOS; never same-position labels)",
        "label_identity_fraction": label_identity_fraction,
        "unique_examples": len(tokenized),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "unique_supervised_target_tokens": unique_total,
        "train_supervised_target_tokens": supervised_tokens(train_rows),
        "val_supervised_target_tokens": supervised_tokens(val_rows),
        "unique_fil_target_tokens": unique_fil,
        "unique_en_target_tokens": unique_en,
        "train_lang_examples": dict(lang_counts(train_rows)),
        "val_lang_examples": dict(lang_counts(val_rows)),
        "train_lang_target_tokens": {
            k: sum(r["n_supervised"] for r in train_rows if r["lang"] == k)
            for k in sorted(lang_counts(train_rows))
        },
        "val_lang_target_tokens": {
            k: sum(r["n_supervised"] for r in val_rows if r["lang"] == k)
            for k in sorted(lang_counts(val_rows))
        },
        "unique_source_examples": dict(source_counts(tokenized)),
        "unique_source_target_tokens": {
            s: sum(r["n_supervised"] for r in tokenized if r["source"] == s)
            for s in sorted(source_counts(tokenized))
        },
        "sampling": {
            "fil_weight": sampling["fil_weight"],
            "english_source_weights": sampling["english_source_weights"],
            "unique_train_examples": len(train_rows),
            "effective_train_examples": sampling["effective_examples"],
            "unique_train_supervised_tokens": supervised_tokens(train_rows),
            "effective_train_supervised_tokens": sampling["effective_supervised_tokens"],
            "effective_fil_supervised_tokens": sampling["effective_fil_tokens"],
            "effective_fil_share": sampling["effective_fil_share"],
            "effective_source_examples": {
                s: sum(r["copies"] for r in train_rows if r["source"] == s)
                for s in sorted(source_counts(train_rows))
            },
        },
        "gate": {
            "floor": 1_000_000,
            "UNIQUE_SUPERVISED_TARGET_TOKENS": unique_total,
            "gate_passed": unique_total >= 1_000_000,
            "note": "unique accepted human tokens BEFORE oversampling; "
                    "weighted repeats never count toward this gate",
        },
        "tokenizer_unk": {
            "prompt_token_count": unk_stats["prompt_tokens"],
            "prompt_unk_count": unk_stats["prompt_unk"],
            "prompt_unk_rate": round(unk_stats["prompt_unk"] / max(1, unk_stats["prompt_tokens"]), 6),
            "target_token_count": unk_stats["target_tokens"],
            "target_unk_count": unk_stats["target_unk"],
            "target_unk_rate": round(unk_stats["target_unk"] / max(1, unk_stats["target_tokens"]), 6),
        },
        "example_percent_with_target_unk": round(
            stats["rejected_target_unk"] / max(1, len(no_probe)) * 100, 3),
        "counters": dict(stats),
        "outputs": {
            "train": str(OUT_DIR / "sft_train.jsonl"),
            "val": str(OUT_DIR / "sft_val.jsonl"),
        },
    }

    # cross-split leakage verification (final): no source identity crosses
    train_ids = {r["id"] for r in train_rows}
    val_ids = {r["id"] for r in val_rows}
    assert not (train_ids & val_ids), "train/val id overlap"
    # Taskmaster: conversation id must not cross splits
    tm_train = {r["id"].rsplit("-", 1)[0] for r in train_rows if r["source"] == "taskmaster1"}
    tm_val = {r["id"].rsplit("-", 1)[0] for r in val_rows if r["source"] == "taskmaster1"}
    assert not (tm_train & tm_val), "Taskmaster conversation crossed train/val"
    # OASST: message_tree_id must not cross splits
    oasst_train_trees = {r["tree_id"] for r in train_rows if r["source"] == "oasst1"}
    oasst_val_trees = {r["tree_id"] for r in val_rows if r["source"] == "oasst1"}
    assert not (oasst_train_trees & oasst_val_trees), "OASST tree crossed train/val"
    sft_stats["leakage_check"] = {
        "train_val_id_overlap": len(train_ids & val_ids),
        "tm_conv_overlap": len(tm_train & tm_val),
        "oasst_tree_overlap": len(oasst_train_trees & oasst_val_trees),
    }

    with open(STATS_DIR / "sft_stats.json", "w", encoding="utf-8") as f:
        json.dump(sft_stats, f, indent=2)
        f.write("\n")

    print(json.dumps(sft_stats, indent=2))
    print("DONE.")


if __name__ == "__main__":
    main()