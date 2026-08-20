"""Chat evaluation (TASK 005 Part K/S/T/U).

Part K: baseline chat evaluation of checkpoints/pretrain_v1/best.pt BEFORE any
SFT weight modification (fixed seed, 20 fixed prompts, greedy + sampled).
Part S: post-SFT comparison of pretrain_v1/best.pt vs chat_v1/best.pt with
identical prompts/seeds + multi-turn context probes.
Part T: deterministic sanity metrics (empty-response rate, EOS termination
rate, role-token leakage rate, <unk> rate, replacement-character rate,
response length, repetition score, language match estimate).

Outputs:
  checkpoints/chat_v1/baseline_chat_samples.txt   (Part K)
  checkpoints/chat_v1/chat_comparison.txt         (Part S)
  checkpoints/chat_v1/eval_metrics.json           (Part T)

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\evaluate_chat.py --mode baseline
  .\\.venv\\Scripts\\python.exe scripts\\evaluate_chat.py --mode compare
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generate import generate  # noqa: E402
from src.model import GPTModel, ModelConfig  # noqa: E402
from src.tokenizer import SentencePieceTokenizer  # noqa: E402

CHAT_OUT = ROOT / "checkpoints" / "chat_v1"

FILIPINO_PROMPTS = [
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
]
ENGLISH_PROMPTS = [
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
]
ALL_PROMPTS = FILIPINO_PROMPTS + ENGLISH_PROMPTS

MULTI_TURN_PROBES = [
    ("Pangalan ng aso ko ay Bruno.", "Ano nga ang pangalan ng aso ko?"),
    ("My dog's name is Bruno.", "What is my dog's name?"),
    # TASK 005.1 Part U NEW held-out multi-turn probes (never in training)
    ("Kumain ako ng saging kanina.", "Ano ang kinain ko?"),
    ("Ang paborito kong kulay ay berde.", "Anong kulay ang paborito ko?"),
    ("Nakatira ako sa Maynila.", "Saan ako nakatira?"),
    ("I visited Paris last summer.", "Where did I go last summer?"),
    ("My favorite food is pizza.", "What is my favorite food?"),
    ("I have two cats named Oreo and Luna.", "What are my cats' names?"),
]

EVAL_SEED = 42


def load_model(checkpoint: str, tokenizer, device: torch.device) -> GPTModel:
    cfg = ModelConfig.from_json(ROOT / "configs" / "sft_chat_v1.json")
    special = tokenizer.special_token_ids
    cfg.bos_id, cfg.eos_id, cfg.pad_id, cfg.unk_id = (
        special["bos"], special["eos"], special["pad"], special["unk"])
    model = GPTModel(cfg)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def chat_prompt_ids(tokenizer, turns: List[Tuple[str, str]]) -> List[int]:
    ids = [tokenizer.bos_id]
    for role, text in turns:
        marker = tokenizer.user_id if role == "user" else tokenizer.assistant_id
        ids += [marker] + tokenizer.encode(text)
    ids += [tokenizer.assistant_id]
    return ids


def generate_reply(model, tokenizer, prompt_ids, max_new_tokens, temperature,
                   top_k, top_p, seed) -> List[int]:
    full = generate(model, tokenizer, prompt_ids, max_new_tokens,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    seed=seed, stop_at_eos=True)
    return full[len(prompt_ids):]


def sample_metrics(tokenizer, reply_ids: List[int], reply_text: str) -> Dict:
    special = set(tokenizer.special_token_ids.values())
    unk = tokenizer.unk_id
    return {
        "tokens": len(reply_ids),
        "eos_terminated": bool(reply_ids and reply_ids[-1] == tokenizer.eos_id),
        "role_leak": any(i in special and i not in (tokenizer.eos_id, tokenizer.unk_id) for i in reply_ids),
        "unk_tokens": reply_ids.count(unk),
        "replacement_chars": reply_text.count("\ufffd"),
        "repetition_score": repetition_score(reply_text),
    }


def repetition_score(text: str) -> float:
    """Ratio of repeated word bigrams (0.0 = none repeated, 1.0 = all repeated)."""
    words = re.findall(r"\S+", text.lower())
    if len(words) < 3:
        return 0.0
    bigrams = [f"{a} {b}" for a, b in zip(words, words[1:])]
    return 1.0 - len(set(bigrams)) / max(1, len(bigrams))


EN_WORDS = re.compile(r"[a-z']+")
TL_MARKS = {"ang", "ng", "mga", "sa", "ko", "ka", "mo", "namin", "ninyo", "ako",
            "ikaw", "siya", "kayo", "tayo", "kami", "kita", "yan", "iyon", "ito",
            "ba", "po", "opo", "hindi", "oo", "at", "o", "ngunit", "dahil", "kaya",
            "ngayon", "bukas", "kahapon", "gawin", "pwede", "ano", "bakit", "saan",
            "kailan", "sino", "paano", "gusto", "kailangan", "mabuti", "masama"}


def language_match_estimate(text: str) -> Dict:
    """Deterministic language estimate based on distinctive Filipino function words."""
    words = [w for w in EN_WORDS.findall(text.lower())]
    tl_hits = sum(1 for w in words if w in TL_MARKS)
    if not words:
        return {"estimate": "unknown", "tl_hits": 0, "tl_ratio": 0.0}
    return {"estimate": "fil" if tl_hits / len(words) >= 0.25 else "eng",
            "tl_hits": tl_hits, "tl_ratio": round(tl_hits / len(words), 4)}


def aggregate(records: List[Dict]) -> Dict:
    n = max(1, len(records))
    eos = sum(1 for r in records if r["metrics"]["eos_terminated"])
    leak = sum(1 for r in records if r["metrics"]["role_leak"])
    unk = sum(r["metrics"]["unk_tokens"] for r in records)
    repl = sum(r["metrics"]["replacement_chars"] for r in records)
    empty = sum(1 for r in records if r["metrics"]["tokens"] == 0)
    lens = [r["metrics"]["tokens"] for r in records]
    reps = [r["metrics"]["repetition_score"] for r in records]
    return {
        "n_responses": len(records),
        "empty_response_rate": round(empty / n, 4),
        "eos_termination_rate": round(eos / n, 4),
        "role_token_leakage_rate": round(leak / n, 4),
        "unk_rate": round(unk / max(1, sum(lens)), 6),
        "replacement_char_rate": round(repl / max(1, sum(len(r["text"]) for r in records)), 6),
        "mean_response_tokens": round(sum(lens) / n, 2),
        "mean_repetition_score": round(sum(reps) / n, 4),
        "fil_prompts_answered_in_fil": round(
            sum(1 for r in records if r["prompt_lang"] == "fil"
                and language_match_estimate(r["text"])["estimate"] == "fil") / max(1, n), 4),
        "eng_prompts_answered_in_eng": round(
            sum(1 for r in records if r["prompt_lang"] == "eng"
                and language_match_estimate(r["text"])["estimate"] == "eng") / max(1, n), 4),
    }


def run_prompts(model, tokenizer, device, max_new_tokens, temperature, top_k, top_p) -> List[Dict]:
    records = []
    for i, prompt in enumerate(ALL_PROMPTS):
        lang = "fil" if i < 10 else "eng"
        prompt_ids = chat_prompt_ids(tokenizer, [("user", prompt)])
        ids = generate_reply(model, tokenizer, prompt_ids, max_new_tokens,
                             temperature, top_k, top_p, EVAL_SEED + i)
        text = tokenizer.decode(ids)
        records.append({"prompt": prompt, "prompt_lang": lang, "text": text,
                        "metrics": sample_metrics(tokenizer, ids, text)})
    return records


def run_multiturn(model, tokenizer, device, max_new_tokens, temperature, top_k, top_p) -> List[Dict]:
    records = []
    for i, (u1, u2) in enumerate(MULTI_TURN_PROBES):
        lang = "fil" if i == 0 else "eng"
        turns = [("user", u1), ("assistant", "Salamat sa pagpapakilala!" if i == 0 else "Thanks for the introduction!")]
        prompt_ids = chat_prompt_ids(tokenizer, turns + [("user", u2)])
        ids = generate_reply(model, tokenizer, prompt_ids, max_new_tokens,
                             temperature, top_k, top_p, EVAL_SEED + 100 + i)
        records.append({"context": turns, "followup": u2, "prompt_lang": lang,
                        "text": tokenizer.decode(ids),
                        "metrics": sample_metrics(tokenizer, ids, tokenizer.decode(ids))})
    return records


def write_section(f, title: str, records: List[Dict]) -> None:
    f.write("=" * 70 + "\n" + title + "\n" + "=" * 70 + "\n")
    for r in records:
        f.write(f"\n[{r['prompt_lang'].upper()}] PROMPT: {r['prompt']}\n")
        f.write(f"RESPONSE: {r['text']}\n")


def baseline_main(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = SentencePieceTokenizer(str(ROOT / "data" / "tokenizer" / "tokenizer_v1.model"))
    model = load_model(args.checkpoint, tok, device)
    CHAT_OUT.mkdir(parents=True, exist_ok=True)

    greedy = run_prompts(model, tok, device, args.max_new_tokens, 0.0, 1, None)
    sampled = run_prompts(model, tok, device, args.max_new_tokens, args.temperature, args.top_k, args.top_p)
    with open(CHAT_OUT / "baseline_chat_samples.txt", "w", encoding="utf-8") as f:
        write_section(f, "BASELINE CHAT (greedy) — checkpoints/pretrain_v1/best.pt (TASK 005 Part K)", greedy)
        write_section(f, "BASELINE CHAT (sampled) — fixed seed", sampled)
    metrics = {
        "checkpoint": args.checkpoint,
        "greedy": aggregate(greedy),
        "sampled": aggregate(sampled),
    }
    with open(CHAT_OUT / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {CHAT_OUT / 'baseline_chat_samples.txt'}")


def compare_main(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = SentencePieceTokenizer(str(ROOT / "data" / "tokenizer" / "tokenizer_v1.model"))
    base_model = load_model("checkpoints/pretrain_v1/best.pt", tok, device)
    chat_model = load_model("checkpoints/chat_v1/best.pt", tok, device)
    CHAT_OUT.mkdir(parents=True, exist_ok=True)

    with open(CHAT_OUT / "chat_comparison.txt", "w", encoding="utf-8") as f:
        for label, model in [("PRETRAIN best.pt", base_model), ("CHAT best.pt", chat_model)]:
            greedy = run_prompts(model, tok, device, args.max_new_tokens, 0.0, 1, None)
            sampled = run_prompts(model, tok, device, args.max_new_tokens, args.temperature, args.top_k, args.top_p)
            mt = run_multiturn(model, tok, device, args.max_new_tokens, args.temperature, args.top_k, args.top_p)
            write_section(f, f"{label} — GREEDY (identical prompts/seeds)", greedy)
            write_section(f, f"{label} — SAMPLED (identical prompts/seeds)", sampled)
            write_section(f, f"{label} — MULTI-TURN CONTEXT", mt)
    print("wrote checkpoints/chat_v1/chat_comparison.txt")


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat evaluation (baseline/compare)")
    ap.add_argument("--mode", choices=["baseline", "compare"], required=True)
    ap.add_argument("--checkpoint", default="checkpoints/pretrain_v1/best.pt")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=None)
    args = ap.parse_args()
    if args.mode == "baseline":
        baseline_main(args)
    else:
        compare_main(args)


if __name__ == "__main__":
    main()