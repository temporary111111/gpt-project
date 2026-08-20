"""Real multi-turn terminal chat interface over a locally trained checkpoint.

Maintains conversation history, uses <user>/<assistant> role markers, keeps the
recent complete turns within the model context (256 tokens, oldest dropped
first), stops at EOS and defensively at unexpected role markers, and never
prints raw special tokens.

Commands:
  exit / quit          leave
  clear / reset        clear the conversation history
  /system TEXT         set a system prompt (default: empty for V1)

Usage:
  .\\.venv\\Scripts\\python.exe -m src.chat --checkpoint checkpoints\\chat_v1\\best.pt --tokenizer-model data\\tokenizer\\tokenizer_v1.model
"""

from __future__ import annotations

import argparse

import torch

from .generate import generate
from .model import GPTModel, ModelConfig
from .tokenizer import SentencePieceTokenizer

ROLE_MARKERS = {5, 6}  # <user>, <assistant>


def trim_reply_ids(ids, role_markers: set, eos_id: int) -> list:
    """Defensively stops a raw reply at the first EOS or unexpected role marker."""
    trimmed = []
    for tok_id in ids:
        if tok_id in role_markers or tok_id == eos_id:
            break
        trimmed.append(tok_id)
    return trimmed


def build_history_ids(tokenizer, history, system_text: str) -> list:
    """Turns (role, text) history into prompt ids, keeping RECENT turns within ctx.

    TASK 005.2 (Part D/G): the most RECENT complete turns are preserved first;
    the OLDEST turns are dropped first when the context window is full — the
    same convention as tokenize_example() in the SFT dataset builder.
    """
    cfg_ctx = 256
    ids: list = [tokenizer.bos_id]
    if system_text:
        ids += [tokenizer.system_id] + tokenizer.encode(system_text)
    # Final user turn + <assistant> marker are always kept.
    core = []
    core += [tokenizer.user_id] + tokenizer.encode(history[-1][1])
    core += [tokenizer.assistant_id]
    budget = cfg_ctx - len(ids) - len(core)
    prefix: list = []
    kept_turns: list = []
    kept_len = 0
    for role, text in reversed(history[:-1]):
        turn = ([tokenizer.user_id] + tokenizer.encode(text)) if role == "user" else \
               ([tokenizer.assistant_id] + tokenizer.encode(text))
        if kept_len + len(turn) > budget:
            break
        kept_turns.append(turn)
        kept_len += len(turn)
    for turn in reversed(kept_turns):
        prefix.extend(turn)
    return ids + prefix + core


def main():
    p = argparse.ArgumentParser(description="Terminal chat with the local model")
    p.add_argument("--config", default="configs/sft_chat_v1.json")
    p.add_argument("--checkpoint", default="checkpoints/chat_v1/best.pt")
    p.add_argument("--tokenizer-model", required=True,
                   help="path to trained SentencePiece .model")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="optional manual seed for reproducible sampling (default: random)")
    p.add_argument("--system", default="",
                   help="optional system prompt (default: empty for V1)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig.from_json(args.config)

    tokenizer = SentencePieceTokenizer(args.tokenizer_model)
    special = tokenizer.special_token_ids
    cfg.bos_id, cfg.eos_id, cfg.pad_id, cfg.unk_id = (
        special["bos"], special["eos"], special["pad"], special["unk"])

    model = GPTModel(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    history: list = []  # (role, text)
    print("Chat with the local from-scratch SFT model (exit/quit to leave, clear/reset to reset).")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if user_text.lower() in ("exit", "quit"):
            break
        if user_text.lower() in ("clear", "reset"):
            history = []
            print("[history cleared]")
            continue
        if user_text.lower().startswith("/system "):
            args.system = user_text[len("/system "):].strip()
            print(f"[system prompt set]")
            continue
        if not user_text:
            continue

        history.append(("user", user_text))
        prompt_ids = build_history_ids(tokenizer, history, args.system)
        full = generate(model, tokenizer, prompt_ids, args.max_new_tokens,
                        temperature=args.temperature, top_k=args.top_k,
                        top_p=args.top_p, seed=args.seed)
        reply_ids = full[len(prompt_ids):]
        # Defensive: stop the reply at an unexpected role marker or EOS.
        trimmed = trim_reply_ids(reply_ids, ROLE_MARKERS, tokenizer.eos_id)
        reply = tokenizer.decode(trimmed)
        if reply:
            history.append(("assistant", reply))
        print(f"AI: {reply}\n")


if __name__ == "__main__":
    main()