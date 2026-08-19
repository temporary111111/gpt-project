"""Simple terminal chat interface over a locally trained checkpoint."""

from __future__ import annotations

import argparse

import torch

from .model import GPTModel, ModelConfig


def build_prompt(tokenizer, system_text: str, user_text: str) -> list:
    tok = tokenizer
    ids: list = []
    if system_text:
        ids += [tok.system_id] + tok.encode(system_text)
    ids += [tok.user_id] + tok.encode(user_text)
    ids += [tok.assistant_id]
    return ids


def main():
    p = argparse.ArgumentParser(description="Terminal chat with the local model")
    p.add_argument("--config", default="configs/model_small.json")
    p.add_argument("--checkpoint", default="checkpoints/checkpoint.pt")
    p.add_argument("--tokenizer-model", required=True,
                   help="path to trained SentencePiece .model")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--system", default="You are a helpful AI assistant trained from scratch.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig.from_json(args.config)

    from .tokenizer import SentencePieceTokenizer
    tokenizer = SentencePieceTokenizer(args.tokenizer_model)
    special = tokenizer.special_token_ids
    cfg.bos_id, cfg.eos_id, cfg.pad_id, cfg.unk_id = (
        special["bos"], special["eos"], special["pad"], special["unk"])

    model = GPTModel(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    from .generate import generate

    print("Chat with the local from-scratch model (Ctrl+C or 'exit' to quit).")
    print("Note: the model is untrained, so replies are not yet meaningful.\n")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if user_text.lower() in ("exit", "quit"):
            break
        if not user_text:
            continue
        prompt_ids = build_prompt(tokenizer, args.system, user_text)
        ids = generate(model, tokenizer, prompt_ids, args.max_new_tokens,
                       temperature=args.temperature, top_k=args.top_k,
                       top_p=args.top_p)
        reply_ids = ids[len(prompt_ids):]
        reply = tokenizer.decode(reply_ids)
        print(f"AI: {reply}\n")


if __name__ == "__main__":
    main()