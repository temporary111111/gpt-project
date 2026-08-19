"""Smoke test: proves the whole training/generation pipeline works end-to-end
with a tiny corpus, a tiny tokenizer, and a much smaller model.

Steps: tokenizer train -> encode -> decode -> forward -> loss -> backward ->
optimizer step -> generation -> checkpoint save -> reload -> verify.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.dataset import BinaryDataset, encode_text_to_bin
from src.model import GPTModel, ModelConfig
from src.tokenizer import SentencePieceTokenizer

TINY_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "hello world this is a tiny test corpus",
    "the model learns to predict the next token",
    "attention is all you need for language models",
    "we train a small transformer from scratch",
    "the cat sat on the mat and watched the bird",
    "machine learning is a lot of fun",
    "tokens are the building blocks of text",
    "a decoder only transformer generates text one token at a time",
    "greedy decoding picks the most likely token",
    "temperature controls the randomness of sampling",
    "top k sampling limits the choice to the k most likely tokens",
    "the quick brown fox is a pangram",
    "python is a programming language",
    "pytorch is a deep learning library",
    "gradient clipping keeps training stable",
    "mixed precision saves memory on the gpu",
    "the checkpoint stores the model weights",
    "causal attention prevents looking into the future",
    "positional embeddings encode token order",
    "the loss decreases as the model improves",
    "validation loss measures generalization",
    "learning rate schedules help convergence",
    "adam optimizer combines momentum and adaptive learning rates",
    "residual connections ease gradient flow",
    "layer normalization stabilizes training",
    "the vocabulary is learned from the training corpus",
    "each sentence ends with an eos token",
    "chat interfaces format prompts with special tokens",
    "smoke tests verify that every pipeline stage works",
]


def main():
    torch.manual_seed(1234)
    print("=== SMOKE TEST: from-scratch pipeline ===")

    # 1. tiny local corpus
    corpus_path = os.path.join("data", "raw", "smoke_corpus.txt")
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("\n".join(TINY_CORPUS) + "\n")
    print(f"[1] tiny corpus written: {corpus_path} ({len(TINY_CORPUS)} lines)")

    # 2. train a tiny tokenizer from scratch
    os.makedirs(os.path.join("data", "tokenizer"), exist_ok=True)
    os.environ.setdefault("SPM_TRAIN_CORPUS", corpus_path)
    import subprocess
    subprocess.run(
        [sys.executable, "scripts/train_tokenizer.py",
         "--input-dir", os.path.dirname(corpus_path),
         "--output-prefix", os.path.join("data", "tokenizer", "smoke_sp"),
         "--vocab-size", "512",
         "--model-type", "bpe"],
        check=True,
    )
    tok = SentencePieceTokenizer(
        os.path.join("data", "tokenizer", "smoke_sp.model"),
        os.path.join("data", "tokenizer", "smoke_sp.meta.json"),
    )
    print(f"[2] tiny tokenizer trained: vocab_size={tok.vocab_size}")

    # 3. encode / decode round trip
    sample = "the quick brown fox jumps over the lazy dog"
    enc = tok.encode(sample)
    dec = tok.decode(enc)
    assert sample in dec, f"decode mismatch: {dec!r} != {sample!r}"
    print(f"[3] encode/decode OK: {enc} -> {dec!r}")

    # 4. tokenize corpus to binary
    bin_path = os.path.join("data", "processed", "smoke.bin")
    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
    n_tokens = encode_text_to_bin(corpus_path, bin_path, tok, add_bos=True, add_eos=True)
    print(f"[4] corpus tokenized: {n_tokens} tokens -> {bin_path}")

    # 5. tiny model
    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=4,
        ffn_dim=256,
        block_size=64,
        norm="rms",
        activation="gelu",
        pos_encoding="rope",
        tie_weights=True,
        dropout=0.0,
        init_std=0.02,
        bos_id=tok.bos_id,
        eos_id=tok.eos_id,
        pad_id=tok.pad_id,
        unk_id=tok.unk_id,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(cfg).to(device)
    counts = model.count_parameters()
    print(f"[5] tiny model on {device}: {counts['total']:,} params")

    # 6. forward + causal loss
    ds = BinaryDataset(bin_path, cfg.block_size, batch_size=4, seed=0)
    x, y = ds.get_batch(device)
    logits, loss = model(x, y, ignore_index=cfg.pad_id)
    assert logits.shape == (4, cfg.block_size, cfg.vocab_size), logits.shape
    assert loss.ndim == 0 and torch.isfinite(loss)
    print(f"[6] forward OK: logits{tuple(logits.shape)} loss={loss.item():.4f}")

    # 7. backward + optimizer step
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    print("[7] backward + optimizer step OK")

    # 8. generation: greedy, temperature, top-k
    model.eval()
    prompt = tok.encode("the quick brown fox")
    with torch.no_grad():
        from src.generate import generate, sample_next

        ids_greedy = generate(model, tok, prompt, max_new_tokens=16,
                              temperature=0.0, top_k=1)
        ids_sampled = generate(model, tok, prompt, max_new_tokens=16,
                               temperature=0.8, top_k=10, seed=42)
    print(f"[8] greedy:  {tok.decode(ids_greedy)!r}")
    print(f"    sampled: {tok.decode(ids_sampled)!r}")

    # 9. checkpoint save + reload
    ckpt_dir = tempfile.mkdtemp(prefix="smoke_ckpt_")
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
    torch.save({"config": cfg.to_dict(), "model_state": model.state_dict()}, ckpt_path)
    cfg2 = ModelConfig.from_dict(torch.load(ckpt_path, weights_only=False)["config"])
    model2 = GPTModel(cfg2)
    model2.load_state_dict(torch.load(ckpt_path, weights_only=False)["model_state"])
    model2.to(device).eval()

    # 10. verify reloaded model produces identical outputs
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)
        same = bool(torch.equal(out1[0], out2[0]))
    assert same, "reloaded model outputs differ!"
    print(f"[9] checkpoint saved to {ckpt_path}")
    print(f"[10] reload verified: outputs identical = {same}")

    print("\n=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()