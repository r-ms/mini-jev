"""Smoke 1: letter tables, chat-template tail, padding side. Prints for the model actually loaded."""
import argparse, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer
from minijev.letters import build_tables, N_LETTERS

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--revision", default=None)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
bare, space = build_tables(tok)
print(f"model: {a.model} rev={a.revision}")
print(f"bare A..Z single-token: {N_LETTERS}/{N_LETTERS}; space A..Z: {N_LETTERS}/{N_LETTERS}")
print(f"bare A..P ids: {bare[:16]}")
print(f"space A..P ids: {space[:16]}")
for s in ("true", "false", " true", " false"):
    print(f"  {s!r}: {tok.encode(s, add_special_tokens=False)}")
msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
for et in (True, False):
    r = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=et)
    print(f"template tail (enable_thinking={et}):", repr(r[-60:]))
# The harness always renders with enable_thinking=False: ignored by the 2507 template,
# REQUIRED on the 0.6B hybrid template (without it the smoke model opens a <think> block
# and the first generated token is not the answer letter).
print("padding_side (default):", tok.padding_side, "| pad:", tok.pad_token_id, "| eos:", tok.eos_token_id)
