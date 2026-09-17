"""Guard for the frozen-position defect in generate() (2026-09-17).

Placebo pair on one prompt: generation WITH a caller-supplied position_ids tensor loops on digits
("5555…"), generation through Engine.generate does not. Both halves are checked: the engine path
must produce the phone number verbatim in a short answer, and the placebo path must NOT (otherwise
this check cannot return a negative and proves nothing). Run before any generation arm on a new
host: MINIJEV_DEVICE=cuda python scripts/smoke_positions.py
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "demo"))
from minijev import config as C, prompts, schema as S
from minijev.engine import Engine, position_ids_from_mask
from minijev.grammar import GrammarCache
import server as D

eng = Engine(revision=C.MODEL_REVISION); gc = GrammarCache(eng.tok, eng.cfg.vocab_size)
text = ("Hi, this is Marina from room 3. The patient is bleeding heavily after a tooth extraction and we "
        "cannot stop it, we need a doctor right now. Please call me back on 555-0102, there are 4 more people waiting in the queue.")
fields = D.parse_schema({"type": "object", "properties": {
    "topic": {"type": "string", "enum": ["bleeding", "pain", "appointment", "billing", "equipment", "other"]},
    "callback_phone": {"type": "string", "maxLength": 20, "description": "phone number to call back, exactly as written"},
    "people_waiting": {"type": "integer", "description": "how many people are waiting in the queue"}}})
sc = D.json_schema_for(fields); comp = gc.json(S.schema_sha(sc), sc)
prompt = prompts.render(eng.tok, prompts.SYSTEM_A, D.user_json(text, fields))

texts, n_gen, finish, *_ = eng.generate([prompt], [gc.processor([comp])], 96)
engine_ok = ("555-0102" in texts[0]) and finish[0] == "eos" and n_gen[0] < 60

enc = eng.encode([prompt])
out = eng.model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                         position_ids=position_ids_from_mask(enc["attention_mask"]), do_sample=False, temperature=None,
                         top_p=None, top_k=None, max_new_tokens=96, pad_token_id=eng.tok.pad_token_id,
                         eos_token_id=eng.tok.eos_token_id, logits_processor=[gc.processor([comp])])
placebo = eng.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
placebo_loops = "555-0102" not in placebo
print(f"engine : n={n_gen[0]} finish={finish[0]} {texts[0][:120]!r}")
print(f"placebo: {placebo[:120]!r}")
print("ENGINE_OK" if engine_ok else "ENGINE_FAIL", "| PLACEBO_LOOPS" if placebo_loops else "| PLACEBO_DID_NOT_LOOP (check is degenerate on this host)")
sys.exit(0 if (engine_ok and placebo_loops) else 1)
