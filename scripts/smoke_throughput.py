"""Smoke 6: calibrate prefill tok/s and decode ms/step. Every wall-time number in the report
is a planning placeholder until this runs on the model that produced the results."""
import argparse, json, sys, pathlib, time, subprocess
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from minijev.engine import Engine
from minijev import config as C, data, prompts, schema

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=C.MODEL); ap.add_argument("--revision", default=C.MODEL_REVISION)
ap.add_argument("--batch-b", type=int, default=C.BATCH_B); ap.add_argument("--batch-a", type=int, default=C.BATCH_A)
a = ap.parse_args()

e = Engine(model_id=a.model, revision=a.revision)
rows, m = data.load_sample()
rs = [r for r in rows if not r["oos"]][:64]
ps = []
for r in rs:
    o, gp, ke, pol = schema.options_for(r["text_id"], "intent", 8, r["gold"]["intent"], m)
    ps.append(prompts.render(e.tok, prompts.SYSTEM_B, prompts.user_b(r["text"], "intent", o)))

e.last_hidden(ps[:a.batch_b])          # warm-up, not counted
tok_total = wall_total = 0.0
for i in range(0, len(ps), a.batch_b):
    ch = ps[i:i + a.batch_b]
    h, nreal, npad, w = e.last_hidden(ch)
    tok_total += sum(nreal) + sum(npad)   # padding is paid for too
    wall_total += w
prefill_tps = tok_total / wall_total

gen_p = ps[:a.batch_a]
e.generate(gen_p, None, 4)
t0 = time.perf_counter(); e.generate(gen_p, None, 32); w32 = time.perf_counter() - t0
ms_step = w32 / 32 * 1000

out = {"model": a.model, "revision": a.revision, "batch_b": a.batch_b, "batch_a": a.batch_a,
       "prefill_tokens": int(tok_total), "prefill_wall_sec": round(wall_total, 3),
       "prefill_tok_per_sec": round(prefill_tps, 1),
       "decode_ms_per_step_at_batch_a": round(ms_step, 1),
       "mem_gb": e.mem_gb(), "measured_at": time.time(),
       "run_mode": C.run_mode(a.batch_b, a.batch_a)}
try:
    out["ollama_ps"] = subprocess.run(["ollama","ps"],capture_output=True,text=True,timeout=10).stdout.strip()
except Exception as ex:
    out["ollama_ps"] = str(ex)
path = pathlib.Path(__file__).resolve().parents[1] / "runs" / "calibration.json"
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
