"""Arm B1score: score the option STRINGS by teacher-forced likelihood, no generation.

This is the third mechanism, distinct from both arms in run.py:
  A arms  -- the model GENERATES under a grammar
  B1      -- we READ the logits of a symbol (a letter) that stands for an option
  B1score -- we READ the likelihood of the option TEXT itself
It is what LMQL's `distribution` does, and it is the diagnostic that separates "the model
cannot bind a letter to an option" from "the model does not know the answer".

Options differ in length, so BOTH normalizations are recorded and neither is chosen silently:
on one measured example the sum picked a 2-token option, the mean-per-token a 3-token one, and
the gold was the 1-token option. Choosing one quietly would be choosing the answer.
"""
import argparse, json, math, subprocess, sys, time, pathlib, importlib.metadata as md
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch

from minijev import config as C, data, manifest as MF, prompts, runner, schema
from minijev.engine import Engine
from minijev.letters import choice_confidence
from minijev.records import Writer, code_rev, key as make_key

ap = argparse.ArgumentParser()
ap.add_argument("--subset", required=True); ap.add_argument("--run-id", required=True)
ap.add_argument("--model", default=C.MODEL); ap.add_argument("--revision", default=None)
ap.add_argument("--n-texts", type=int, default=None); ap.add_argument("--stride", type=int, default=None)
ap.add_argument("--unique-prompts", action="store_true")
ap.add_argument("--attn", default=C.ATTN_IMPLEMENTATION)
a = ap.parse_args()

rev = a.revision if a.revision is not None else (C.MODEL_REVISION if a.model == C.MODEL else None)
ROOT = pathlib.Path(__file__).resolve().parents[1]
run_dir = ROOT / "runs" / a.run_id
rows, dmanifest = data.load_sample()
eng = Engine(model_id=a.model, revision=rev, attn=a.attn)
rv, dirty = code_rev()
certified = {"protocol_id": C.PROTOCOL_ID, "model_id": a.model, "model_revision": rev,
             "torch": torch.__version__, "transformers": md.version("transformers"),
             "xgrammar": md.version("xgrammar"), "device": eng.device, "dtype": eng.dtype_name,
             "attn": a.attn, "batch_b": C.BATCH_B, "batch_a": C.BATCH_A, "seed": C.SEED,
             "data_manifest_sha": data.manifest_sha(),
             "template_shas": prompts.template_shas()}
MF.refuse_resume_on_mismatch(run_dir, certified)

units = runner.build_units(a.subset, rows, dmanifest, n_texts=a.n_texts)
if a.stride:
    units = units[::a.stride]
writer = Writer(run_dir / "B1score.jsonl")
items = runner.b_prompts(eng, units, "eq")
items = [(i, f, p) for (i, f, p) in items
         if not writer.has(make_key(units[i]["condition"], units[i]["row"]["text_id"], f, "B1score"))]
if a.unique_prompts:
    seen, dedup = set(), []
    for it in items:
        k = (prompts.prompt_sha(it[-1]), it[1])
        if k in seen:
            continue
        seen.add(k); dedup.append(it)
    print(f"  --unique-prompts: {len(items)} -> {len(dedup)}")
    items = dedup
items.sort(key=lambda it: (-len(eng.tok(it[-1], add_special_tokens=False)["input_ids"]), str(it[:-1])))
print(json.dumps({"arm": "B1score", "subset": a.subset, "units": len(items),
                  "already": len(writer.seen), "dropped_truncated_tail": writer.dropped_tail}))

run_mode = C.run_mode(C.BATCH_B, C.BATCH_A, a.attn, extra="mech=label-likelihood")
base = {"run_id": a.run_id, "arm": "B1score", "subset": a.subset, "model": a.model,
        "model_revision": rev, "run_mode": run_mode, "seed": C.SEED,
        "prompt_template_sha": prompts.template_sha(), "code_rev": None, "code_rev_now": rv,
        "code_dirty": dirty, "engine": "model", "batch_order": "primary", "option_format": "eq"}


def soft(v):
    m = max(v); e = [math.exp(x - m) for x in v]; z = sum(e)
    return [x / z for x in e]


t0all = time.time(); n_written = 0
for n, (i, f, p) in enumerate(items):
    u = units[i]; opts = list(u["options"][f])
    t0 = time.perf_counter()
    res = eng.sequence_logprobs(p, opts)
    wall = time.perf_counter() - t0
    sums = [r[0] for r in res]; means = [r[1] for r in res]; ntok = [r[2] for r in res]
    p_sum, p_mean = soft(sums), soft(means)
    pos_s = max(range(len(sums)), key=sums.__getitem__)
    pos_m = max(range(len(means)), key=means.__getitem__)
    srt = sorted(sums, reverse=True)
    writer.write({**base, "key": make_key(u["condition"], u["row"]["text_id"], f, "B1score"),
                  "condition": u["condition"], "text_id": u["row"]["text_id"], "field": f,
                  "fields_order": u["fields"], "k": u["k"], "k_eff": len(opts), "options": opts,
                  "gold": u["row"]["gold"].get(f), "gold_pos": u["gold_pos"][f],
                  "policy": u["policy"][f], "prompt_sha": prompts.prompt_sha(p),
                  "schema_sha": schema.schema_sha({f: opts}), "sent_at": time.time(),
                  "batch_id": n, "batch_size": len(opts), "batch_wall_sec": round(wall, 4),
                  "elapsed_sec": round(wall, 5), "prompt_tokens": None, "pad_tokens": None,
                  "logprob_sum": sums, "logprob_mean_per_token": means, "option_ntokens": ntok,
                  "p_cand": [round(x, 6) for x in p_sum], "pred_pos": pos_s, "pred": opts[pos_s],
                  "p_mean": [round(x, 6) for x in p_mean], "pred_pos_mean": pos_m,
                  "pred_mean": opts[pos_m], "confidence": round(choice_confidence(p_sum), 6),
                  "gap": round(srt[0] - srt[1], 6) if len(srt) > 1 else None,
                  "tie": sums.count(max(sums)) > 1, "argmax_class": "label_score",
                  "argmax_id": None, "argmax_token": None, "forward_passes": len(opts),
                  "generated_tokens": 0})
    n_written += 1
    if n == 0:
        print("  first unit wall %.2fs  mem_gb %s" % (wall, eng.mem_gb()))
writer.close()
MF.write(run_dir, {"certified": certified, "status": "finished", "arm": "B1score",
                   "subset": a.subset, "run_mode": run_mode, "units": len(items),
                   "records_written": n_written, "wall_sec": round(time.time() - t0all, 1),
                   "finished_at": time.time(),
                   "not_attested": ["throughput is machine- and moment-specific"]})
print(json.dumps({"written": n_written, "wall_sec": round(time.time() - t0all, 1)}))
