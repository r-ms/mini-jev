"""Cost crossover against INPUT LENGTH. Accuracy is not measured here and must not be read.

The per-field arm pays one prefill per field over the SAME text; the joint arm pays one
prefill plus decode. Which wins is a function of input length, and CLINC utterances are ~15
tokens -- the shortest possible case, and the one most favourable to repeating the prefill.
At 4000 tokens the arithmetic can invert. Batching fields into one call does NOT make them
one forward in the compute sense: it is one batched call, and the work still scales with the
number of sequences. (A shared-representation architecture such as GLiClass is a different
thing, and is not what this harness measures.)
"""
import argparse, json, statistics, sys, time, pathlib, importlib.metadata as md
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch

from minijev import config as C, data, manifest as MF, prompts, schema
from minijev.engine import Engine
from minijev.grammar import GrammarCache

FILLER = ("The customer has been a member since two thousand nineteen and has contacted "
          "support about this account on several previous occasions without resolution. ")

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
ap.add_argument("--model", default=C.MODEL); ap.add_argument("--revision", default=None)
ap.add_argument("--lengths", default="32,128,512,2048,4096")
ap.add_argument("--n-texts", type=int, default=8)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--repeats", type=int, default=3)
# Long inputs do not fit 24 GB at the main-grid batch sizes (2048: joint OOM at A.batch=8, eager
# attention grows as B*S^2); the batch is part of the run mode and is written into every row.
ap.add_argument("--batch-b", type=int, default=None)
ap.add_argument("--batch-a", type=int, default=None)
a = ap.parse_args()

rev = a.revision if a.revision is not None else (C.MODEL_REVISION if a.model == C.MODEL else None)
ROOT = pathlib.Path(__file__).resolve().parents[1]
run_dir = ROOT / "runs" / a.run_id
eng = Engine(model_id=a.model, revision=rev)
gc = GrammarCache(eng.tok, eng.cfg.vocab_size)
rows, m = data.load_sample()
texts = [r for r in rows if not r["oos"]][:a.n_texts]
lengths = [int(x) for x in a.lengths.split(",")]
BATCH_B = a.batch_b or C.BATCH_B
BATCH_A = a.batch_a or C.BATCH_A
RUN_MODE = C.run_mode(batch_b=BATCH_B, batch_a=BATCH_A)


def pad(text, target):
    out = text
    while len(eng.tok(out, add_special_tokens=False)["input_ids"]) < target:
        out += " " + FILLER
    ids = eng.tok(out, add_special_tokens=False)["input_ids"][:target]
    return eng.tok.decode(ids)


out_path = run_dir / "cost.jsonl"
run_dir.mkdir(parents=True, exist_ok=True)
f = out_path.open("a", encoding="utf-8")
print(json.dumps({"lengths": lengths, "n_texts": len(texts), "k": a.k, "repeats": a.repeats}))

for L in lengths:
    padded = [(r, pad(r["text"], L)) for r in texts]
    for fields_n in (1, 2, 3):
        fields = C.FIELD_CONDITIONS[f"f{fields_n}"]
        # per-field arm: one prefill per field, nothing generated
        b_wall, b_tok = [], 0
        for rep in range(a.repeats):
            ps = []
            for r, txt in padded:
                for fl in fields:
                    o, gp, ke, pol = schema.options_for(r["text_id"], fl, a.k, r["gold"][fl], m)
                    ps.append(prompts.render(eng.tok, prompts.SYSTEM_B, prompts.user_b(txt, fl, o)))
            t0 = time.perf_counter()
            for i in range(0, len(ps), BATCH_B):
                _, nreal, npad, _ = eng.last_hidden(ps[i:i + BATCH_B])
                if rep == 0:
                    b_tok += sum(nreal) + sum(npad)
            b_wall.append(time.perf_counter() - t0)
        # broadcast arm: ONE prefill of the text per request, cache broadcast to the fields
        from minijev.prompts import split_b_prompt
        c_wall, c_tok = [], 0
        for rep in range(a.repeats):
            t0 = time.perf_counter()
            for r, txt in padded:
                fulls = [prompts.render(eng.tok, prompts.SYSTEM_B,
                         prompts.user_b(txt, fl, schema.options_for(r["text_id"], fl, a.k, r["gold"][fl], m)[0]))
                         for fl in fields]
                pre = None; sufs = []
                for full in fulls:
                    p_, s_ = split_b_prompt(full)
                    pre = pre or p_; sufs.append(s_)
                _, P, S, _ = eng.broadcast_last_hidden(pre, sufs)
                if rep == 0:
                    c_tok += P + sum(S)
            c_wall.append(time.perf_counter() - t0)
        # joint arm: one prefill per text, decode the JSON
        a_wall, a_tok, a_gen = [], 0, 0
        for rep in range(a.repeats):
            ps, schemas = [], []
            for r, txt in padded:
                fo = {fl: schema.options_for(r["text_id"], fl, a.k, r["gold"][fl], m)[0] for fl in fields}
                ps.append(prompts.render(eng.tok, prompts.SYSTEM_A, prompts.user_a(txt, fo)))
                sc = schema.json_schema(fo); schemas.append(gc.json(schema.schema_sha(sc), sc))
            t0 = time.perf_counter()
            for i in range(0, len(ps), BATCH_A):
                proc = [gc.processor(schemas[i:i + BATCH_A])]
                _, ng, fin, nreal, npad, _ = eng.generate(ps[i:i + BATCH_A], proc, C.MAX_NEW_TOKENS_A)
                if rep == 0:
                    a_tok += sum(nreal) + sum(npad); a_gen += sum(ng)
            a_wall.append(time.perf_counter() - t0)
        rec = {"input_tokens": L, "fields": fields_n, "k": a.k, "n_texts": len(texts),
               "broadcast_wall_median": round(statistics.median(c_wall), 4),
               "broadcast_wall_all": [round(x, 4) for x in c_wall],
               "broadcast_prefill_tokens": c_tok,
               "ratio_wall_broadcast_vs_joint": round(statistics.median(c_wall) / statistics.median(a_wall), 4),
               "ratio_wall_broadcast_vs_perfield": round(statistics.median(c_wall) / statistics.median(b_wall), 4),
               "run_mode": RUN_MODE, "model": a.model, "model_revision": rev,
               "perfield_wall_median": round(statistics.median(b_wall), 4),
               "perfield_wall_all": [round(x, 4) for x in b_wall],
               "perfield_prefill_tokens": b_tok, "perfield_generated_tokens": 0,
               "joint_wall_median": round(statistics.median(a_wall), 4),
               "joint_wall_all": [round(x, 4) for x in a_wall],
               "joint_prefill_tokens": a_tok, "joint_generated_tokens": a_gen,
               "ratio_wall": round(statistics.median(b_wall) / statistics.median(a_wall), 4),
               "ratio_prefill": round(b_tok / max(a_tok, 1), 4),
               "measured_at": time.time(), "note": "COST ONLY -- accuracy is meaningless here"}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        print(f"  L={L:<5} f={fields_n}  per-field {rec['perfield_wall_median']:6.2f}s  "
              f"broadcast {rec['broadcast_wall_median']:6.2f}s  joint {rec['joint_wall_median']:6.2f}s  "
              f"| bc/joint {rec['ratio_wall_broadcast_vs_joint']:.2f}  bc/perfield {rec['ratio_wall_broadcast_vs_perfield']:.2f}  "
              f"| prefill pf/bc/joint {b_tok}/{c_tok}/{a_tok}", flush=True)
f.close()
MF.write(run_dir, {"certified": {"protocol_id": C.PROTOCOL_ID, "model_id": a.model,
                                 "model_revision": rev, "torch": torch.__version__,
                                 "transformers": md.version("transformers"),
                                 "xgrammar": md.version("xgrammar"), "device": eng.device,
                                 "dtype": eng.dtype_name, "attn": C.ATTN_IMPLEMENTATION,
                                 "batch_b": BATCH_B, "batch_a": BATCH_A, "seed": C.SEED,
                                 "data_manifest_sha": data.manifest_sha(),
                                 "template_shas": prompts.template_shas()},
                   "status": "finished", "arm": "cost-sweep", "subset": "lengths",
                   "run_mode": RUN_MODE, "finished_at": time.time(),
                   "not_attested": ["accuracy is not measured in this sweep"]})
