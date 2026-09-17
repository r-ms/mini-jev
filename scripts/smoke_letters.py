"""Smoke 3: does the model actually want to emit a candidate letter at the answer position?

This is a MECHANISM counter, not an accuracy: if the unconstrained argmax is not a candidate
letter, arm B is not reading what we think it reads.
"""
import argparse, sys, pathlib, collections
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from minijev.engine import Engine
from minijev.letters import build_tables, candidate_logits_fp32, classify_argmax, score
from minijev import data, schema, prompts, config as C

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--revision", default=None)
ap.add_argument("--k", type=int, default=4); ap.add_argument("--n", type=int, default=40)
ap.add_argument("--batch", type=int, default=C.BATCH_B)
ap.add_argument("--policy", default="within_domain")
a = ap.parse_args()

e = Engine(model_id=a.model, revision=a.revision)
bare, space = build_tables(e.tok)
rows, m = data.load_sample()
rs = [r for r in rows if not r["oos"]][:a.n]

units = []
for r in rs:
    opts, gp, keff, pol = schema.options_for(r["text_id"], "intent", a.k, r["gold"]["intent"], m, a.policy)
    units.append((r, opts, gp, keff,
                  prompts.render(e.tok, prompts.SYSTEM_B, prompts.user_b(r["text"], "intent", opts))))

hist = collections.Counter(); others = collections.Counter()
pos_hist = collections.Counter(); correct = 0; ties = 0
for i in range(0, len(units), a.batch):
    chunk = units[i:i + a.batch]
    h, _, _, _ = e.last_hidden([u[4] for u in chunk])
    full = e.full_logits_from_hidden(h)
    amax = full.argmax(-1).tolist()
    for (r, opts, gp, keff, _), hh, am in zip(chunk, h, amax):
        k = len(opts)
        cb = candidate_logits_fp32(hh.unsqueeze(0), e.embed_weight, bare[:k])[0]
        cs = candidate_logits_fp32(hh.unsqueeze(0), e.embed_weight, space[:k])[0]
        s = score(cb, cs)
        cls = classify_argmax(am, bare, space, k)
        hist[cls] += 1
        if cls == "other":
            others[repr(e.tok.decode([am]))] += 1
        pos_hist[s["pred_pos"]] += 1
        ties += int(s["tie"])
        correct += int(s["pred_pos"] == gp)

n = len(units)
print(f"model={a.model} k={a.k} n={n} policy={a.policy}")
print("argmax class histogram (denominator %d):" % n)
for cls in ("bare_candidate", "space_candidate", "letter_beyond_k", "other"):
    print(f"  {cls:18s} {hist[cls]}/{n}")
if others:
    print("  other tokens:", others.most_common(6))
print("M5 letter emission = %d/%d = %.3f" % (hist['bare_candidate'], n, hist['bare_candidate']/n))
print("predicted position histogram:", dict(sorted(pos_hist.items())), "| fp32 ties:", ties)
print("accuracy (smoke model, NOT a result): %d/%d = %.3f" % (correct, n, correct/n))
