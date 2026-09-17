"""Execute one arm over one subset. Resumable by key; never rewrites an existing run file."""
import argparse, json, subprocess, sys, time, pathlib, importlib.metadata as md
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch

from minijev import config as C, data, manifest as MF, prompts, runner, schema
from minijev.engine import Engine
from minijev.letters import (build_tables, candidate_logits_fp32, choice_confidence,
                             classify_argmax, score)
from minijev.records import Writer, code_rev, key as make_key
from minijev import oracles

ARMS = ("B1", "B1cache", "B1perm", "A1joint", "A1split", "A1letter", "A1label", "A1prob",
        "B1free", "A0")

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=ARMS)
ap.add_argument("--subset", required=True)
ap.add_argument("--run-id", required=True)
ap.add_argument("--model", default=C.MODEL)
ap.add_argument("--revision", default=None)
ap.add_argument("--engine", default="model", choices=("model", "oracle-gold", "oracle-next"))
ap.add_argument("--n-texts", type=int, default=None)
ap.add_argument("--batch-b", type=int, default=C.BATCH_B)
ap.add_argument("--batch-a", type=int, default=C.BATCH_A)
ap.add_argument("--attn", default=C.ATTN_IMPLEMENTATION)
ap.add_argument("--option-format", default="eq")
ap.add_argument("--batch-order", default="primary", choices=("primary", "secondary"))
ap.add_argument("--max-units", type=int, default=None)
# The generation cap is part of the run mode (written into run_mode). The TypeSafe form (A1prob) at
# k=16 closes only 3-4 of 16 values within 96 tokens (~25 tokens per key with whitespace allowed):
# 2104 / 2104 records of the first b2 attempt hit the cap; ~640 is needed.
ap.add_argument("--max-new-tokens", type=int, default=None)
ap.add_argument("--stride", type=int, default=None,
                help="keep every Nth unit: units are built condition by condition, so a stride\n                      spans every k and field-condition evenly, while --max-units would take\n                      the first N and land entirely in the earliest cells")
ap.add_argument("--unique-prompts", action="store_true",
                help="keep one record per (prompt_sha, field): the f axis re-renders the\n                      SAME prompt for a field, so 60%% of a per-field arm is re-computation")
a = ap.parse_args()

rev = a.revision if a.revision is not None else (C.MODEL_REVISION if a.model == C.MODEL else None)
ROOT = pathlib.Path(__file__).resolve().parents[1]
run_dir = ROOT / "runs" / a.run_id

rows, dmanifest = data.load_sample()
if a.engine == "model":
    eng = Engine(model_id=a.model, revision=rev, attn=a.attn)
else:
    from transformers import AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(a.model, revision=rev); tok.padding_side = "left"
    cfg = AutoConfig.from_pretrained(a.model, revision=rev)
    eng = (oracles.OracleGold if a.engine == "oracle-gold" else oracles.OracleNext)(tok, vocab=cfg.vocab_size)
bare, space = build_tables(eng.tok)

rv, dirty = code_rev()
certified = {
    "protocol_id": C.PROTOCOL_ID, "model_id": a.model, "model_revision": rev,
    "torch": torch.__version__, "transformers": md.version("transformers"),
    "xgrammar": md.version("xgrammar"), "device": getattr(eng, "device", "cpu"),
    "dtype": eng.dtype_name, "attn": a.attn, "batch_b": a.batch_b, "batch_a": a.batch_a,
    "seed": C.SEED, "prompt_template_sha": prompts.template_sha(),
    "template_shas": prompts.template_shas(),
    "data_manifest_sha": data.manifest_sha(), "code_rev": None,
    "code_rev_now": rv,   # recorded, not certified: see manifest.CERTIFIED
}
MF.refuse_resume_on_mismatch(run_dir, certified)

try:
    ollama = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10).stdout.strip()
except Exception as e:
    ollama = f"unavailable: {e}"

units = runner.build_units(a.subset, rows, dmanifest, n_texts=a.n_texts)
if a.stride:
    before = len(units)
    units = units[::a.stride]
    print(json.dumps({"stride": a.stride, "units_before": before, "units_kept": len(units)}))
if a.max_units:
    units = units[:a.max_units]
run_mode = C.run_mode(a.batch_b, a.batch_a, a.attn,
                      extra=f"engine={a.engine},order={a.batch_order},fmt={a.option_format}"
                            + (f",maxnew={a.max_new_tokens}" if a.max_new_tokens else ""))

writer = Writer(run_dir / f"{a.arm}.jsonl")
todo_note = {"units": len(units), "already": len(writer.seen), "dropped_truncated_tail": writer.dropped_tail}
print(json.dumps({"arm": a.arm, "subset": a.subset, **todo_note}, ensure_ascii=False))

base = {"run_id": a.run_id, "arm": a.arm, "subset": a.subset, "model": a.model,
        "model_revision": rev, "run_mode": run_mode, "seed": C.SEED,
        "prompt_template_sha": prompts.template_sha(), "code_rev": rv, "code_dirty": dirty,
        "engine": a.engine, "batch_order": a.batch_order, "option_format": a.option_format}


def order_items(items):
    srt, lens = runner.sort_units(items, eng)
    if a.batch_order == "secondary":
        import random
        rng = random.Random(C.SEED + 1)
        idx = list(range(len(srt))); rng.shuffle(idx)
        srt = [srt[i] for i in idx]; lens = [lens[i] for i in idx]
    return srt, lens


t_start = time.time()
n_written = 0

if a.arm == "B1cache":
    # ONE prefill of the text per request, the cache is broadcast to the fields. The unit of work is
    # the TEXT (all its fields together), not the field: that is what a real request looks like, and
    # that is the honest unit for the cost.
    from minijev.prompts import split_b_prompt
    groups = {}
    for u in units:
        groups.setdefault((u["condition"], u["row"]["text_id"]), u)
    todo = [u for u in groups.values()
            if not all(writer.has(make_key(u["condition"], u["row"]["text_id"], f, a.arm)) for f in u["fields"])]
    print(f"  texts to process: {len(todo)} of {len(groups)}")
    for n_i, u in enumerate(todo):
        fulls = [prompts.render(eng.tok, prompts.SYSTEM_B, prompts.user_b(u["row"]["text"], f, u["options"][f], a.option_format))
                 for f in u["fields"]]
        pre, sufs = None, []
        for full in fulls:
            p_, s_ = split_b_prompt(full)
            if pre is None: pre = p_
            assert p_ == pre, "prefix differs across fields of one text"
            sufs.append(s_)
        h, P, S, wall = eng.broadcast_last_hidden(pre, sufs)
        full = eng.full_logits_from_hidden(h); amax = full.argmax(-1).tolist()
        logZ = torch.logsumexp(full.float(), dim=-1)
        for i, (f, p_full) in enumerate(zip(u["fields"], fulls)):
            opts = u["options"][f]; k = len(opts); hh = h[i]
            cb = candidate_logits_fp32(hh.unsqueeze(0), eng.embed_weight, bare[:k])[0]
            cs = candidate_logits_fp32(hh.unsqueeze(0), eng.embed_weight, space[:k])[0]
            sc = score(cb, cs)
            fb = full[i][bare[:k]].float(); fs = full[i][space[:k]].float(); lz = float(logZ[i])
            cm = min(float(torch.exp(torch.logsumexp(fb, 0) - lz)), 1.0)
            cms = min(float(torch.exp(torch.logsumexp(torch.cat([fb, fs]), 0) - lz)), 1.0)
            rec = {**base, "key": make_key(u["condition"], u["row"]["text_id"], f, a.arm),
                   "condition": u["condition"], "text_id": u["row"]["text_id"], "field": f,
                   "fields_order": u["fields"], "k": u["k"], "k_eff": k, "options": opts,
                   "gold": u["row"]["gold"].get(f), "gold_pos": u["gold_pos"][f], "policy": u["policy"][f],
                   "prompt_sha": prompts.prompt_sha(p_full), "schema_sha": schema.schema_sha({f: opts}),
                   "sent_at": time.time(), "batch_id": n_i, "batch_size": len(u["fields"]),
                   "batch_wall_sec": round(wall, 4), "elapsed_sec": round(wall / len(u["fields"]), 5),
                   "prompt_tokens": P + S[i], "prefix_tokens_shared": P, "suffix_tokens": S[i],
                   "text_prefill_tokens_total": P + sum(S), "pad_tokens": max(S) - S[i],
                   "argmax_id": amax[i], "argmax_token": eng.tok.decode([amax[i]]),
                   "argmax_class": classify_argmax(amax[i], bare, space, k),
                   "cand_logits_bare": [round(float(x), 5) for x in cb],
                   "cand_logits_space": [round(float(x), 5) for x in cs],
                   "logZ": round(lz, 5), "candidate_mass": round(cm, 8), "candidate_mass_with_space": round(cms, 8),
                   "pred_pos_native_bf16": int(torch.argmax(fb).item()),
                   "native_matches_fp32": int(torch.argmax(fb).item()) == sc["pred_pos"],
                   "pred": opts[sc["pred_pos"]], "forward_passes": 2, **sc}
            writer.write(rec); n_written += 1
        if n_i == 0:
            print("  mem_gb:", eng.mem_gb(), "| first text wall %.3fs" % wall)

elif a.arm in ("B1", "B1perm"):
    items = runner.b_prompts(eng, units, a.option_format)
    items = [(i, f, p) for (i, f, p) in items
             if not writer.has(make_key(units[i]["condition"], units[i]["row"]["text_id"], f, a.arm))]
    if a.unique_prompts:
        seen_p, dedup = set(), []
        for it in items:
            kk = (prompts.prompt_sha(it[-1]), it[1])
            if kk in seen_p:
                continue
            seen_p.add(kk); dedup.append(it)
        print(f"  --unique-prompts: {len(items)} -> {len(dedup)} "
              f"({100*(1-len(dedup)/max(len(items),1)):.1f}% was re-computation)")
        items = dedup
    items, _ = order_items(items)
    for bi in range(0, len(items), a.batch_b):
        chunk = items[bi:bi + a.batch_b]
        if a.engine != "model":
            eng.set_targets([oracles.target_for(a.engine, units[i]["gold_pos"][f],
                                                len(units[i]["options"][f])) for i, f, _ in chunk])
        h, n_real, n_pad, wall = eng.last_hidden([p for *_, p in chunk])
        full = eng.full_logits_from_hidden(h)
        amax = full.argmax(-1).tolist()
        # CANDIDATE MASS. softmax over the k candidates answers "which option, GIVEN the answer
        # is one of them" -- it says nothing about how much probability the model put on the
        # candidates at all. A set can hold the top token and still carry 16% of the mass, and
        # the normalized numbers would look confident anyway. logZ makes that measurable.
        logZ = torch.logsumexp(full.float(), dim=-1)
        for (i, f, p), hh, am, nr, npd in zip(chunk, h, amax, n_real, n_pad):
            u = units[i]; opts = u["options"][f]; k = len(opts)
            cb = candidate_logits_fp32(hh.unsqueeze(0), eng.embed_weight, bare[:k])[0]
            cs = candidate_logits_fp32(hh.unsqueeze(0), eng.embed_weight, space[:k])[0]
            s = score(cb, cs)
            row_i = chunk.index((i, f, p))
            lz = float(logZ[row_i])
            # Numerator and denominator MUST come from one distribution. Taking the numerator
            # from the fp32 recompute and the denominator from the model's own bf16 head gave
            # candidate_mass = 1.0192 -- a probability mass above 1, which is how the mismatch
            # announced itself. The mass is therefore read entirely off the model's own head;
            # the fp32 recompute stays where it is needed, deciding the WINNER among near-ties.
            fb = full[row_i][bare[:k]].float()
            fs = full[row_i][space[:k]].float()
            cand_mass = float(torch.exp(torch.logsumexp(fb, 0) - lz))
            cand_mass_sp = float(torch.exp(torch.logsumexp(torch.cat([fb, fs]), 0) - lz))
            # The tolerance follows the ARITHMETIC, not a wish: logsumexp over 151936 terms in float32
            # carries a relative error of order 1e-5, and 1e-6 does not survive it. Measured on a live
            # run: a mass of 1.0000038 killed the arm at record 607. The check stays: it catches MIXED
            # distributions (that case was 1.0192, four orders of magnitude larger).
            assert cand_mass <= 1.0 + 1e-4, f"candidate mass {cand_mass} > 1: mixed distributions again"
            cand_mass = min(cand_mass, 1.0)
            cand_mass_sp = min(cand_mass_sp, 1.0)
            # The scorer reads fp32 recomputed candidates; the generate arms sample from the
            # model's own bf16 head. How often do the two disagree on the winner?
            argmax_native = int(torch.argmax(fb).item())
            rec = {**base, "key": make_key(u["condition"], u["row"]["text_id"], f, a.arm),
                   "condition": u["condition"], "text_id": u["row"]["text_id"], "field": f,
                   "fields_order": u["fields"], "k": u["k"], "k_eff": len(opts),
                   "options": opts, "gold": u["row"]["gold"].get(f), "gold_pos": u["gold_pos"][f],
                   "policy": u["policy"][f], "prompt_sha": prompts.prompt_sha(p),
                   "schema_sha": schema.schema_sha({f: opts}), "sent_at": time.time(),
                   "batch_id": bi // a.batch_b, "batch_size": len(chunk),
                   "batch_wall_sec": round(wall, 4), "elapsed_sec": round(wall / len(chunk), 5),
                   "prompt_tokens": nr, "pad_tokens": npd,
                   "argmax_id": am, "argmax_token": eng.tok.decode([am]),
                   "argmax_class": classify_argmax(am, bare, space, k),
                   "cand_logits_bare": [round(float(x), 5) for x in cb],
                   "cand_logits_space": [round(float(x), 5) for x in cs],
                   "logZ": round(lz, 5), "candidate_mass": round(cand_mass, 8),
                   "candidate_mass_with_space": round(cand_mass_sp, 8),
                   "pred_pos_native_bf16": argmax_native,
                   "native_matches_fp32": argmax_native == s["pred_pos"],
                   "pred": opts[s["pred_pos"]], "forward_passes": 1,
                   "shift": u.get("shift"), **s}
            writer.write(rec); n_written += 1
        if bi == 0:
            print("  mem_gb:", eng.mem_gb(), "| first batch wall %.2fs" % wall)

else:
    from minijev.grammar import GrammarCache
    # B1free is the UNCONSTRAINED control: it must not get a grammar, or it stops being
    # the operand outside the constraint.
    NO_GRAMMAR = ("A0", "B1free")
    gc = GrammarCache(eng.tok, eng.cfg.vocab_size) if a.arm not in NO_GRAMMAR else None
    items = []
    for i, u in enumerate(units):
        if a.arm == "A1split":
            for f in u["fields"]:
                items.append((i, (f,), None))
        elif a.arm in ("A1letter", "A1label", "A1prob", "B1free"):
            for f in u["fields"]:
                items.append((i, (f,), {"A1letter": "letter", "A1label": "label",
                                        "A1prob": "prob", "B1free": "free"}[a.arm]))
        else:
            items.append((i, tuple(u["fields"]), None))
    prepared = []
    for i, fields, mode in items:
        u = units[i]
        fkey = fields[0] if len(fields) == 1 else "_all"
        if a.arm == "A1joint" or a.arm == "A0":
            fkey = "_all"
        if writer.has(make_key(u["condition"], u["row"]["text_id"], fkey, a.arm)):
            continue
        if mode in ("letter", "label", "free", "prob"):
            f = fields[0]
            sysmsg = {"label": prompts.SYSTEM_B_LABEL,
                      "prob": prompts.SYSTEM_B_PROB}.get(mode, prompts.SYSTEM_B)
            p = prompts.render(eng.tok, sysmsg,
                               prompts.user_b(u["row"]["text"], f, u["options"][f], a.option_format))
            prepared.append((i, fields, mode, fkey, None, p))
        else:
            p, fo = runner.a_prompt(eng, u, list(fields))
            prepared.append((i, fields, "json", fkey, fo, p))
    if a.unique_prompts:
        seen_p, dedup = set(), []
        for it in prepared:
            kk = (prompts.prompt_sha(it[-1]), it[3])
            if kk in seen_p:
                continue
            seen_p.add(kk); dedup.append(it)
        print(f"  --unique-prompts: {len(prepared)} -> {len(dedup)} "
              f"({100*(1-len(dedup)/max(len(prepared),1)):.1f}% was re-computation)")
        prepared = dedup
    prepared, _ = order_items(prepared)
    for bi in range(0, len(prepared), a.batch_a):
        chunk = prepared[bi:bi + a.batch_a]
        proc = None
        if a.arm not in NO_GRAMMAR:
            compiled = []
            for i, fields, mode, fkey, fo, p in chunk:
                u = units[i]
                if mode == "letter":
                    compiled.append(gc.letters(len(u["options"][fields[0]])))
                elif mode == "label":
                    compiled.append(gc.labels(u["options"][fields[0]]))
                elif mode == "prob":
                    compiled.append(gc.probs(u["options"][fields[0]]))
                else:
                    sc = schema.json_schema(fo)
                    compiled.append(gc.json(schema.schema_sha(sc), sc))
            proc = [gc.processor(compiled)]
        mnt = {"A1letter": 4, "B1free": C.MAX_NEW_TOKENS_TEXT_CONTROL,
               "A1label": 24}.get(a.arm, C.MAX_NEW_TOKENS_A)
        if a.max_new_tokens:
            mnt = a.max_new_tokens
        texts, n_gen, finish, n_real, n_pad, wall = eng.generate([c[-1] for c in chunk], proc, mnt)
        cp = proc[0] if proc else None
        for (i, fields, mode, fkey, fo, p), raw, ng, fin, nr, npd in zip(chunk, texts, n_gen, finish, n_real, n_pad):
            u = units[i]
            extra_prob = {}
            if mode in ("letter", "free"):
                f = fields[0]; opts = u["options"][f]
                tok_txt = raw.strip()
                pos = (ord(tok_txt[0]) - 65) if tok_txt[:1].isalpha() and tok_txt[:1].isupper() else -1
                state = "ok" if 0 <= pos < len(opts) else "out_of_enum"
                pred = {f: opts[pos]} if state == "ok" else {}
                schema_obj = {"ebnf_letters": len(opts)} if mode == "letter" else {"free_text": len(opts)}
            elif mode == "label":
                f = fields[0]; opts = u["options"][f]
                txt = raw.strip()
                state = "ok" if txt in opts else "out_of_enum"
                pred = {f: txt} if state == "ok" else {}
                schema_obj = {"ebnf_labels": list(opts)}
            elif mode == "prob":
                f = fields[0]; opts = u["options"][f]
                try:
                    vals = [float(json.loads(raw)["probabilities"][o]) for o in opts]  # grid strings -> float
                    tot = sum(vals)
                    # TypeSafe's own rescale falls back to UNIFORM on a zero total. Reproduced,
                    # but NAMED: a model that wrote 0 for every option did not answer, and
                    # calling that "ok" turns a refusal into a confident-looking uniform.
                    pr = [v / tot for v in vals] if tot else [1.0 / len(vals)] * len(vals)
                    pos = max(range(len(pr)), key=pr.__getitem__)
                    state = "ok" if tot else "degenerate_probs"
                    pred = {f: opts[pos]} if tot else {}
                    extra_prob = {"p_written": [round(v, 6) for v in vals],
                                  "p_normalized": [round(v, 6) for v in pr],
                                  "prob_sum_error": round(abs(tot - 1.0), 6),
                                  "confidence": round(choice_confidence(pr), 6), "pred_pos": pos}
                except Exception as ex:
                    state = "malformed_json"; pred = {}
                    extra_prob = {"parse_error": type(ex).__name__}
                schema_obj = schema.prob_schema(list(opts))
            else:
                state, pred = runner.parse_a(raw, fo)
                schema_obj = schema.json_schema(fo)
            ws = sum(1 for ch in raw if ch.isspace())
            rec = {**base, "key": make_key(u["condition"], u["row"]["text_id"], fkey, a.arm),
                   "condition": u["condition"], "text_id": u["row"]["text_id"], "field": fkey,
                   "fields_order": list(fields), "k": u["k"],
                   "k_eff": {f: len(u["options"][f]) for f in fields},
                   "options": {f: u["options"][f] for f in fields},
                   "gold": {f: u["row"]["gold"].get(f) for f in fields},
                   "gold_pos": {f: u["gold_pos"][f] for f in fields},
                   "policy": {f: u["policy"][f] for f in fields},
                   "prompt_sha": prompts.prompt_sha(p), "schema_sha": schema.schema_sha(schema_obj),
                   "sent_at": time.time(), "batch_id": bi // a.batch_a, "batch_size": len(chunk),
                   "batch_wall_sec": round(wall, 4), "elapsed_sec": round(wall / len(chunk), 5),
                   "prompt_tokens": nr, "pad_tokens": npd, "raw_response": raw,
                   "generated_tokens": ng, "whitespace_chars": ws, "finish_reason": fin,
                   "state": state, "per_field_pred": pred,
                   "grammar_steps": (cp.steps if cp else 0),
                   "mask_bite_steps": (cp.bite_steps if cp else 0),
                   "allowed_step1": (cp.allowed_step1 if cp else None),
                   "forward_passes": 1 + ng, **extra_prob}
            writer.write(rec); n_written += 1
        if bi == 0:
            print("  mem_gb:", eng.mem_gb(), "| first batch wall %.2fs" % wall)

writer.close()
elapsed = time.time() - t_start
MF.write(run_dir, {"certified": certified, "status": "finished", "arm": a.arm, "subset": a.subset,
                   "run_mode": run_mode, "units": len(units), "records_written": n_written,
                   "wall_sec": round(elapsed, 1), "ollama_ps_at_start": ollama,
                   "finished_at": time.time(),
                   "not_attested": ["throughput is machine- and moment-specific",
                                    "numbers from different run modes are not comparable"]})
print(json.dumps({"written": n_written, "wall_sec": round(elapsed, 1)}, ensure_ascii=False))
