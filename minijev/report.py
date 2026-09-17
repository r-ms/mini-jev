"""Report: join-key check first, denominators first, mechanism counters before any accuracy."""
import collections, json, math
from pathlib import Path

from . import config as C
from .records import read_jsonl
from .scoring import (accuracy, balanced_accuracy, cluster_bootstrap_delta, correct,
                      join_key_check, minimal_detectable_delta, paired, unit_rows)


def load_run(run_dir):
    run_dir = Path(run_dir)
    recs = {}
    for p in sorted(run_dir.glob("*.jsonl")):
        # cost.jsonl measures COST against input length: its records have no join key and no
        # accuracy and must not be mixed with arm records. It is read as its own section.
        if p.stem == "cost":
            continue
        recs[p.stem] = read_jsonl(p)
    return recs


def _p(x, n):
    return f"{x}/{n} = {x/n:.3f}" if n else f"{x}/0 = n/a"


def mechanism_tables(recs, out):
    out.append("## Mechanism counters (read BEFORE any accuracy)")
    # M5 counts the LETTER mechanism: did the unconstrained argmax land on a candidate letter.
    # B1score never reads letters (its argmax_class = label_score); applying this check to it would
    # print 'mechanism did not fire' about an arm the question does not apply to.
    for arm in ("B1", "B1perm"):
        rs = recs.get(arm)
        if not rs:
            continue
        n = len(rs)
        hist = collections.Counter(r["argmax_class"] for r in rs)
        out.append(f"### {arm}: argmax class, denominator {n}")
        for cls in ("bare_candidate", "space_candidate", "letter_beyond_k", "other"):
            out.append(f"  {cls:18s} {_p(hist[cls], n)}")
        others = collections.Counter(r["argmax_token"] for r in rs if r["argmax_class"] == "other")
        if others:
            out.append(f"  other tokens: {others.most_common(8)}")
        out.append(f"  M5 letter emission pooled: {_p(hist['bare_candidate'], n)}"
                   f"  (P3 needs >= {C.P3_M5_POOLED})")
        cells = collections.defaultdict(lambda: [0, 0])
        for r in rs:
            c = cells[(r["k"], len(r["fields_order"]) if "fields_order" in r else 1)]
            c[1] += 1
            c[0] += int(r["argmax_class"] == "bare_candidate")
        out.append("  M5 per cell (k, fields):")
        for key in sorted(cells):
            good, tot = cells[key]
            flag = "  <-- MECHANISM DID NOT FIRE" if tot and good / tot < C.STOP_M5_CELL else ""
            out.append(f"    k={key[0]:<3} f={key[1]}  {_p(good, tot)}{flag}")
        ties = sum(1 for r in rs if r["tie"])
        out.append(f"  fp32 ties: {_p(ties, n)} (STOP above {C.STOP_FP32_TIE_RATE})")

    for arm in ("A1joint", "A1split", "A1letter", "A1label", "A1prob", "A0", "B1free"):
        rs = recs.get(arm)
        if not rs:
            continue
        n = len(rs)
        states = collections.Counter(r["state"] for r in rs)
        bite = sum(r.get("mask_bite_steps", 0) for r in rs)
        steps = sum(r.get("grammar_steps", 0) for r in rs)
        lengths = sum(1 for r in rs if r["finish_reason"] == "length")
        gen = sum(r["generated_tokens"] for r in rs)
        out.append(f"### {arm}: denominator {n}")
        out.append(f"  states: {dict(sorted(states.items()))}")
        out.append(f"  finish=length: {_p(lengths, n)} (STOP above {C.STOP_LENGTH_RATE})")
        out.append(f"  generated tokens total {gen}, mean {gen/n:.1f}" if n else "")
        if arm != "A0":
            out.append(f"  grammar row-steps {steps}; mask BIT the argmax on {bite} of them"
                       f" ({bite/steps:.3f})" if steps else "  grammar row-steps 0")
            out.append("   ('the processor ran' is green by construction; this counter is not)")
            viol = sum(1 for r in rs if r["state"] == "out_of_enum")
            out.append(f"  enum violations: {_p(viol, n)}  (S2 STOP: must be 0)")


def accuracy_tables(recs, out):
    rows_by_arm = {arm: unit_rows(rs) for arm, rs in recs.items()}
    out.append("")
    out.append("## Accuracy (denominators first; a non-answer counts WRONG, never dropped)")
    for arm in sorted(rows_by_arm):
        rows = [r for r in rows_by_arm[arm] if r["condition"].startswith(("main", "easy"))]
        if not rows:
            continue
        cells = collections.defaultdict(list)
        for r in rows:
            cells[(r["field"], r["k"])].append(r)
        out.append(f"### {arm}")
        for keyc in sorted(cells, key=lambda t: (t[0], t[1])):
            c, n, acc = accuracy(cells[keyc])
            extra = ""
            if keyc[0] in C.BOOL_FIELDS:
                extra = f"  balanced {balanced_accuracy(cells[keyc]):.3f}"
            out.append(f"  field={keyc[0]:<28} k={keyc[1]:<3} {_p(c, n)}{extra}")
    return rows_by_arm


def delta_tables(rows_by_arm, out):
    if "B1" not in rows_by_arm or "A1joint" not in rows_by_arm:
        return
    out.append("")
    out.append("## P1: paired delta on the PRIMARY unit (field=intent only)")
    a = [r for r in rows_by_arm["A1joint"] if r["field"] == "intent" and r["condition"].startswith("main")]
    b = [r for r in rows_by_arm["B1"] if r["field"] == "intent" and r["condition"].startswith("main")]
    pairs, b_only, a_only = paired(a, b)
    if not pairs:
        out.append("  no common pairs")
        return
    point, (lo, hi) = cluster_bootstrap_delta(pairs)
    mdd = minimal_detectable_delta(pairs)
    out.append(f"  pairs {len(pairs)}; discordant b(B1 right, A1 wrong)={b_only}, c(A1 right, B1 wrong)={a_only}")
    out.append(f"  delta = {point*100:+.2f} pp, 95% cluster bootstrap [{lo*100:+.2f}, {hi*100:+.2f}] pp")
    out.append(f"  minimal detectable half-width from observed discordance: +-{mdd*100:.2f} pp"
               f"  (P1 bound {C.P1_BOUND_POOLED_PP} pp)")
    if mdd * 100 > abs(C.P1_BOUND_POOLED_PP):
        out.append("  NOTE: the design cannot resolve the pre-registered bound on this sample.")


def ladder_table(rows_by_arm, out):
    order = ["A1joint", "A1split", "A1letter", "B1"]
    have = [a for a in order if a in rows_by_arm]
    if len(have) < 2:
        return
    out.append("")
    out.append("## M13: the ladder, each step isolating one difference (field=intent)")
    prev = None
    for arm in have:
        rows = [r for r in rows_by_arm[arm] if r["field"] == "intent" and r["condition"].startswith("main")]
        c, n, acc = accuracy(rows)
        line = f"  {arm:<10} {_p(c, n)}"
        if prev is not None:
            pairs, b_only, a_only = paired(prev[1], rows)
            if pairs:
                pt, (lo, hi) = cluster_bootstrap_delta(pairs)
                line += f"   delta vs {prev[0]}: {pt*100:+.2f} pp [{lo*100:+.2f},{hi*100:+.2f}]"
        out.append(line)
        prev = (arm, rows)


def m14_agreement(recs, out):
    if "A1letter" not in recs or "B1" not in recs:
        return
    bl = {(r["condition"], r["text_id"], r["field"]): r for r in recs["B1"]}
    agree = tot = 0
    disagreements = []
    for r in recs["A1letter"]:
        f = r["fields_order"][0]
        k = (r["condition"], r["text_id"], f)
        if k not in bl:
            continue
        tot += 1
        p_letter = (r.get("per_field_pred") or {}).get(f)
        if p_letter == bl[k]["pred"]:
            agree += 1
        else:
            disagreements.append((k, p_letter, bl[k]["pred"]))
    out.append("")
    out.append("## M14: greedy-under-letter-grammar vs the logit read (harness control)")
    out.append(f"  agreement {_p(agree, tot)} (P3 needs >= {C.P3_M14_AGREEMENT})")
    out.append("  LIMIT: M14 cannot catch both arms being wrong the same way (shared prompt and")
    out.append("  position). The text-answer control is the operand outside that mechanism.")
    for d in disagreements[:10]:
        out.append(f"    disagree: {d[0]} letter={d[1]!r} logit={d[2]!r}")


def controls(recs, out):
    out.append("")
    out.append("## Free-text control: does the UNCONSTRAINED model answer with the letter we read?")
    # Pair by prompt_sha and nothing else. Pairing by (text_id, field, k) once matched every
    # B1free record to a B1 record from the EASY subset (different distractors), and the
    # "agreement" that came out was simply B1free's accuracy against gold (189/200). A control
    # with no twin is NOT MEASURED, and the report must say so instead of printing a number.
    fr = recs.get("B1free") or []
    b1_by_sha = {r["prompt_sha"]: r for r in recs.get("B1", [])}
    twins = [(r, b1_by_sha[r["prompt_sha"]]) for r in fr if r["prompt_sha"] in b1_by_sha]
    if fr and not twins:
        out.append(f"  NOT MEASURED: {len(fr)} B1free records, 0 with a B1 twin by prompt_sha.")
    elif twins:
        ag = sum(1 for f_, b_ in twins if (f_.get("per_field_pred") or {}).get(f_["fields_order"][0]) == b_["pred"])
        out.append(f"  twins by prompt_sha {len(twins)}/{len(fr)}; agreement {_p(ag, len(twins))} (P3 needs >= 0.98)")
    out.append("")
    out.append("## Controls")
    for arm, rs in sorted(recs.items()):
        oos = [r for r in rs if str(r.get("condition", "")).startswith("oos:")]
        if oos:
            for cond in ("oos:oos", "oos:in_scope"):
                sel = [r for r in oos if r["condition"] == cond]
                if not sel:
                    continue
                if arm in ("B1", "B1perm", "B1cache"):
                    nota = sum(1 for r in sel if r["pred"] == C.NOTA)
                else:
                    nota = sum(1 for r in sel
                               if (r.get("per_field_pred") or {}).get("intent") == C.NOTA)
                out.append(f"  {arm} {cond}: chose '{C.NOTA}' {_p(nota, len(sel))}")
    out.append("  (both rows are required: on oos alone 'always none of the above' scores 100%)")

    for arm in ("B1",):
        rs = [r for r in recs.get(arm, []) if r["condition"] == "ctl:nogold"]
        if rs:
            bc = [r for r in rs if r["argmax_class"] == "bare_candidate"]
            import statistics
            mx = statistics.median([max(r["p_cand"]) for r in bc]) if bc else float("nan")
            gp = statistics.median([r["gap"] for r in bc]) if bc else float("nan")
            main = [r for r in recs.get(arm, [])
                    if r["condition"].startswith("main") and r["argmax_class"] == "bare_candidate"]
            mxm = statistics.median([max(r["p_cand"]) for r in main]) if main else float("nan")
            gpm = statistics.median([r["gap"] for r in main]) if main else float("nan")
            out.append(f"  {arm} no-gold: median max(p) {mx:.3f} vs main {mxm:.3f}; "
                       f"median gap {gp:.2f} vs main {gpm:.2f}")
            out.append("   (if no-gold is as peaked as main, the probabilities carry nothing)")


def position_prior(recs, out):
    rs = recs.get("B1perm") or []
    if not rs:
        return
    out.append("")
    out.append("## M8: position prior, estimated on REAL texts from the permutation arm")
    by_k = collections.defaultdict(lambda: collections.Counter())
    tot = collections.Counter()
    for r in rs:
        by_k[r["k_eff"]][r["pred_pos"]] += 1
        tot[r["k_eff"]] += 1
    for k in sorted(by_k):
        dist = {p: round(by_k[k][p] / tot[k], 3) for p in range(k)}
        out.append(f"  k={k} (n={tot[k]}): chosen-position share {dist}  (uniform = {1/k:.3f})")


def build(run_dirs):
    out = []
    all_recs = {}
    for d in run_dirs:
        recs = load_run(d)
        for arm, rs in recs.items():
            all_recs.setdefault(arm, []).extend(rs)
        out.append(f"# run {Path(d).name}: " + ", ".join(f"{a}={len(r)}" for a, r in sorted(recs.items())))
    flat = [r for rs in all_recs.values() for r in rs]
    n, u = join_key_check(flat)
    out.append("")
    out.append(f"## Join key: {n} keys, {u} unique" + ("" if n == u else "   <-- STOP S1: duplicates"))
    if n != u:
        return "\n".join(out)
    mechanism_tables(all_recs, out)
    rows_by_arm = accuracy_tables(all_recs, out)
    m14_agreement(all_recs, out)
    ladder_table(rows_by_arm, out)
    delta_tables(rows_by_arm, out)
    position_prior(all_recs, out)
    controls(all_recs, out)
    return "\n".join(out)
