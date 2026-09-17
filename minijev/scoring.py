"""Scoring, denominators, paired deltas, cluster bootstrap. Aggregations use sorted keys only."""
import collections, math, random

from . import config as C


def join_key_check(records):
    keys = [r["key"] for r in records]
    return len(keys), len(set(keys))


def b_pred(rec):
    return rec["pred"]


def a_preds(rec):
    """-> {field: predicted string} ; empty when the record is not a usable answer."""
    return rec.get("per_field_pred") or {}


def unit_rows(records):
    """Flatten every arm to one row per (condition, text_id, field): the comparison unit."""
    out = []
    for r in records:
        # B1score is also a PER-FIELD arm (it reads, it does not generate), and its records have the
        # B-arm shape: k_eff is a number, not a per-field dict. Treating it as a generating arm would
        # be a parsing error, not a data error.
        if r["arm"] in ("B1", "B1perm", "B1score", "B1cache"):
            out.append({"condition": r["condition"], "text_id": r["text_id"], "field": r["field"],
                        "arm": r["arm"], "k": r["k"], "k_eff": r["k_eff"],
                        "gold": r["gold"], "pred": r["pred"], "gold_pos": r["gold_pos"],
                        "pred_pos": r["pred_pos"], "policy": r["policy"],
                        "argmax_class": r["argmax_class"], "tie": r["tie"], "gap": r["gap"],
                        "state": "ok", "shift": r.get("shift")})
        else:
            preds = a_preds(r)
            for f in r["fields_order"]:
                out.append({"condition": r["condition"], "text_id": r["text_id"], "field": f,
                            "arm": r["arm"], "k": r["k"], "k_eff": r["k_eff"].get(f),
                            "gold": r["gold"].get(f), "pred": preds.get(f),
                            "gold_pos": r["gold_pos"].get(f), "pred_pos": None,
                            "policy": r["policy"].get(f), "argmax_class": None, "tie": None,
                            "gap": None, "state": r["state"], "shift": None})
    return out


def correct(row):
    """Exact string equality. A record that is not a usable answer is WRONG, never dropped."""
    return row["pred"] is not None and row["pred"] == row["gold"]


def accuracy(rows):
    n = len(rows)
    c = sum(1 for r in rows if correct(r))
    return c, n, (c / n if n else float("nan"))


def balanced_accuracy(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["gold"]].append(correct(r))
    if not by:
        return float("nan")
    return sum(sum(v) / len(v) for v in by.values()) / len(by)


def paired(rows_a, rows_b):
    """-> (pairs, b_only_right, a_only_right) keyed by (condition, text_id, field)."""
    ia = {(r["condition"], r["text_id"], r["field"]): r for r in rows_a}
    ib = {(r["condition"], r["text_id"], r["field"]): r for r in rows_b}
    common = sorted(set(ia) & set(ib))
    pairs = [(k, correct(ia[k]), correct(ib[k])) for k in common]
    b_only = sum(1 for _, ca, cb in pairs if cb and not ca)
    a_only = sum(1 for _, ca, cb in pairs if ca and not cb)
    return pairs, b_only, a_only


def cluster_bootstrap_delta(pairs, resamples=None, seed=None):
    """Delta = acc(B) - acc(A), resampled over text_id clusters."""
    resamples = resamples or C.BOOTSTRAP_RESAMPLES
    rng = random.Random(seed or C.BOOTSTRAP_SEED)
    by_text = collections.defaultdict(list)
    for (cond, tid, field), ca, cb in pairs:
        by_text[tid].append((ca, cb))
    texts = sorted(by_text)
    if not texts:
        return float("nan"), (float("nan"), float("nan"))
    na = sum(ca for v in by_text.values() for ca, _ in v)
    nb = sum(cb for v in by_text.values() for _, cb in v)
    n = sum(len(v) for v in by_text.values())
    point = (nb - na) / n
    deltas = []
    for _ in range(resamples):
        pick = [by_text[texts[rng.randrange(len(texts))]] for _ in texts]
        flat = [x for v in pick for x in v]
        if not flat:
            continue
        deltas.append((sum(cb for _, cb in flat) - sum(ca for ca, _ in flat)) / len(flat))
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]
    return point, (lo, hi)


def minimal_detectable_delta(pairs):
    """Half-width the design can resolve, from the observed discordance (McNemar SE)."""
    n = len(pairs)
    b = sum(1 for _, ca, cb in pairs if cb and not ca)
    c = sum(1 for _, ca, cb in pairs if ca and not cb)
    if n == 0:
        return float("nan")
    se = math.sqrt(max(b + c, 1)) / n
    return 1.96 * se
