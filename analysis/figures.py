"""Figures for the README and the article, drawn as plain SVG from the run records.

No plotting dependency on purpose: the project pins torch/transformers/xgrammar and nothing else,
and a figure must be reproducible from `runs/` by anyone who reran the arms. Every number drawn
here is recomputed from the JSONL records, never typed in.

    uv run python analysis/figures.py          # writes docs/figures/*.svg and prints the numbers
"""
import json, pathlib, statistics, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from minijev.records import read_jsonl                                            # noqa: E402
from minijev.scoring import unit_rows, paired, cluster_bootstrap_delta, correct   # noqa: E402
from minijev import config as C                                                   # noqa: E402

OUT = ROOT / "docs" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREEN, GREY, INK, MUTE = "#1f5f8b", "#d97b27", "#2f7d4f", "#c9c5bc", "#1d1f23", "#6b6f76"
FONT = "font-family='-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif'"


class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def add(self, s): self.parts.append(s)

    def text(self, x, y, s, size=12, color=INK, anchor="start", weight="normal", mono=False):
        fam = "font-family='SF Mono,Menlo,Consolas,monospace'" if mono else FONT
        self.add(f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{color}' text-anchor='{anchor}' font-weight='{weight}' {fam}>{s}</text>")

    def line(self, x1, y1, x2, y2, color=GREY, width=1, dash=None):
        d = f" stroke-dasharray='{dash}'" if dash else ""
        self.add(f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{color}' stroke-width='{width}'{d}/>")

    def rect(self, x, y, w, h, fill, rx=2, opacity=1.0):
        self.add(f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' fill='{fill}' rx='{rx}' opacity='{opacity}'/>")

    def circle(self, x, y, r, fill):
        self.add(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{fill}'/>")

    def path(self, pts, color, width=2):
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        self.add(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='{width}' stroke-linejoin='round'/>")

    def save(self, name):
        body = "\n".join(self.parts)
        (OUT / name).write_text(f"<svg xmlns='http://www.w3.org/2000/svg' width='{self.w}' height='{self.h}' viewBox='0 0 {self.w} {self.h}'>"
                                f"<rect width='{self.w}' height='{self.h}' fill='#ffffff'/>{body}</svg>\n")
        print("wrote", OUT / name)


def rows(run, arm, field=None, cond_prefix="main"):
    rs = unit_rows(read_jsonl(ROOT / "runs" / run / f"{arm}.jsonl"))
    return [r for r in rs if r["condition"].startswith(cond_prefix) and (field is None or r["field"] == field)]


# ----------------------------------------------------------------------------- 1. accuracy by k
def fig_accuracy_by_k():
    aj, b1 = rows("b2", "A1joint", "intent"), rows("b2", "B1", "intent")
    pairs, _, _ = paired(aj, b1)
    ks = sorted({r["k"] for r in b1})
    stats = []
    for k in ks:
        pk = [p for p in pairs if f":k{k}:" in p[0][0]]
        acc_a = sum(1 for _, a, _ in pk if a) / len(pk); acc_b = sum(1 for _, _, b in pk if b) / len(pk)
        d, (lo, hi) = cluster_bootstrap_delta(pk, resamples=5000)
        stats.append((k, acc_a, acc_b, d, lo, hi, len(pk)))
    pooled, (plo, phi) = cluster_bootstrap_delta(pairs)
    print("accuracy by k (intent, b2):", [(k, round(a, 4), round(b, 4), round(d * 100, 2), round(lo * 100, 2), round(hi * 100, 2), n) for k, a, b, d, lo, hi, n in stats])
    print(f"pooled delta {pooled*100:+.2f} pp [{plo*100:+.2f}, {phi*100:+.2f}], pairs {len(pairs)}")

    W, H = 760, 340; L, R, T, B = 60, 300, 40, 60
    s = SVG(W, H)
    s.text(L, 22, "Accuracy on the intent field by number of options k", 14, weight="600")
    s.text(L, 38, "Qwen3-4B-Instruct-2507 · CLINC150, within-domain distractors · 450 texts × 3 field conditions per k · run b2", 11, MUTE)
    # left panel: accuracies
    px0, px1, py0, py1 = L, W - R, T + 20, H - B
    ymin, ymax = 0.80, 1.00
    def X(i): return px0 + (i + 0.5) * (px1 - px0) / len(ks)
    def Y(v): return py1 - (v - ymin) / (ymax - ymin) * (py1 - py0)
    for v in (0.80, 0.85, 0.90, 0.95, 1.00):
        s.line(px0, Y(v), px1, Y(v), GREY, 1, "2,3"); s.text(px0 - 8, Y(v) + 4, f"{v:.2f}", 11, MUTE, "end")
    for i, (k, a, b, *_r) in enumerate(stats):
        s.text(X(i), py1 + 18, f"k = {k}", 11, MUTE, "middle")
    s.path([(X(i), Y(a)) for i, (k, a, *_r) in enumerate(stats)], BLUE, 2.5)
    s.path([(X(i), Y(b)) for i, (k, a, b, *_r) in enumerate(stats)], ORANGE, 2.5)
    for i, (k, a, b, *_r) in enumerate(stats):
        s.circle(X(i), Y(a), 4, BLUE); s.circle(X(i), Y(b), 4, ORANGE)
    s.rect(px0, py0 - 18, 10, 10, BLUE); s.text(px0 + 16, py0 - 9, "Generate JSON (grammar-constrained)", 11)
    s.rect(px0 + 250, py0 - 18, 10, 10, ORANGE); s.text(px0 + 266, py0 - 9, "Read the letter (logits, no generation)", 11)
    # right panel: paired delta with CI
    qx0, qx1 = W - R + 50, W - 30
    s.text(qx0, py0 - 9, "Δ letters − JSON, pp, 95% CI", 11, weight="600")
    dmin, dmax = -4, 4
    def QX(v): return qx0 + (v - dmin) / (dmax - dmin) * (qx1 - qx0)
    s.line(QX(0), py0, QX(0), py1, INK, 1)
    for v in (-4, -2, 0, 2, 4):
        s.text(QX(v), py1 + 18, f"{v:+d}" if v else "0", 10, MUTE, "middle")
    for i, (k, a, b, d, lo, hi, n) in enumerate(stats):
        y = py0 + (i + 0.5) * (py1 - py0) / len(stats)
        s.line(QX(lo * 100), y, QX(hi * 100), y, ORANGE, 2); s.circle(QX(d * 100), y, 4, ORANGE)
        s.text(qx0 - 6, y + 4, f"k={k}", 10, MUTE, "end"); s.text(qx1 + 4, y + 4, f"{d*100:+.1f}", 10, MUTE)
    s.text(L, H - 12, f"Pooled over k: Δ = {pooled*100:+.2f} pp, 95% CI [{plo*100:+.2f}, {phi*100:+.2f}], {len(pairs)} paired observations on 450 texts (cluster bootstrap by text).", 11, MUTE)
    s.save("accuracy_by_k.svg")


# ----------------------------------------------------------------------------- 2. cost vs length
def fig_cost_by_length():
    recs = {}
    for run in ("b2", "b3"):
        for line in (ROOT / "runs" / run / "cost.jsonl").read_text().splitlines():
            r = json.loads(line); recs[(r["input_tokens"], r["fields"])] = r
    lengths = sorted({k[0] for k in recs}); fields = (1, 2, 3)
    print("cost rows:", {(L_, f): (recs[(L_, f)]["perfield_wall_median"], recs[(L_, f)].get("broadcast_wall_median"), recs[(L_, f)]["joint_wall_median"]) for L_ in lengths for f in fields if (L_, f) in recs})
    W, H = 960, 330; s = SVG(W, H)
    s.text(40, 22, "Wall time per 8 texts vs text length, RTX 4090, one request at a time", 14, weight="600")
    s.text(40, 38, "k = 8 options · median of 3 · 32/128/512 from run b2 (batches 16/8); 2048 from run b3 (batches 4/2, comparable only within its row)", 11, MUTE)
    panel_w = (W - 60) / 3
    ymax = max(r["perfield_wall_median"] for r in recs.values()) * 1.08
    import math
    for pi, f in enumerate(fields):
        x0 = 40 + pi * panel_w; x1 = x0 + panel_w - 30; y0 = 70; y1 = H - 60
        s.text(x0, y0 - 8, f"{f} field{'s' if f > 1 else ''}", 12, weight="600")
        def X(L_): return x0 + (math.log2(L_) - 5) / (11 - 5) * (x1 - x0)
        def Y(v): return y1 - v / ymax * (y1 - y0)
        for v in range(0, int(ymax) + 1, 2):
            s.line(x0, Y(v), x1, Y(v), GREY, 1, "2,3")
            if pi == 0: s.text(x0 - 6, Y(v) + 4, f"{v} s", 10, MUTE, "end")
        for L_ in lengths:
            s.text(X(L_), y1 + 16, str(L_), 10, MUTE, "middle")
        series = [("perfield_wall_median", ORANGE, "Read the letter, no cache"), ("broadcast_wall_median", GREEN, "Read the letter, shared prefix"), ("joint_wall_median", BLUE, "Generate JSON")]
        for key, color, _ in series:
            pts = [(X(L_), Y(recs[(L_, f)][key])) for L_ in lengths if (L_, f) in recs and recs[(L_, f)].get(key) is not None]
            s.path(pts, color, 2.2)
            for x, y in pts: s.circle(x, y, 3.5, color)
        s.text((x0 + x1) / 2, H - 28, "text length, tokens (log scale)", 10, MUTE, "middle")
    lx = 40
    for key, color, label in (("perfield_wall_median", ORANGE, "Read the letter, no cache"), ("broadcast_wall_median", GREEN, "Read the letter, shared prefix"), ("joint_wall_median", BLUE, "Generate JSON")):
        s.rect(lx, H - 14, 10, 10, color); s.text(lx + 15, H - 5, label, 11); lx += 230
    s.save("cost_by_length.svg")


# ----------------------------------------------------------------------------- 3. out of scope
def fig_oos():
    A = [r for r in read_jsonl(ROOT / "runs/b2/A1joint.jsonl") if r["condition"].startswith("oos")]
    Bq = [r for r in read_jsonl(ROOT / "runs/b2/B1.jsonl") if r["condition"].startswith("oos")]
    def none_of(r):
        p = (r.get("per_field_pred") or {}).get("intent") if r["arm"] == "A1joint" else r.get("pred")
        return p == C.NOTA
    vals = {}
    for cond, label in (("oos:oos", "out-of-scope texts"), ("oos:in_scope", "in-scope texts")):
        a = [r for r in A if r["condition"] == cond]; b = [r for r in Bq if r["condition"] == cond]
        vals[label] = (sum(map(none_of, a)), len(a), sum(map(none_of, b)), len(b))
    print("oos:", vals)
    W, H = 520, 260; s = SVG(W, H)
    s.text(30, 22, "Choosing “none of the above”, k = 4 + none, 50 + 50 texts, run b2", 14, weight="600")
    x0, y0, y1 = 170, 60, H - 50
    def X(v): return x0 + v / 50 * (W - 40 - x0)
    for v in (0, 10, 20, 30, 40, 50):
        s.line(X(v), y0, X(v), y1, GREY, 1, "2,3"); s.text(X(v), y1 + 16, str(v), 10, MUTE, "middle")
    s.text((x0 + W - 40) / 2, H - 14, "texts out of 50 answered “none of the above”", 10, MUTE, "middle")
    y = y0 + 10
    for label, (a, na, b, nb) in vals.items():
        s.text(x0 - 10, y + 22, label, 12, INK, "end")
        s.rect(x0, y, X(a) - x0, 16, BLUE); s.text(X(a) + 6, y + 12, f"{a} / {na}", 11, BLUE)
        s.rect(x0, y + 20, X(b) - x0, 16, ORANGE); s.text(X(b) + 6, y + 32, f"{b} / {nb}", 11, ORANGE)
        y += 70
    s.rect(30, H - 14, 10, 10, BLUE); s.text(45, H - 5, "Generate JSON", 11); s.rect(160, H - 14, 10, 10, ORANGE); s.text(175, H - 5, "Read the letter", 11)
    s.save("out_of_scope.svg")


# ----------------------------------------------------------------------------- 4. symbol vs label (provisional)
def fig_symbol_vs_label():
    lab, let = rows("b1", "A1label"), rows("b1", "A1letter")
    out = []
    for field, name in (("intent", "intent"), ("domain", "domain"), ("is_banking_or_credit_cards", "boolean")):
        pairs, _, _ = paired([r for r in lab if r["field"] == field], [r for r in let if r["field"] == field])
        acc_l = sum(1 for _, a, _ in pairs if a) / len(pairs); acc_s = sum(1 for _, _, b in pairs if b) / len(pairs)
        d, (lo, hi) = cluster_bootstrap_delta(pairs)
        out.append((name, acc_l, acc_s, d, lo, hi, len(pairs)))
    print("symbol vs label (b1, PROVISIONAL):", [(n, round(a, 4), round(b, 4), round(d * 100, 2), round(lo * 100, 2), round(hi * 100, 2), k) for n, a, b, d, lo, hi, k in out])
    W, H = 620, 250; s = SVG(W, H)
    s.text(30, 22, "Answering with a letter vs writing the option name, same question, run b1", 14, weight="600")
    s.text(30, 38, "PROVISIONAL: the name-writing arm is being re-measured after a harness fix (see article §6); the letter arm is unaffected", 11, "#b3372e")
    x0, y = 130, 62
    def X(v): return x0 + (v - 0.5) / 0.5 * (W - 60 - x0)
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        s.line(X(v), 55, X(v), H - 45, GREY, 1, "2,3"); s.text(X(v), H - 30, f"{v:.1f}", 10, MUTE, "middle")
    for name, a, b, d, lo, hi, n in out:
        s.text(x0 - 10, y + 18, name, 12, INK, "end")
        s.rect(x0, y, X(a) - x0, 13, GREY); s.text(X(a) + 6, y + 11, f"name {a:.3f}", 10, MUTE)
        s.rect(x0, y + 16, X(b) - x0, 13, ORANGE); s.text(X(b) + 6, y + 27, f"letter {b:.3f}  Δ {d*100:+.1f} pp [{lo*100:+.1f}, {hi*100:+.1f}], n={n}", 10, ORANGE)
        y += 52
    s.text(30, H - 8, "accuracy, exact match", 10, MUTE)
    s.save("symbol_vs_label.svg")


if __name__ == "__main__":
    fig_accuracy_by_k(); fig_cost_by_length(); fig_oos(); fig_symbol_vs_label()
