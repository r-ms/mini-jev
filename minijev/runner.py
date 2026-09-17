"""Unit construction and execution for every arm."""
import json, time
import torch

from . import config as C, data, prompts, schema
from .letters import build_tables, candidate_logits_fp32, classify_argmax, score, letter
from .records import key as make_key


def _conds(subset):
    if subset in ("main", "easy"):
        pol = "within_domain" if subset == "main" else "cross_domain"
        ks = list(C.K_LEVELS) + ([C.K_MIXED] if subset == "main" else [])
        return [(f"{subset}:k{k}:{fc}", k, fc, pol) for k in ks for fc in C.FIELD_CONDITIONS]
    raise ValueError(subset)


def build_units(subset, rows, manifest, n_texts=None):
    """-> list of dicts describing one model call each (B: one field; A: one request)."""
    in_scope = [r for r in rows if not r["oos"]]
    oos = [r for r in rows if r["oos"]]
    if n_texts:
        in_scope = in_scope[:n_texts]
        oos = oos[:max(1, n_texts // 4)]
    units = []

    if subset in ("main", "easy"):
        for cond, k, fc, pol in _conds(subset):
            fields = C.FIELD_CONDITIONS[fc]
            for r in in_scope:
                fo, gold_pos, kefs, pols = {}, {}, {}, {}
                for f in fields:
                    o, gp, ke, p = schema.options_for(r["text_id"], f, k, r["gold"][f], manifest, pol)
                    fo[f], gold_pos[f], kefs[f], pols[f] = o, gp, ke, p
                units.append({"condition": cond, "k": k, "fields": list(fields), "row": r,
                              "options": fo, "gold_pos": gold_pos, "k_eff": kefs, "policy": pols})
        return units

    if subset == "perm":
        for k in C.PERM_K:
            for r in in_scope[:C.PERM_N_TEXTS]:
                o, gp, ke, p = schema.options_for(r["text_id"], "intent", k, r["gold"]["intent"], manifest)
                for shift in range(len(o)):
                    rot = o[shift:] + o[:shift]
                    units.append({"condition": f"perm:k{k}:shift{shift}", "k": k, "fields": ["intent"],
                                  "row": r, "options": {"intent": rot},
                                  "gold_pos": {"intent": rot.index(r["gold"]["intent"])},
                                  "k_eff": {"intent": len(rot)}, "policy": {"intent": "within_domain"},
                                  "shift": shift})
        return units

    if subset == "oos":
        # PAIRED: oos texts AND in-scope texts under the same option set. oos alone is a
        # one-sided probe -- "always none of the above" would score 100%.
        for r in oos[:C.N_OOS]:
            base = in_scope[0]
            o, _, ke, p = schema.options_for(r["text_id"], "intent", 4, base["gold"]["intent"],
                                             manifest, include_nota=True)
            units.append({"condition": "oos:oos", "k": 4, "fields": ["intent"], "row": r,
                          "options": {"intent": o}, "gold_pos": {"intent": o.index(C.NOTA)},
                          "k_eff": {"intent": len(o)}, "policy": {"intent": "nota"}})
        for r in in_scope[:C.N_INSCOPE_PAIRED]:
            o, gp, ke, p = schema.options_for(r["text_id"], "intent", 4, r["gold"]["intent"],
                                              manifest, include_nota=True)
            units.append({"condition": "oos:in_scope", "k": 4, "fields": ["intent"], "row": r,
                          "options": {"intent": o}, "gold_pos": {"intent": gp},
                          "k_eff": {"intent": len(o)}, "policy": {"intent": "nota"}})
        return units

    if subset == "nogold":
        for r in in_scope[:C.NO_GOLD_N]:
            o, gp, ke, p = schema.options_no_gold(r["text_id"], "intent", 4, r["gold"]["intent"], manifest)
            units.append({"condition": "ctl:nogold", "k": 4, "fields": ["intent"], "row": r,
                          "options": {"intent": o}, "gold_pos": {"intent": -1},
                          "k_eff": {"intent": len(o)}, "policy": {"intent": "nogold"}})
        return units

    if subset == "textanswer":
        for r in in_scope[:C.TEXT_ANSWER_CONTROL_N]:
            o, gp, ke, p = schema.options_for(r["text_id"], "intent", 4, r["gold"]["intent"], manifest)
            units.append({"condition": "ctl:textanswer", "k": 4, "fields": ["intent"], "row": r,
                          "options": {"intent": o}, "gold_pos": {"intent": gp},
                          "k_eff": {"intent": len(o)}, "policy": {"intent": "within_domain"}})
        return units

    raise ValueError(subset)


def b_prompts(engine, units, option_format="eq"):
    """One prompt per (unit, field). Returns flat list of (unit_idx, field, prompt)."""
    out = []
    for i, u in enumerate(units):
        for f in u["fields"]:
            p = prompts.render(engine.tok, prompts.SYSTEM_B,
                               prompts.user_b(u["row"]["text"], f, u["options"][f], option_format))
            out.append((i, f, p))
    return out


def a_prompt(engine, u, fields=None):
    fields = fields or u["fields"]
    fo = {f: u["options"][f] for f in fields}
    return prompts.render(engine.tok, prompts.SYSTEM_A, prompts.user_a(u["row"]["text"], fo)), fo


def sort_units(items, engine):
    """Frozen batching rule: longest first, key ascending. Identical in every run unless the
    run deliberately uses the second frozen order (the honest noise band)."""
    lens = [len(engine.tok(p, add_special_tokens=False)["input_ids"]) for *_, p in items]
    order = sorted(range(len(items)), key=lambda i: (-lens[i], str(items[i][:-1])))
    return [items[i] for i in order], [lens[i] for i in order]


def parse_a(raw, fields_options):
    """-> (state, per_field_pred). Never turns a truncation into a valid empty answer."""
    txt = (raw or "").strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        txt = txt.split("\n", 1)[1] if "\n" in txt else txt
    try:
        obj = json.loads(txt)
    except Exception:
        return "malformed_json", {}
    if not isinstance(obj, dict):
        return "malformed_json", {}
    missing = [f for f in fields_options if f not in obj]
    if missing:
        return "missing_key", {}
    extra = [k for k in obj if k not in fields_options]
    if extra:
        return "extra_key", {}
    pred = {}
    for f, opts in fields_options.items():
        v = obj[f]
        if isinstance(v, bool):
            v = "true" if v else "false"
        pred[f] = v
    bad = [f for f, v in pred.items() if v not in fields_options[f]]
    return ("out_of_enum" if bad else "ok"), pred
