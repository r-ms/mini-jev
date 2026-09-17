"""Local bench: text + flat JSON schema -> the same answer obtained three ways on one model.

It shows the MECHANICS, not production latency: everything runs on our own engine (HF
transformers, MPS/CUDA), sequentially, batch size one. The three ways:
  json    -- one JSON object for all fields under an xgrammar grammar (today's path);
  split   -- the schema is decomposed by field type: enum/boolean -> a lettered multiple-choice
             question whose answer is READ from the logits on one shared prefill of the text;
             string/number -> generation under a single-field grammar;
  labels  -- the same multiple-choice question, but the model WRITES the option name under a
             grammar (the rung that lost 10 points to the letter) -- enum fields only.
Each way runs `repeats` times; the reported time is the median, the pictures come from the last run.

Run: MINIJEV_DEVICE=mps uv run python demo/server.py  ->  http://127.0.0.1:8765/
"""
import json, os, pathlib, statistics, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import xgrammar as xgr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from minijev import config as C, letters as L, prompts, schema as S              # noqa: E402
from minijev.engine import Engine                                                # noqa: E402
from minijev.grammar import GrammarCache, _apply_cpu                            # noqa: E402

PORT = int(os.environ.get("MINIJEV_DEMO_PORT", "8765"))
PROGRESS = {"stage": "idle", "done": 0, "total": 0}


def progress(stage, done=None, total=None):
    PROGRESS["stage"] = stage
    if done is not None: PROGRESS["done"] = done
    if total is not None: PROGRESS["total"] = total
MAX_NEW = 256
LOCK = threading.Lock()


# ----------------------------------------------------------------------------- schema -> fields
def parse_schema(schema: dict):
    """Flat schema only: every property is enum (string+enum) | boolean | string | integer | number.
    Nested objects and arrays are refused -- the bench does not decompose them."""
    if schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
        raise ValueError("expected a schema of the form {type: object, properties: {...}}")
    fields = []
    for name, spec in schema["properties"].items():
        t = spec.get("type")
        desc = spec.get("description") or f'value of "{name}"'
        if t == "string" and "enum" in spec:
            opts = [str(o) for o in spec["enum"]]
            if not 2 <= len(opts) <= 26:
                raise ValueError(f"{name}: enum has {len(opts)} values; the bench supports 2..26 (one letter per option)")
            fields.append({"name": name, "kind": "enum", "options": opts, "desc": desc})
        elif t == "boolean":
            fields.append({"name": name, "kind": "boolean", "options": ["true", "false"], "desc": desc})
        elif t == "string":
            fields.append({"name": name, "kind": "string", "desc": desc,
                           "maxLength": int(spec.get("maxLength", 60))})
        elif t in ("integer", "number"):
            fields.append({"name": name, "kind": t, "desc": desc})
        else:
            raise ValueError(f"{name}: type {t!r} is not decomposed by the bench (nested structures and arrays are out of scope)")
    return fields


def json_schema_for(fields):
    props = {}
    for f in fields:
        if f["kind"] == "enum":
            props[f["name"]] = {"type": "string", "enum": f["options"]}
        elif f["kind"] == "boolean":
            props[f["name"]] = {"type": "boolean"}
        elif f["kind"] == "string":
            props[f["name"]] = {"type": "string", "maxLength": f["maxLength"]}
        else:
            props[f["name"]] = {"type": f["kind"]}
    return {"type": "object", "properties": props, "required": [f["name"] for f in fields],
            "additionalProperties": False}


# ----------------------------------------------------------------------------- prompts
def user_json(text, fields):
    lines = []
    for f in fields:
        if f["kind"] == "enum":
            rendered = " | ".join(json.dumps(o, ensure_ascii=False) for o in f["options"])
            lines.append(f'- "{f["name"]}": {f["desc"]}. Options: {rendered}')
        elif f["kind"] == "boolean":
            lines.append(f'- "{f["name"]}": {f["desc"]}. Options: true | false (JSON booleans)')
        elif f["kind"] == "string":
            lines.append(f'- "{f["name"]}": {f["desc"]}. A short string copied from the TEXT, at most {f["maxLength"]} characters')
        else:
            lines.append(f'- "{f["name"]}": {f["desc"]}. A JSON {f["kind"]}')
    return (f"TEXT:\n{text}\n\nFIELDS (one value per field; for options copy them verbatim):\n"
            + "\n".join(lines) + "\n\nReturn the JSON object now.")


def user_mcq(text, f):
    lines = [f"{L.letter(i)} = {o}" for i, o in enumerate(f["options"])]
    q = f["desc"] if f["kind"] == "enum" else f"Is this true for the TEXT: {f['desc']}?"
    return f"TEXT:\n{text}\n\nQUESTION: {q}\n" + "\n".join(lines) + "\n\nANSWER:"


# ----------------------------------------------------------------------------- tracing processor
class TraceProcessor(xgr.contrib.hf.LogitsProcessor):
    """One generate() call, batch size one. Per step: how many tokens the grammar allowed, what the
    model wanted (top-5 before the mask), what came out after the mask, whether the mask changed
    the choice, and the step time."""

    def __init__(self, compiled, tok):
        super().__init__(compiled)
        self.tok = tok
        self.steps = []
        self.t_prev = None

    def _apply(self, input_ids, scores):
        return _apply_cpu(self, input_ids, scores)

    def __call__(self, input_ids, scores):
        now = time.perf_counter()
        raw = scores[0].float()
        top = torch.topk(raw, 5)
        before = int(raw.argmax().item())
        out = self._apply(input_ids, scores)
        after = out[0]
        allowed = int(torch.isfinite(after).sum().item())
        masked_arg = int(after.argmax().item())
        # best tokens AMONG those the grammar allowed (masked logits keep the raw values)
        k_allowed = min(3, allowed) if allowed > 0 else 0
        a_top = torch.topk(after.float(), k_allowed) if k_allowed else None
        self.steps.append({
            "allowed": allowed,
            "allowed_top": [{"tok": self.tok.decode([int(i)]), "logit": round(float(v), 2)}
                            for v, i in zip(a_top.values.tolist(), a_top.indices.tolist())] if a_top is not None else [],
            # per candidate: was it still allowed after the grammar mask?
            "raw_top": [{"tok": self.tok.decode([int(i)]), "logit": round(float(v), 2),
                         "allowed": bool(torch.isfinite(after[int(i)]).item())}
                        for v, i in zip(top.values.tolist(), top.indices.tolist())],
            "wanted": self.tok.decode([before]),
            "bite": before != masked_arg,
            "ms": None if self.t_prev is None else round((now - self.t_prev) * 1000, 1),
        })
        self.t_prev = now
        return out


class Demo:
    def __init__(self):
        self.eng = Engine(revision=C.MODEL_REVISION)
        self.tok = self.eng.tok
        self.gc = GrammarCache(self.tok, self.eng.cfg.vocab_size)
        self.bare, self.space = L.build_tables(self.tok)
        self.mode = C.run_mode(batch_b=1, batch_a=1, extra="demo")

    # ---- traced generation, batch size one
    @torch.no_grad()
    def generate_traced(self, prompt, compiled):
        enc = self.eng.encode([prompt])
        # No position_ids here: a caller-supplied tensor is frozen across decode steps by
        # transformers 4.57 and every generated token lands on the last prompt position (the
        # "5555…" loop this bench found on 2026-09-17). See Engine.generate.
        proc = TraceProcessor(compiled, self.tok)
        t0 = time.perf_counter()
        proc.t_prev = t0
        out = self.eng.model.generate(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=MAX_NEW,
            pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id,
            logits_processor=[proc], return_dict_in_generate=True)
        self.eng._sync()
        wall = time.perf_counter() - t0
        gen = out.sequences[0, enc["input_ids"].shape[1]:].tolist()
        eos = self.tok.eos_token_id
        eos = eos if isinstance(eos, (list, tuple)) else [eos]
        cut = next((i for i, t in enumerate(gen) if t in eos), len(gen))
        ids = gen[:cut]
        steps = proc.steps[:len(ids) + (1 if cut < len(gen) else 0)]
        for st, tid in zip(steps, ids + ([eos[0]] if cut < len(gen) else [])):
            st["chosen"] = self.tok.decode([tid]) if tid not in eos else "<eos>"
        # The first processor call happens right after the prefill: its delta is the prefill time,
        # every later delta is one decode step.
        prefill_ms = steps[0]["ms"] if steps else round(wall * 1000, 1)
        decode_ms = [s["ms"] for s in steps[1:] if s["ms"] is not None]
        return {"prompt": prompt, "text": self.tok.decode(ids, skip_special_tokens=True), "finish": "eos" if cut < len(gen) else "length",
                "prompt_tokens": int(enc["attention_mask"].sum().item()), "gen_tokens": len(ids) + (1 if cut < len(gen) else 0),
                "wall": wall, "prefill_ms": prefill_ms, "decode_ms": decode_ms, "steps": steps}

    # ---- way 1: one JSON object for all fields
    def run_json(self, text, fields):
        sc = json_schema_for(fields)
        compiled = self.gc.json(S.schema_sha(sc), sc)
        prompt = prompts.render(self.tok, prompts.SYSTEM_A, user_json(text, fields))
        r = self.generate_traced(prompt, compiled)
        r["constraint"] = {"kind": "json_schema", "text": json.dumps(sc, ensure_ascii=False, indent=1)}
        try:
            r["answer"] = json.loads(r["text"])
            r["state"] = "ok"
        except Exception as e:
            r["answer"] = None
            r["state"] = f"malformed_json: {type(e).__name__}"
        r["prompt"] = prompt
        return r

    # ---- way 2: decomposed by type; letters are read from the logits on one shared prefill
    @torch.no_grad()
    def run_split(self, text, fields):
        choice = [f for f in fields if f["kind"] in ("enum", "boolean")]
        free = [f for f in fields if f["kind"] not in ("enum", "boolean")]
        res = {"fields": {}, "prefix_tokens": 0, "tail_tokens": [], "wall_letters": 0.0, "walls_gen": {}}
        if choice:
            rendered = [prompts.render(self.tok, prompts.SYSTEM_B, user_mcq(text, f)) for f in choice]
            prefix = None; tails = []
            for r in rendered:
                p_, s_ = prompts.split_b_prompt(r)
                prefix = prefix or p_
                assert p_ == prefix, "fields have different prefixes -- nothing to share"
                tails.append(s_)
            h, P, S_, wall = self.eng.broadcast_last_hidden(prefix, tails)
            res.update(prefix_tokens=P, tail_tokens=S_, wall_letters=wall, prefix_text=prefix, tails_text=tails)
            full = self.eng.full_logits_from_hidden(h).float()          # the model's own head, its own dtype
            logZ = torch.logsumexp(full, dim=-1)
            for i, f in enumerate(choice):
                k = len(f["options"]); ids = self.bare[:k]
                cand = L.candidate_logits_fp32(h[i:i + 1], self.eng.embed_weight, ids)[0]
                sc = L.score(cand, L.candidate_logits_fp32(h[i:i + 1], self.eng.embed_weight, self.space[:k])[0])
                top = torch.topk(full[i], 10)
                cand_mass = float(torch.exp(torch.logsumexp(full[i][ids], 0) - logZ[i]))
                argmax_id = int(full[i].argmax().item())
                native = full[i][ids]                                       # bf16 head, what greedy would compare
                native_pos = int(torch.argmax(native).item())
                pred = f["options"][sc["pred_pos"]]
                res["fields"][f["name"]] = {
                    "kind": f["kind"], "options": f["options"], "answer": (pred == "true") if f["kind"] == "boolean" else pred,
                    "p": sc["p_cand"], "gap": sc["gap"], "tie": sc["tie"],
                    "logits": [round(float(x), 3) for x in cand],
                    "top_vocab": [{"tok": self.tok.decode([int(t)]), "logit": round(float(v), 2), "is_letter": int(t) in ids}
                                  for v, t in zip(top.values.tolist(), top.indices.tolist())],
                    "cand_mass": round(min(cand_mass, 1.0), 8),
                    "native_logits": [round(float(x), 3) for x in native],
                    "native_pos": native_pos, "native_matches_fp32": native_pos == sc["pred_pos"],
                    "argmax_class": L.classify_argmax(argmax_id, self.bare, self.space, k),
                    "tail_tokens": S_[i],
                    "request": prefix + tails[i],          # the exact bytes the model saw
                    "letter": L.letter(sc["pred_pos"]),
                }
        for f in free:
            sc = json_schema_for([f])
            compiled = self.gc.json(S.schema_sha(sc), sc)
            prompt = prompts.render(self.tok, prompts.SYSTEM_A, user_json(text, [f]))
            r = self.generate_traced(prompt, compiled)
            r["constraint"] = {"kind": "json_schema", "text": json.dumps(sc, ensure_ascii=False, indent=1)}
            try:
                ans = json.loads(r["text"]).get(f["name"])
            except Exception:
                ans = None
            res["fields"][f["name"]] = {"kind": f["kind"], "answer": ans, "gen": r}
            res["walls_gen"][f["name"]] = r["wall"]
            progress(f"generating {f['name']}", PROGRESS["done"] + 1)
        # Sequential sum is what the bench actually spent; the pieces are independent, so a server
        # that batches them concurrently pays the longest piece. Both are reported; the concurrent
        # figure is the headline, the sum is shown next to it.
        res["wall_sequential"] = res["wall_letters"] + sum(res["walls_gen"].values())
        res["wall"] = max([res["wall_letters"]] + list(res["walls_gen"].values()))
        res["answer"] = {n: v["answer"] for n, v in res["fields"].items()}
        return res

    # ---- way 3: the option name under a grammar (enum fields only)
    def run_labels(self, text, fields):
        out = {"fields": {}, "wall": 0.0}
        for f in fields:
            if f["kind"] != "enum":
                continue
            compiled = self.gc.labels(f["options"])
            prompt = prompts.render(self.tok, prompts.SYSTEM_B_LABEL, user_mcq(text, f))
            r = self.generate_traced(prompt, compiled)
            r["constraint"] = {"kind": "ebnf", "text": S.labels_ebnf(f["options"]).strip()}
            progress(f"name for {f['name']}", PROGRESS["done"] + 1)
            # The first-token race: the pre-mask score of every option's first token at step 1.
            # The trace keeps only the top-5 of the vocabulary, so one extra prefill (no
            # generation) recovers the full row; it is cheap and it is not counted in the wall time.
            race = []
            if r["steps"]:
                h, _, _, _ = self.eng.last_hidden([prompt])
                full = self.eng.full_logits_from_hidden(h).float()[0]
                for o in f["options"]:
                    first = self.tok.encode(o, add_special_tokens=False)
                    race.append({"option": o, "n_tokens": len(first), "first_tok": self.tok.decode([first[0]]),
                                 "logit": round(float(full[first[0]]), 2)})
            out["fields"][f["name"]] = {"answer": r["text"], "gen": r, "race": race, "options": f["options"]}
            out["wall"] += r["wall"]
        out["answer"] = {n: v["answer"] for n, v in out["fields"].items()}
        return out

    def run(self, text, schema, repeats=3, with_labels=True):
        fields = parse_schema(schema)
        n_free = sum(1 for f in fields if f["kind"] not in ("enum", "boolean"))
        n_enum = sum(1 for f in fields if f["kind"] == "enum") if with_labels else 0
        per_run = 1 + 1 + n_free + n_enum          # json, letters, each generated field, each name
        with LOCK:
            progress("starting", 0, per_run * repeats)
            walls = {"json": [], "split": [], "labels": []}
            walls_seq = []
            last = {}
            try:
                for _ in range(repeats):
                    progress("generating JSON")
                    last["json"] = self.run_json(text, fields); walls["json"].append(last["json"]["wall"])
                    progress("reading letters", PROGRESS["done"] + 1)
                    last["split"] = self.run_split(text, fields); walls["split"].append(last["split"]["wall"]); walls_seq.append(last["split"]["wall_sequential"])
                    progress("letters read", PROGRESS["done"] + 1)
                    if with_labels and any(f["kind"] == "enum" for f in fields):
                        last["labels"] = self.run_labels(text, fields); walls["labels"].append(last["labels"]["wall"])
            finally:
                progress("idle", PROGRESS["total"])
        for arm in list(walls):
            if walls[arm]:
                last[arm]["walls"] = [round(w, 3) for w in walls[arm]]
                last[arm]["wall_median"] = round(statistics.median(walls[arm]), 3)
        if walls_seq:
            last["split"]["walls_sequential"] = [round(w, 3) for w in walls_seq]
            last["split"]["wall_sequential_median"] = round(statistics.median(walls_seq), 3)
        # tokens read / written per way
        jl = last["json"]
        jl["read_tokens"] = jl["prompt_tokens"]; jl["written_tokens"] = jl["gen_tokens"]
        sp = last["split"]
        gen_prompt = sum(v["gen"]["prompt_tokens"] for v in sp["fields"].values() if "gen" in v)
        sp["read_tokens"] = sp["prefix_tokens"] + sum(sp["tail_tokens"]) + gen_prompt
        sp["read_tokens_no_cache"] = sp["prefix_tokens"] * len(sp["tail_tokens"]) + sum(sp["tail_tokens"]) + gen_prompt
        sp["written_tokens"] = sum(v["gen"]["gen_tokens"] for v in sp["fields"].values() if "gen" in v)
        if "labels" in last:
            lb = last["labels"]
            lb["read_tokens"] = sum(v["gen"]["prompt_tokens"] for v in lb["fields"].values())
            lb["written_tokens"] = sum(v["gen"]["gen_tokens"] for v in lb["fields"].values())
        return {"model": C.MODEL, "revision": C.MODEL_REVISION[:8], "device": self.eng.device, "mode": self.mode,
                "fields": fields, "repeats": repeats, "arms": last}


# ----------------------------------------------------------------------------- HTTP
DEMO = None
INDEX = pathlib.Path(__file__).with_name("index.html")


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        sys.stderr.write("[demo] " + (fmt % a) + "\n")

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")   # re-read per request: edit without restart
        elif self.path == "/progress":
            self._send(200, PROGRESS)
        elif self.path == "/health":
            self._send(200, {"ok": True, "model": C.MODEL, "device": DEMO.eng.device, "mem_gb": DEMO.eng.mem_gb()})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/run":
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", "0"))
        if n > 200_000:
            return self._send(413, {"error": "request too large (200 KB limit)"})
        try:
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            schema = req["schema"] if isinstance(req["schema"], dict) else json.loads(req["schema"])
            text = str(req["text"])
            if len(text) > 20_000:
                return self._send(400, {"error": "text longer than 20 000 characters"})
            if len(schema.get("properties", {})) > 12:
                return self._send(400, {"error": "at most 12 fields in the bench"})
            repeats = max(1, min(int(req.get("repeats", 1)), 5))
            res = DEMO.run(text, schema, repeats=repeats, with_labels=bool(req.get("with_labels", True)))
            self._send(200, res)
        except Exception as e:
            self._send(400, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    print(f"[demo] loading {C.MODEL} on {C.DEVICE} ...", flush=True)
    DEMO = Demo()
    print("[demo] warm-up ...", flush=True)
    DEMO.run("I want to pay my electricity bill today.",
             {"type": "object", "properties": {"intent": {"type": "string", "enum": ["pay_bill", "bill_balance", "cancel"]},
                                               "urgent": {"type": "boolean"}}}, repeats=1, with_labels=True)
    print(f"[demo] ready: http://127.0.0.1:{PORT}/  (memory {DEMO.eng.mem_gb()} GB)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
