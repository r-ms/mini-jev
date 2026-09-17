"""CLINC150 adapter: stratified sample, domain map, per-field gold, frozen manifest.

Runs read data/clinc/sample.jsonl only; they never touch HF. The domain map is pinned by
sha256 and its coverage over the 150 in-scope intents is asserted, not assumed.
"""
import hashlib, json, os, random, urllib.request
from pathlib import Path

from . import config as C

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch_domains(force=False) -> dict:
    """intent -> domain, pinned by sha256."""
    path = DATA / "domains.json"
    if force or not path.exists():
        raw = urllib.request.urlopen(C.DOMAINS_URL, timeout=60).read()
        got = _sha256_bytes(raw)
        if got != C.DOMAINS_SHA256:
            raise AssertionError(f"domains.json sha256 {got} != pinned {C.DOMAINS_SHA256}")
        path.write_bytes(raw)
    raw = path.read_bytes()
    got = _sha256_bytes(raw)
    if got != C.DOMAINS_SHA256:
        raise AssertionError(f"domains.json on disk sha256 {got} != pinned {C.DOMAINS_SHA256}")
    by_domain = json.loads(raw)
    intent2domain = {}
    for dom, intents in by_domain.items():
        for it in intents:
            if it in intent2domain:
                raise AssertionError(f"intent {it} in two domains")
            intent2domain[it] = dom
    if len(intent2domain) != 150 or len(by_domain) != 10:
        raise AssertionError(f"expected 10 domains x 15 = 150, got {len(by_domain)} / {len(intent2domain)}")
    return intent2domain


def gold_fields(intent: str, intent2domain: dict) -> dict:
    dom = intent2domain[intent]
    g = {"intent": intent, "domain": dom}
    for f in C.BOOL_FIELDS:
        g[f] = "true" if dom in C.BOOL_DOMAINS[f] else "false"
    return g


def build_sample(n_texts=None, n_per_domain=None):
    """Stratified N in-scope texts + N_OOS oos texts. Writes sample.jsonl and data_manifest.json."""
    from datasets import load_dataset
    n_texts = n_texts or C.N_TEXTS
    n_per_domain = n_per_domain or C.N_PER_DOMAIN
    intent2domain = fetch_domains()

    ds = load_dataset(C.DATASET, C.DATASET_CONFIG, split=C.DATASET_SPLIT, revision=C.DATASET_REVISION)
    names = ds.features["intent"].names
    n_rows = len(ds)

    by_domain, oos_rows = {}, []
    for idx, row in enumerate(ds):
        name = names[row["intent"]]
        if name == "oos":
            oos_rows.append((idx, row["text"]))
            continue
        by_domain.setdefault(intent2domain[name], []).append((idx, row["text"], name))

    if set(by_domain) != set(intent2domain.values()):
        raise AssertionError("domain coverage in the split does not match the domain map")

    # Stratify by INTENT, not by domain: the experiment is about label sets assembled at
    # request time, so every one of the 150 intents must appear as a gold answer. Domain
    # balance follows for free (15 intents per domain), the reverse does not.
    per_intent = C.N_PER_INTENT if n_texts == C.N_TEXTS else max(1, n_texts // 150)
    by_intent = {}
    for dom, items in by_domain.items():
        for idx, text, name in items:
            by_intent.setdefault(name, []).append((idx, text, name))
    missing = sorted(set(intent2domain) - set(by_intent))
    if missing:
        raise AssertionError(f"{len(missing)} intents absent from the split: {missing[:5]}")
    samples = []
    for name in sorted(by_intent):
        rng = random.Random(f"{C.SEED}:clinc:sample:{name}")
        pool = sorted(by_intent[name])
        for idx, text, _ in rng.sample(pool, min(per_intent, len(pool))):
            samples.append({"text_id": f"clinc-{idx:06d}", "row": idx, "text": text,
                            "gold": gold_fields(name, intent2domain), "oos": False})
    samples.sort(key=lambda s: s["text_id"])

    rng = random.Random(f"{C.SEED}:clinc:oos")
    n_oos = C.N_OOS if n_texts == C.N_TEXTS else max(1, n_texts // 4)
    oos = [{"text_id": f"clinc-{i:06d}", "row": i, "text": t, "gold": {"intent": "oos"}, "oos": True}
           for i, t in sorted(rng.sample(sorted(oos_rows), min(n_oos, len(oos_rows))))]

    out_dir = DATA / "clinc"; out_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(s, ensure_ascii=False, sort_keys=True) for s in samples + oos]
    (out_dir / "sample.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sample_sha = _sha256_bytes("\n".join(sorted(lines)).encode("utf-8"))

    vocab = {"intent": sorted({s["gold"]["intent"] for s in samples}),
             "domain": sorted(set(intent2domain.values())),
             **{f: ["true", "false"] for f in C.BOOL_FIELDS}}
    intents_all = sorted(intent2domain)
    manifest = {
        "dataset": C.DATASET, "config": C.DATASET_CONFIG, "split": C.DATASET_SPLIT,
        "hf_revision": C.DATASET_REVISION, "n_rows_split": n_rows,
        "domains_sha256": C.DOMAINS_SHA256, "domain_coverage": f"{len(intent2domain)}/150",
        "filters": {"in_scope": "intent != oos",
                    "stratification": f"{per_intent} per intent x 150 (every intent covered)"},
        "n_in_scope": len(samples), "n_oos": len(oos), "sample_sha": sample_sha,
        # NAMED SO IT CANNOT BE MISREAD: the label SPACE is 150 intents; only some of them
        # happen to be the gold of a sampled text. Distractors are drawn from the space, not
        # from the sample.
        "label_space_sizes": {"intent": len(intents_all), "domain": 10,
                              **{f: 2 for f in C.BOOL_FIELDS}},
        "n_distinct_gold_in_sample": {k: len(v) for k, v in vocab.items()},
        "intent_vocab_full": intents_all,
        "intents_by_domain": {d: sorted(i for i in intents_all if intent2domain[i] == d)
                              for d in sorted(set(intent2domain.values()))},
        "seed": C.SEED,
    }
    (DATA / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                                        sort_keys=True), encoding="utf-8")
    return manifest


def load_sample():
    path = DATA / "clinc" / "sample.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/build_data.py first")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    manifest = json.loads((DATA / "data_manifest.json").read_text(encoding="utf-8"))
    return rows, manifest


def manifest_sha():
    return _sha256_bytes((DATA / "data_manifest.json").read_bytes())
