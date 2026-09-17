"""RUN_MANIFEST.json: certified profile, atomic write, resume refusal on any mismatch."""
import json, os, time
from pathlib import Path

CERTIFIED = ("protocol_id", "model_id", "model_revision", "torch", "transformers", "xgrammar",
             "device", "dtype", "attn", "batch_b", "batch_a", "seed",
             "data_manifest_sha")
# code_rev and the template hashes are compared with their own rules below: a commit that adds
# an unused template must not invalidate records whose own prompt_sha still reproduce.


def write(run_dir, manifest):
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "RUN_MANIFEST.json"
    tmp = run_dir / f"RUN_MANIFEST.json.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
                   encoding="utf-8")
    os.replace(tmp, path)


def load(run_dir):
    path = Path(run_dir) / "RUN_MANIFEST.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def refuse_resume_on_mismatch(run_dir, certified):
    old = load(run_dir)
    if old is None:
        return
    prev = old.get("certified", {})
    diff = {k: (prev.get(k), certified.get(k)) for k in CERTIFIED if prev.get(k) != certified.get(k)}
    # Templates: compare only the ones BOTH runs know about. Adding a template is not a change
    # to the existing ones; changing one that the earlier records used is.
    pt, ct = prev.get("template_shas") or {}, certified.get("template_shas") or {}
    for k in sorted(set(pt) & set(ct)):
        if pt[k] != ct[k]:
            diff[f"template_shas.{k}"] = (pt[k], ct[k])
    if diff:
        raise SystemExit("REFUSING TO RESUME: certified profile changed: "
                         + json.dumps(diff, ensure_ascii=False))
