"""Per-request JSONL records: append with flush, resume by key, truncated-tail repair."""
import hashlib, json, os, subprocess, time
from pathlib import Path


def code_rev():
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parents[1]).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                    cwd=Path(__file__).resolve().parents[1]).stdout.strip())
        return rev or "no-commit", dirty
    except Exception:
        return "unknown", True


def key(condition, text_id, field, arm):
    return f"{condition}|{text_id}|{field}|{arm}"


class Writer:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seen = set()
        self.dropped_tail = 0
        if self.path.exists():
            self._load_existing()
        self.f = self.path.open("a", encoding="utf-8")

    def _load_existing(self):
        """A kill during flush leaves a truncated last line. Drop it and say so -- silently
        keeping it would make a half-written record look like a finished one."""
        raw = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        good = []
        for i, line in enumerate(raw):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                if i == len(raw) - 1:
                    self.dropped_tail = 1
                    continue
                raise
            good.append(line)
            self.seen.add(rec["key"])
        if self.dropped_tail:
            self.path.write_text("\n".join(good) + ("\n" if good else ""), encoding="utf-8")

    def has(self, k):
        return k in self.seen

    def write(self, rec):
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()
        self.seen.add(rec["key"])

    def close(self):
        self.f.close()


def read_jsonl(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out
