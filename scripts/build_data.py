import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from minijev import data
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--n-texts", type=int, default=None)
a = ap.parse_args()
m = data.build_sample(n_texts=a.n_texts)
print(json.dumps({k: v for k, v in m.items() if k not in ("intent_vocab_full", "intents_by_domain")},
                 ensure_ascii=False, indent=2))
print("manifest sha:", data.manifest_sha())
