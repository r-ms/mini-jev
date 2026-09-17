"""Per-request option sets (seeded), JSON schemas, EBNF letter grammars, hashes."""
import hashlib, json, random

from . import config as C
from .letters import letter


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, separators=(",", ":"),
                                     sort_keys=False).encode("utf-8")).hexdigest()


def options_for(text_id, field, k, gold, manifest, policy="within_domain", include_nota=False):
    """Returns (options, gold_pos, k_eff, policy_used).

    The RNG key deliberately excludes the field-condition, so intent@k=4 for a text is the
    same option set in f1, f2 and f3 -- that is what isolates the k axis from the f axis.
    """
    rng = random.Random(f"{C.SEED}:clinc:{text_id}:{field}:{k}:{policy}")
    if field in C.BOOL_FIELDS:
        opts = ["true", "false"]
        rng.shuffle(opts)
        return opts, opts.index(gold), 2, "boolean"

    if field == "domain":
        pool = [d for d in manifest["intents_by_domain"] if d != gold]
    elif policy == "within_domain":
        dom = next(d for d, ints in manifest["intents_by_domain"].items() if gold in ints)
        pool = [i for i in manifest["intents_by_domain"][dom] if i != gold]
    else:
        pool = [i for i in manifest["intent_vocab_full"] if i != gold]

    used = policy
    want = k - 1
    if want > len(pool):
        if field == "intent" and policy == "within_domain":
            # k=16 needs a cross-domain fill; reported on its own row, never merged with
            # the pure within-domain levels.
            dom = next(d for d, ints in manifest["intents_by_domain"].items() if gold in ints)
            extra = [i for i in manifest["intent_vocab_full"]
                     if i != gold and i not in manifest["intents_by_domain"][dom]]
            picked = list(pool) + rng.sample(sorted(extra), want - len(pool))
            used = "mixed"
        else:
            picked = list(pool)
            want = len(pool)
    else:
        picked = rng.sample(sorted(pool), want)

    opts = [gold] + picked
    if include_nota:
        opts.append(C.NOTA)
    rng.shuffle(opts)
    return opts, opts.index(gold), len(opts), used


def options_no_gold(text_id, field, k, gold, manifest, policy="within_domain"):
    """Control 5: the gold option is absent. A1 must still answer in-enum."""
    opts, gold_pos, _, used = options_for(text_id, field, k + 1, gold, manifest, policy)
    opts = [o for o in opts if o != gold]
    return opts, -1, len(opts), used


def json_schema(fields_options: dict) -> dict:
    """Key order is part of the instruction; never sorted."""
    props = {}
    for f, opts in fields_options.items():
        if f in C.BOOL_FIELDS:
            props[f] = {"type": "boolean"}
        else:
            props[f] = {"type": "string", "enum": list(opts)}
    return {"type": "object", "properties": props,
            "required": list(fields_options.keys()), "additionalProperties": False}


def letters_ebnf(k: int) -> str:
    if k < 2:
        raise ValueError("k must be >= 2")
    return "root ::= " + " | ".join(f'"{letter(i)}"' for i in range(k)) + "\n"


def prob_schema(options) -> dict:
    """TypeSafe's own LLM baseline shape (system-one-adapter, answer_mode='probabilities'):
    the MODEL writes a number per option and code normalizes afterwards. Recorded here so our
    logit-read arm is compared against the baseline THEY published, not only against JSON
    labels. Note what this costs: the model generates both the labels and the digits."""
    # THE BOUND ON DIGITS IS AN ENUMERATION, not a pattern. With `{"type":"number"}` the grammar
    # admits an endless fraction and the model looped: 4337 of 5369 answers hit the token cap on
    # `0.99999999...` (run b1, 2026-09-16). The first fix through `pattern` DOES NOT BIND: the regex
    # `^(0(\.[0-9]{1,3})?|1(\.0{1,3})?)$` is correct under Python (checked on 8 probes), yet under
    # xgrammar 0.2.7 generation still emitted `"000"`, i.e. pattern support there is incomplete.
    # An enumeration of strings is the mechanism proven to bind in this same harness (0 enum
    # violations on the whole A1joint grid). The grid is 0.00...1.00 in steps of 0.01 (101 strings);
    # the code reads the string as a float, as TypeSafe's adapter reads its values.
    grid = [f"{i/100:.2f}" for i in range(101)]
    return {"type": "object",
            "properties": {"probabilities": {
                "type": "object",
                "properties": {o: {"type": "string", "enum": grid} for o in options},
                "required": list(options), "additionalProperties": False}},
            "required": ["probabilities"], "additionalProperties": False}


def labels_ebnf(options) -> str:
    """Grammar over the option strings themselves. This is the rung that separates
    'answer through a symbol' from 'answer with the label': same MCQ prompt as arm B,
    but the model emits the option text. Without it, the letter-vs-label difference is
    confounded with the JSON-prompt-vs-MCQ-prompt difference."""
    def esc(o):
        return '"' + o.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return "root ::= " + " | ".join(esc(o) for o in options) + "\n"


def schema_sha(obj) -> str:
    return _sha(obj)
