"""Prompt templates. Class A (role, format) in system with a pointer; class B (the specific
question) after the text. A0 and A1 render byte-identically -- asserted at call sites."""
import hashlib, json

from . import config as C
from .letters import letter

SYSTEM_B = ("You are a classifier. You will be shown a TEXT and then ONE multiple-choice "
            "QUESTION about it. The question is at the end of the message. Reply with exactly "
            "one capital letter, the label of the chosen option, and nothing else: no words, "
            "no punctuation, no explanation.")

SYSTEM_A = ("You are a classifier. You will be shown a TEXT and then a list of FIELDS to fill. "
            "The field list is at the end of the message. Reply with one single-line JSON object "
            "whose keys are exactly the field names in the given order and whose values are copied "
            "verbatim from that field's allowed options. Output the JSON object only: no code "
            "fences, no explanation.")

SYSTEM_B_LABEL = ("You are a classifier. You will be shown a TEXT and then ONE multiple-choice "
                  "QUESTION about it. The question is at the end of the message. Reply with the "
                  "exact text of the chosen option, copied verbatim, and nothing else: no letter, "
                  "no punctuation, no explanation.")

SYSTEM_B_PROB = ("You are a classifier. You will be shown a TEXT and then ONE multiple-choice "
                 "QUESTION about it. The question is at the end of the message. Reply with one "
                 "JSON object giving the probability of each option, as numbers between 0 and 1 "
                 "that sum to 1. Output the JSON object only: no explanation.")

OPTION_FORMATS = {"eq": "{L} = {opt}", "paren": "({L}) {opt}"}   # 'eq' is the frozen primary


def user_b(text, field, options, option_format="eq"):
    fmt = OPTION_FORMATS[option_format]
    lines = [fmt.format(L=letter(i), opt=o) for i, o in enumerate(options)]
    return (f"TEXT:\n{text}\n\nQUESTION: {C.FIELD_QUESTIONS[field]['question']}\n"
            + "\n".join(lines) + "\n\nANSWER:")


def user_a(text, fields_options):
    lines = []
    for f, opts in fields_options.items():
        desc = C.FIELD_QUESTIONS[f]["desc"]
        if f in C.BOOL_FIELDS:
            rendered = "true | false (JSON booleans)"
        else:
            rendered = " | ".join(json.dumps(o, ensure_ascii=False) for o in opts)
        lines.append(f'- "{f}": {desc}. Options: {rendered}')
    return (f"TEXT:\n{text}\n\nFIELDS (choose exactly one option per field, copy it verbatim):\n"
            + "\n".join(lines) + "\n\nReturn the JSON object now.")


def render(tok, system, user):
    """enable_thinking=False is ignored by the Qwen3-4B-Instruct-2507 template and REQUIRED by
    the Qwen3-0.6B hybrid template (without it the smoke model opens a <think> block and the
    first generated token is not the answer)."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def prompt_sha(rendered: str) -> str:
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _h(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def template_shas() -> dict:
    """PER TEMPLATE, not one aggregate. An aggregate hash refuses a resume when an UNUSED
    template is added -- measured: adding SYSTEM_B_LABEL moved the aggregate and would have
    invalidated 6000 already-correct B1 records whose own prompt_sha reproduce exactly."""
    return {"system_a": _h(SYSTEM_A), "system_b": _h(SYSTEM_B),
            "system_b_label": _h(SYSTEM_B_LABEL), "system_b_prob": _h(SYSTEM_B_PROB), "fields": _h(C.FIELD_QUESTIONS),
            "formats": _h(OPTION_FORMATS)}


def template_sha() -> str:
    return _h(template_shas())


def split_b_prompt(rendered: str):
    """Split a rendered B prompt into (shared prefix up to and including the text, per-field
    suffix from QUESTION: onward). The per-field arm and the broadcast arm must see the SAME
    bytes; the engine asserts the token boundary is clean."""
    i = rendered.index("QUESTION: ")
    return rendered[:i], rendered[i:]
