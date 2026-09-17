"""Frozen constants. Changing anything here changes the run mode and invalidates comparisons."""

SEED = 20260916
BOOTSTRAP_SEED = 20260916
BOOTSTRAP_RESAMPLES = 2000

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
SMOKE_MODEL = "Qwen/Qwen3-0.6B"

DATASET = "clinc/clinc_oos"
DATASET_CONFIG = "plus"
DATASET_SPLIT = "test"
DATASET_REVISION = "155b9c710419136e17307b80d0a13e68cd46b4ec"
DOMAINS_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/domains.json"
DOMAINS_SHA256 = "b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a"

# The task is about label sets built at request time, so coverage of the 150 INTENTS
# matters more than balance over the 10 domains: 3 per intent covers every label the
# schema builder can draw. Narrower intervals are a side effect, not the reason.
N_TEXTS = 450
N_PER_INTENT = 3
N_PER_DOMAIN = 45
# 15 = gold + ALL 14 same-domain distractors. It was 14, which left one distractor unused
# for no stated reason -- an off-by-one found by external review, not by the code.
K_LEVELS = (2, 4, 8, 15)
K_MIXED = 16               # needs cross-domain fill; reported on its own row
FIELD_CONDITIONS = {"f1": ("intent",),
                    "f2": ("intent", "domain"),
                    "f3": ("intent", "domain", "is_banking_or_credit_cards")}
BOOL_FIELDS = ("is_banking_or_credit_cards",)
BOOL_DOMAINS = {"is_banking_or_credit_cards": ("banking", "credit_cards")}

DISTRACTOR_POLICIES = ("within_domain", "cross_domain")   # primary (hard), secondary (easy)

NOTA = "none of the above"
N_OOS = 50
N_INSCOPE_PAIRED = 50
PERM_N_TEXTS = 50
PERM_K = (4, 8)
NO_GOLD_N = 40
TEXT_ANSWER_CONTROL_N = 200
CLOZE_N = 50

BATCH_B = 16
BATCH_A = 8
MAX_NEW_TOKENS_A = 96
MAX_NEW_TOKENS_TEXT_CONTROL = 8
ATTN_IMPLEMENTATION = "eager"
DTYPE = "bfloat16"
import os as _os
# .strip() is not cosmetic: `set MINIJEV_DEVICE=cuda && cmd` on Windows puts a TRAILING
# SPACE in the value, and torch then fails with "Invalid device string: 'cuda '".
DEVICE = _os.environ.get("MINIJEV_DEVICE", "mps").strip()   # "cuda" on the GPU host, "mps" on the Mac
MAX_WHITESPACE_CNT = 4

# STOP thresholds (PREREG)
STOP_M5_GATE = 0.90            # S3: letter emission on the first 64 4B requests
STOP_M5_CELL = 0.90            # per-cell: below this the cell is "mechanism did not fire"
STOP_SATURATION = 0.97         # S8: A0 intent accuracy at k_max
STOP_LENGTH_RATE = 0.01        # S9: A1 state=length share
STOP_FP32_TIE_RATE = 0.005     # S10
P3_M5_POOLED = 0.95
P3_M14_AGREEMENT = 0.99
P3_TEXT_CONTROL_MISMATCH = 0.02
P1_BOUND_POOLED_PP = -3.0
P1_BOUND_KMAX_PP = -5.0
P2_COST_RATIO_F1 = 0.8

PROTOCOL_ID = "mini-jev-v1"

FIELD_QUESTIONS = {
    "intent": {
        "question": "What is the intent of the user in the text above?",
        "desc": "the user's intent",
    },
    "domain": {
        "question": "Which topic domain does the request above belong to?",
        "desc": "the topic domain of the request",
    },
    "is_banking_or_credit_cards": {
        "question": "Is the request above about banking or credit cards?",
        "desc": "whether the request is about banking or credit cards",
    },
}


def run_mode(batch_b=BATCH_B, batch_a=BATCH_A, attn=ATTN_IMPLEMENTATION, extra=""):
    import torch, transformers, importlib.metadata as md
    # The DEVICE goes in the string. It said "hf-mps-greedy" on a CUDA run once (caught
    # 2026-09-16, 30 seconds in): a provenance label that names the wrong engine is worse
    # than none, because it invites pooling two hosts' numbers as if they were one run.
    s = (f"hf-{DEVICE}-greedy,{DTYPE},attn={attn},B.batch={batch_b},A.batch={batch_a},"
         f"torch={torch.__version__},transformers={transformers.__version__},"
         f"xgrammar={md.version('xgrammar')}")
    return s + ("," + extra if extra else "")
