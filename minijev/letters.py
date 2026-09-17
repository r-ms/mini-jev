"""Letter token tables and arm-B scoring.

Candidate logits are computed in fp32 from the last hidden state, never read from the
bf16 lm_head output: at magnitudes 16..32 the bf16 grid step is 0.125, so two candidate
letters whose true gap is 0.05 compare EQUAL and the tie rule would hand the win to the
lower position. That would be a letter-A bias made by the measuring instrument and
indistinguishable from the model's own prior (measured 2026-09-16).
"""
import math
import torch

N_LETTERS = 26


def letter(i: int) -> str:
    if not 0 <= i < N_LETTERS:
        raise ValueError(f"letter index {i} outside A..Z; k>26 needs the two-level code, not run in this PoC")
    return chr(65 + i)


def build_tables(tok):
    """(bare_ids, space_ids). Both asserted single-token; a multi-token letter would make
    one number in the distribution stop meaning one option."""
    bare, space = [], []
    for i in range(N_LETTERS):
        c = letter(i)
        b = tok.encode(c, add_special_tokens=False)
        s = tok.encode(" " + c, add_special_tokens=False)
        if len(b) != 1 or len(s) != 1:
            raise AssertionError(f"letter {c!r}: bare={b} space={s}; both must be single tokens")
        bare.append(b[0]); space.append(s[0])
    return bare, space


def candidate_logits_fp32(hidden_last, weight, ids):
    """hidden_last [B,H] (any dtype) x W[ids] -> [B,k] in fp32."""
    h = hidden_last.float()
    w = weight[ids].float()
    return h @ w.T


def classify_argmax(argmax_id, bare_ids, space_ids, k):
    if argmax_id in bare_ids[:k]:
        return "bare_candidate"
    if argmax_id in space_ids[:k]:
        return "space_candidate"
    if argmax_id in bare_ids or argmax_id in space_ids:
        return "letter_beyond_k"
    return "other"


def score(cand_bare, cand_space):
    """cand_bare/[k] fp32 -> dict. Tie -> lowest position, flagged."""
    k = cand_bare.shape[-1]
    p = torch.softmax(cand_bare, dim=-1)
    best = torch.max(cand_bare)
    n_best = int((cand_bare == best).sum().item())
    pred_pos = int(torch.argmax(cand_bare).item())        # argmax already returns the lowest index
    srt = torch.sort(cand_bare, descending=True).values
    gap = float(srt[0] - srt[1]) if k >= 2 else float("nan")
    merged = torch.logaddexp(cand_bare, cand_space)
    return {
        "p_cand": [round(float(x), 6) for x in p],
        "pred_pos": pred_pos,
        "gap": round(gap, 6),
        "tie": n_best > 1,
        "p_merged": [round(float(x), 6) for x in torch.softmax(merged, dim=-1)],
    }


def choice_confidence(probs):
    """TypeSafe's own definition, copied from their system-one-adapter
    (_utils/confidence_metrics.py): peak probability rescaled from uniform to certainty.
    Adopted verbatim so our confidence numbers are comparable to their published ones
    instead of being a third definition nobody can line up."""
    if len(probs) == 1:
        return 1.0
    total = sum(probs)
    p = [x / total for x in probs] if total else [1.0 / len(probs)] * len(probs)
    u = 1.0 / len(p)
    return (max(p) - u) / (1.0 - u)
