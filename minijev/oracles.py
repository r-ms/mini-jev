"""Fake engines that travel the REAL run.py -> records -> report path.

Control (a) (re-score against a random gold) is 1/k for ANY predictor, so it proves the
scorer's arithmetic and nothing about the plumbing: it stays green if gold_pos is recorded
before the shuffle, if options[pred_pos] is off by one, or if the letter table came from a
different tokenizer. These oracles are the external operand for exactly that.

They expose the SAME interface as Engine (last_hidden, full_logits_from_hidden, embed_weight),
so run.py takes one code path for both. A second code path for the oracle would mean the
oracle no longer tests the path the real run uses.
"""
import torch

from .letters import build_tables, N_LETTERS

PEAK = 30.0


class _OracleBase:
    def __init__(self, real_tok, vocab=None):
        self.tok = real_tok
        self.cfg = type("cfg", (), {"vocab_size": vocab or len(real_tok)})()
        self.device = "cpu"
        self.model_id = f"oracle:{self.name}"
        self.revision = None
        self.attn = "none"
        self.dtype_name = "float32"
        self.bare, self.space = build_tables(real_tok)
        # W[bare[j]] = e_j  =>  h @ W[bare[:k]].T picks out exactly the target position
        W = torch.zeros(self.cfg.vocab_size, N_LETTERS)
        for j in range(N_LETTERS):
            W[self.bare[j], j] = PEAK
        self.embed_weight = W
        self._targets = []

    def set_targets(self, targets):
        """targets: [(k, position)] aligned with the next batch."""
        self._targets = list(targets)

    def last_hidden(self, prompts):
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        n_real = enc["attention_mask"].sum(1).tolist()
        n_pad = (enc["attention_mask"].shape[1] - enc["attention_mask"].sum(1)).tolist()
        h = torch.zeros(len(prompts), N_LETTERS)
        for i, (k, pos) in enumerate(self._targets[:len(prompts)]):
            h[i, pos % k] = 1.0
        return h, n_real, n_pad, 0.0

    def full_logits_from_hidden(self, h):
        return h @ self.embed_weight.T

    def mem_gb(self):
        return None


class OracleGold(_OracleBase):
    name = "oracle-gold"


class OracleNext(_OracleBase):
    name = "oracle-next"


def target_for(engine_name, gold_pos, k):
    if engine_name == "oracle-gold":
        return (k, gold_pos)
    if engine_name == "oracle-next":
        return (k, (gold_pos + 1) % k)
    raise ValueError(engine_name)
