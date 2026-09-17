"""xgrammar wiring: compiled-grammar cache plus a processor that counts whether the mask BIT.

'The processor ran' is green by construction (it is called on every step whether or not it
removes anything), so the honest counter is: on how many steps did the mask CHANGE the argmax.
"""
import torch
import xgrammar as xgr

from . import config as C
from .schema import labels_ebnf, letters_ebnf, prob_schema


class CountingProcessor(xgr.contrib.hf.LogitsProcessor):
    """Single-use per generate() call, as upstream requires.

    The mask is applied on the CPU on EVERY device, not only on the non-CUDA ones upstream
    routes that way. Reason, measured 2026-09-16: on CUDA upstream dispatches to a Triton
    kernel, and Triton has no Windows build, so every grammar arm died with "Triton is not
    installed". Forcing the CPU path also keeps the GPU host doing the same arithmetic the
    Mac host did, which is what makes the two hosts' numbers comparable at all. It costs a
    host round trip of one [batch, vocab] tensor per decode step and is counted in the wall
    time like any other cost of this arm.
    """

    def __init__(self, compiled):
        super().__init__(compiled)
        self.steps = 0          # ROW-steps, same unit as bite_steps
        self.calls = 0          # generate() decode steps
        self.bite_steps = 0
        self.allowed_step1 = None

    def _apply(self, input_ids, scores):
        return _apply_cpu(self, input_ids, scores)

    def __call__(self, input_ids, scores):
        before = scores.argmax(-1).clone()
        out = self._apply(input_ids, scores)
        after = out.argmax(-1)
        # Both counters are in ROW-steps. Counting calls here and rows there made
        # "the mask bit on 600 of 138 steps" -- a number in two different units.
        self.calls += 1
        self.steps += int(scores.shape[0])
        self.bite_steps += int((before != after).sum().item())
        if self.allowed_step1 is None:
            self.allowed_step1 = torch.isfinite(out).sum(-1).tolist()
        return out


def _apply_cpu(processor, input_ids, scores):
    """Upstream's body with the device branch removed: always mask on the CPU."""
    if len(processor.matchers) == 0:
        processor.batch_size = input_ids.shape[0]
        processor.compiled_grammars = (processor.compiled_grammars
                                       if len(processor.compiled_grammars) > 1
                                       else processor.compiled_grammars * processor.batch_size)
        processor.matchers = [xgr.GrammarMatcher(processor.compiled_grammars[i])
                              for i in range(processor.batch_size)]
        processor.token_bitmask = xgr.allocate_token_bitmask(processor.batch_size,
                                                             processor.full_vocab_size)
    if not processor.prefilled:
        processor.prefilled = True
    else:
        for i in range(processor.batch_size):
            if not processor.matchers[i].is_terminated():
                assert processor.matchers[i].accept_token(int(input_ids[i][-1].item()))
    for i in range(processor.batch_size):
        if not processor.matchers[i].is_terminated():
            processor.matchers[i].fill_next_token_bitmask(processor.token_bitmask, i)
    dev = scores.device
    cpu_scores = scores.to("cpu")
    xgr.apply_token_bitmask_inplace(cpu_scores, processor.token_bitmask)
    return cpu_scores.to(dev)


class GrammarCache:
    def __init__(self, tok, vocab_size):
        self.info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab_size)
        self.compiler = xgr.GrammarCompiler(self.info)
        self._json = {}
        self._letters = {}
        self._labels = {}

    def json(self, schema_sha, schema):
        if schema_sha not in self._json:
            self._json[schema_sha] = self.compiler.compile_json_schema(
                schema, any_whitespace=True, strict_mode=True,
                max_whitespace_cnt=C.MAX_WHITESPACE_CNT)
        return self._json[schema_sha]

    def letters(self, k):
        if k not in self._letters:
            self._letters[k] = self.compiler.compile_grammar(
                xgr.Grammar.from_ebnf(letters_ebnf(k)))
        return self._letters[k]

    def labels(self, options):
        k = tuple(options)
        if k not in self._labels:
            self._labels[k] = self.compiler.compile_grammar(
                xgr.Grammar.from_ebnf(labels_ebnf(list(options))))
        return self._labels[k]

    def probs(self, options):
        k = ("prob",) + tuple(options)
        if k not in self._labels:
            self._labels[k] = self.compiler.compile_json_schema(
                prob_schema(list(options)), any_whitespace=True, strict_mode=True,
                max_whitespace_cnt=C.MAX_WHITESPACE_CNT)
        return self._labels[k]

    def processor(self, compiled_list):
        return CountingProcessor(compiled_list)
