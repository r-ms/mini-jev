"""Model loading and the two forward paths.

Both arms run on the same engine so cost and accuracy are comparable. Positions are passed
EXPLICITLY on bare forwards: there transformers derives position_ids from cache_position
(modeling_qwen3.py:383), which under left padding offsets padded rows. generate() derives them
from the mask itself at every step and MUST NOT be given the tensor (see Engine.generate).
"""
import time
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from . import config as C


def position_ids_from_mask(mask):
    return (mask.cumsum(-1) - 1).clamp(min=0)


class Engine:
    name = "model"

    def __init__(self, model_id=None, revision=None, device=None, attn=None, dtype=None):
        self.model_id = model_id or C.MODEL
        self.revision = revision
        self.device = device or C.DEVICE
        self.attn = attn or C.ATTN_IMPLEMENTATION
        dt = dtype or C.DTYPE
        self.torch_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dt]
        self.dtype_name = dt

        self.tok = AutoTokenizer.from_pretrained(self.model_id, revision=revision)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.cfg = AutoConfig.from_pretrained(self.model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, revision=revision, dtype=self.torch_dtype,
            attn_implementation=self.attn, low_cpu_mem_usage=True).to(self.device).eval()
        self.embed_weight = self.model.get_input_embeddings().weight

    def encode(self, prompts):
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def last_hidden(self, prompts):
        """-> (h_last [B,H], prompt_tokens [B], pad_tokens [B], wall_s)."""
        enc = self.encode(prompts)
        pos = position_ids_from_mask(enc["attention_mask"])
        t0 = time.perf_counter()
        out = self.model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                               position_ids=pos, use_cache=False)
        self._sync()
        wall = time.perf_counter() - t0
        h = out.last_hidden_state[:, -1, :]
        n_real = enc["attention_mask"].sum(1).tolist()
        n_pad = (enc["attention_mask"].shape[1] - enc["attention_mask"].sum(1)).tolist()
        return h, n_real, n_pad, wall

    @torch.no_grad()
    def full_logits_from_hidden(self, h):
        """The model's own decoder head, in its own dtype: what greedy would actually sample."""
        return self.model.lm_head(h)

    @torch.no_grad()
    def generate(self, prompts, processors=None, max_new_tokens=None):
        """position_ids are NOT passed here, on purpose (defect found 2026-09-17 by the demo bench).

        transformers 4.57 keeps a caller-supplied position_ids tensor unchanged across decode steps
        and slices its LAST column for every new token (generation/utils.py, step 5 of
        prepare_inputs_for_generation; _update_model_kwargs_for_generation extends attention_mask
        and cache_position but never position_ids). Every generated token therefore sat at the
        last prompt position and the model looped: "555-0102" became "5555555555555555555" and
        a JSON integer became "4555555…". Measured placebo: the same prompt gives
        {"callback_phone": "555-0102", "people_waiting": 4} in 29 tokens without the tensor and a
        96-token loop with it. The first generated token is identical either way (the prefill
        positions are right in both), which is why the one-token arms were unaffected.
        Without the tensor generate() derives positions from the attention mask at EVERY step
        (cumsum-1, pads set to 1 and masked), i.e. the same convention position_ids_from_mask
        uses for the prompt. smoke_positions.py guards this."""
        enc = self.encode(prompts)
        t0 = time.perf_counter()
        out = self.model.generate(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            do_sample=False, temperature=None, top_p=None, top_k=None,
            max_new_tokens=max_new_tokens or C.MAX_NEW_TOKENS_A,
            pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id,
            logits_processor=processors, return_dict_in_generate=True)
        self._sync()
        wall = time.perf_counter() - t0
        gen = out.sequences[:, enc["input_ids"].shape[1]:]
        texts, n_gen, finish = [], [], []
        eos_ids = self.tok.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
        for row in gen:
            ids = row.tolist()
            cut = len(ids)
            for i, t in enumerate(ids):
                if t in eos_ids:
                    cut = i
                    break
            texts.append(self.tok.decode(ids[:cut], skip_special_tokens=True))
            n_gen.append(cut + (1 if cut < len(ids) else 0))   # the eos step costs a decode step
            finish.append("eos" if cut < len(ids) else "length")
        n_real = enc["attention_mask"].sum(1).tolist()
        n_pad = (enc["attention_mask"].shape[1] - enc["attention_mask"].sum(1)).tolist()
        return texts, n_gen, finish, n_real, n_pad, wall

    def _sync(self):
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

    def mem_gb(self):
        if self.device == "mps":
            return round(torch.mps.driver_allocated_memory() / 1e9, 2)
        if self.device == "cuda":
            return round(torch.cuda.memory_allocated() / 1e9, 2)
        return None

    @torch.no_grad()
    def sequence_logprobs(self, prompt, continuations):
        """Teacher-forced log P(continuation | prompt) for each continuation, in ONE batch.

        This is the label-scoring mechanism (LMQL's `distribution`): the candidate STRINGS are
        scored by likelihood instead of being named through a symbol. It separates 'the model
        cannot bind a letter to an option' from 'the model does not know the answer'.
        Returns (sum_logprob, mean_per_token, n_tokens) per continuation -- length
        normalization is reported, never silently chosen, because options differ in length.

        Two things here are about speed, and both were measured, not guessed: casting the whole
        [B, L, V] logit block to fp32 allocated ~1.9 GB per batch, and summing the target
        logprobs in a Python loop synchronised the device on every `.item()`. Together they cost
        6.0 s per unit. Only the last max_cont+1 positions are needed, and the sum is a gather.
        """
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        seqs, cont_lens = [], []
        for c in continuations:
            c_ids = self.tok(c, add_special_tokens=False)["input_ids"]
            seqs.append(p_ids + c_ids)
            cont_lens.append(len(c_ids))
        width = max(len(s) for s in seqs)
        max_n = max(cont_lens)
        pad = self.tok.pad_token_id
        input_ids = torch.tensor([[pad] * (width - len(s)) + s for s in seqs], device=self.device)
        mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs], device=self.device)
        pos = position_ids_from_mask(mask)
        out = self.model(input_ids=input_ids, attention_mask=mask, position_ids=pos,
                         use_cache=False, logits_to_keep=max_n + 1)
        # logits_to_keep=max_n+1 gives positions [width-max_n-1 .. width-1]; the prediction for
        # input position t sits at t-1, so drop the last one and keep max_n prediction slots.
        lp = torch.log_softmax(out.logits[:, :-1, :].float(), dim=-1)      # [B, max_n, V]
        tgt = input_ids[:, width - max_n:]                                 # [B, max_n]
        got = lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)                  # [B, max_n]
        # each row's continuation occupies the LAST n slots of the max_n window
        idx = torch.arange(max_n, device=self.device).unsqueeze(0)
        n_t = torch.tensor(cont_lens, device=self.device).unsqueeze(1)
        keep = idx >= (max_n - n_t)
        sums = (got * keep).sum(1).tolist()
        return [(round(t, 5), round(t / max(n, 1), 5), n) for t, n in zip(sums, cont_lens)]

    def _sync(self):
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

    def mem_gb(self):
        if self.device == "mps":
            return round(torch.mps.driver_allocated_memory() / 1e9, 2)
        if self.device == "cuda":
            return round(torch.cuda.memory_allocated() / 1e9, 2)
        return None

    @torch.no_grad()
    def sequence_logprobs(self, prompt, continuations):
        """Teacher-forced log P(continuation | prompt) for each continuation, in ONE batch.

        This is the label-scoring mechanism (LMQL's `distribution`): the candidate STRINGS are
        scored by likelihood instead of being named through a symbol. It separates 'the model
        cannot bind a letter to an option' from 'the model does not know the answer'.
        Returns (sum_logprob, mean_per_token, n_tokens) per continuation -- length
        normalization is reported, never silently chosen, because options differ in length.
        """
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        seqs, cont_lens = [], []
        for c in continuations:
            c_ids = self.tok(c, add_special_tokens=False)["input_ids"]
            seqs.append(p_ids + c_ids)
            cont_lens.append(len(c_ids))
        width = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id
        input_ids = torch.tensor([[pad] * (width - len(s)) + s for s in seqs], device=self.device)
        mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs], device=self.device)
        pos = position_ids_from_mask(mask)
        out = self.model(input_ids=input_ids, attention_mask=mask, position_ids=pos, use_cache=False)
        logprobs = torch.log_softmax(out.logits.float(), dim=-1)
        res = []
        for i, n in enumerate(cont_lens):
            tot = 0.0
            for j in range(n):
                tgt_pos = width - n + j          # position of the continuation token
                tok_id = input_ids[i, tgt_pos].item()
                tot += float(logprobs[i, tgt_pos - 1, tok_id])
            res.append((round(tot, 5), round(tot / max(n, 1), 5), n))
        return res

    @torch.no_grad()
    def broadcast_last_hidden(self, prefix, suffixes):
        """ONE prefill of the shared prefix, its KV-cache broadcast to every suffix.

        This is the optimisation harshatheg's repo has and the per-field arm lacks: the text is
        prefilled once per request instead of once per field. Returns (h_last [B,H],
        prefix_tokens, suffix_tokens [B], wall_s). The tokenisation boundary is ASSERTED, not
        assumed: tokens(prefix)+tokens(suffix) must equal tokens(prefix+suffix), otherwise the
        cached prefix and the suffix do not compose into the prompt the per-field arm saw.
        """
        p_ids = self.tok(prefix, add_special_tokens=False)["input_ids"]
        s_ids = []
        for s in suffixes:
            ids = self.tok(s, add_special_tokens=False)["input_ids"]
            whole = self.tok(prefix + s, add_special_tokens=False)["input_ids"]
            if p_ids + ids != whole:
                raise AssertionError("tokenisation boundary is not clean: prefix+suffix tokens "
                                     f"!= whole ({len(p_ids)}+{len(ids)} vs {len(whole)})")
            s_ids.append(ids)
        P = len(p_ids); B = len(suffixes); S = max(len(x) for x in s_ids)
        pad = self.tok.pad_token_id
        t0 = time.perf_counter()
        pre = torch.tensor([p_ids], device=self.device)
        out = self.model.model(input_ids=pre, attention_mask=torch.ones_like(pre),
                               position_ids=torch.arange(P, device=self.device)[None], use_cache=True)
        cache = out.past_key_values
        cache.batch_repeat_interleave(B)
        suf = torch.tensor([x + [pad] * (S - len(x)) for x in s_ids], device=self.device)   # right pad
        smask = torch.tensor([[1] * len(x) + [0] * (S - len(x)) for x in s_ids], device=self.device)
        mask = torch.cat([torch.ones(B, P, dtype=smask.dtype, device=self.device), smask], dim=1)
        pos = (P + torch.arange(S, device=self.device))[None].expand(B, S)
        out2 = self.model.model(input_ids=suf, attention_mask=mask, position_ids=pos,
                                past_key_values=cache, use_cache=True)
        self._sync()
        wall = time.perf_counter() - t0
        last_idx = torch.tensor([len(x) - 1 for x in s_ids], device=self.device)
        h = out2.last_hidden_state[torch.arange(B, device=self.device), last_idx, :]
        return h, P, [len(x) for x in s_ids], wall
