# PREREG mini-jev-v1 — decision scorer (B1) vs constrained JSON (A1) on one Qwen

Frozen 2026-09-16 before the graded runs. Tag `prereg-v1`.

## 0. What was read BEFORE this freeze (stated, not hidden)

Pre-registered gates are meant to be read before freezing, and they were:

| read | value | why it was allowed |
|---|---|---|
| S3 gate, M5 letter emission, 4B, k=4, n=64 | 64/64 = 1.000 | S3 is the gate that decides whether arm B exists at all |
| S7 calibration | prefill 808.4 tok/s, decode 55.4 ms/step at batch 8, 9.4 GB | S7 decides the grid size before freezing |
| position histogram on those same 64 | {0:16, 1:22, 2:13, 3:13} | mechanism diagnostic attached to S3 |
| accuracy on those same 64 | 59/64 = 0.922 | **A PEEK AT A RESULT.** It is disclosed here rather than
| | | denied. Those 64 texts stay in the corpus; the primary
| | | numbers are pooled over 200 texts and 4 k-levels, and this
| | | slice is reported separately in RESULT so a reader can
| | | subtract it. It also discharges S8: 0.922 < 0.97. |

Smoke-model (Qwen3-0.6B) numbers were read freely and are never compared to 4B numbers:
M5 40/40, position histogram {0:37,1:1,2:1,3:1}, accuracy 12/40. The 0.6B has a near-total
letter-A prior; the 4B does not. That contrast is why M8 is kept as a first-class metric.

## 1. Decided / not decided

Decided: whether B1 is non-inferior to A1-joint in accuracy on runtime-built k-way schemas
(k ∈ {2,4,8,14} within domain, 16 mixed; fields ∈ {1,2,3}) on Qwen3-4B-Instruct-2507 in run
mode `hf-mps-greedy,bf16,attn=eager`; how the difference decomposes across the ladder; what
the letter/position prior is; the measured crossover f*.

Not decided: calibration of the probabilities; k > 26; prefix-shared B1 (KV reuse); vLLM or
production latency; any claim about another model.

## 2. Corpus

`clinc/clinc_oos` rev `155b9c710419136e17307b80d0a13e68cd46b4ec`, config `plus`, split `test`.
Domain map `clinc/oos-eval data/domains.json` sha256 `b947b579…1211b3a`, coverage 150/150.
200 in-scope texts, 20 per domain, seed 20260916; sample sha in `data/data_manifest.json`.
Same 200 texts in every condition. 50 oos texts for the paired abstention series.
Distractors WITHIN domain (hard) as primary; cross-domain (easy) as the second factor;
k=16 requires a cross-domain fill and is labelled `mixed`, never merged with the pure levels.

## 3. Metrics — {name, numerator, denominator, unit, matching rule}

M1 accuracy = correct / field-questions in the cell; exact string equality; a non-answer is
   WRONG, never dropped. Unit = one field-question.
M2 Δ = acc(B1) − acc(A1joint), paired by (condition, text_id, field); 95% cluster bootstrap
   over text_id, 2000 resamples, seed 20260916; discordant counts b and c printed.
M3/M4 A0 parse-fail and out-of-enum — DESCRIPTIVE rows, not a condition of the verdict.
M5 letter emission = argmax_class == bare_candidate / B1 records; printed per (k, fields) cell.
M6 A1 enum violations (must be 0) and mask-bite = row-steps where the mask changed the argmax
   / grammar row-steps. "The processor ran" is green by construction and is not counted.
M7 accuracy by gold position, per k.  M8 chosen-position share from the permutation arm on
   REAL texts (not from empty text).  M9 measured wall per request.
M10 prefill tokens, generated tokens, forward passes.  M11 run-to-run flip rate.
M12 acc(B1perm) − acc(B1).  M13 ladder decomposition Δ₁ = split − joint, Δ₂ = letter − split,
   Δ₃ = B1 − letter.  M14 A1letter/B1 agreement.
M15 model-based cost = prefill_tokens / 808.4 + own_decode_steps × 55.4 ms.
M16 crossover f* where f × prefill_B equals prefill_A + steps_A × step cost.

## 4. Positive result

P1 accuracy: primary unit is field `intent` pooled over k and f; lower 95% bound of Δ > −3.0 pp,
   and the intent cell at the largest pure k > −5.0 pp. `domain` and the boolean are secondary
   tables with balanced accuracy and are NOT in P1: they are near-trivial and would dilute the
   denominator. If the minimal detectable half-width from the observed discordance exceeds
   3.0 pp, the design cannot resolve the bound and the verdict is INCONCLUSIVE — printed, not
   worked around.
P2 cost: M15 ratio at f=1 ≤ 0.8 AND f* printed as a measured number. Measured wall is secondary
   because this bench inflates A1 by batch drag and by moving the mask to CPU each step.
P3 mechanism: M5 ≥ 0.95 pooled, M6 violations = 0, M14 ≥ 0.99, text-answer control mismatch ≤ 2%.

Noise band: r2 repeats B1 and A1joint on the same units under the SECOND frozen batch order
(permutation seeded SEED+1), so padding and batch composition differ. A repeat under the same
order would give a flip rate of 0 by construction and pass for any verdict.

## 5. STOP

S1 duplicate join keys. S2 A1 enum violations > 0 → A1 numbers not read. S3 M5 < 0.90 on the
first 64 4B requests → one pre-registered repair (prefill `Answer:`, read SPACE letters as
primary), re-measure; still below → "mechanism did not fire", no accuracy claim. **Discharged:
64/64.** S4 random-gold control outside 1/k ± 3 SE. S5 no-gold control: A1 in-enum < 100%.
S6 MPS OOM or NaN. S7 calibration < 250 tok/s → shrink the grid before freezing. **Discharged:
808.4.** S8 pooled A0 intent accuracy at the largest k ≥ 0.97 → grid declared "does not
discriminate". S9 A1 `finish=length` share > 1% → grammar configuration defect, not a result.
S10 fp32 tie rate > 0.5%.

## 6. Not claimed

Probability calibration; k > 26; vLLM or production latency; other models; anything carried
over from the 0.6B. A repair after this freeze requires tag `prereg-v1.1` and a list of the
post-tag diffs in RESULT.md.

---

# AMENDMENT prereg-v1.1 (2026-09-16, after adversarial review of the ladder)

Written after B1/main was complete (6000 records) and A1joint/main was 256/3000. No accuracy
number of any A arm had been read. Three corrections, all because a metric could not have
returned a negative answer.

## A. M14 is an identity, not a validation — relabelled

A1letter and B1 compute the SAME quantity by construction: same rendered prompt, same position
(the first assistant token), the letter grammar admits exactly the k bare-letter ids, greedy
takes the argmax over those ids, B1 takes the argmax over the same ids. So M14 ≈ 1.0 is an
identity plus numerics, and **Δ₃ ("generate vs read") is zero by construction, not a measured
term.** M14 is kept and renamed **the position/id harness check**: it does catch a wrong padding
side, a wrong last position, and a wrong letter-id table (xgrammar maps the string "A" to ids
independently of `letters.py`). It cannot catch a letter prior, binding failure, a verbose or
newline-first answer, or a shared renderer bug. The plan's earlier wording claimed more.

## B. New rung A1label — the symbol-binding number is Δ₂b, not Δ₂

Δ₂ ("label vs letter") changed two things at once: the prompt family (JSON instruction vs MCQ)
AND the output symbol. New arm **A1label**: the SAME MCQ prompt as B1, instruction "reply with
the exact text of the chosen option", grammar `root ::= "<opt1>" | "<opt2>" | …` over the option
strings (verified: 13 legal tokens at step 1 on a 3-option set, all valid prefixes). Then
Δ₂a = A1split → A1label is the prompt effect, and **Δ₂b = A1label → B1 is the pure cost of
answering through a symbol** — the binding diagnostic. Δ₂b is the pre-registered number.

## C. New control B1free — the operand outside the logit read

Unconstrained `generate` (≤ 8 tokens) under the B prompt, letter parsed from the text, compared
with B1's argmax letter. This is the only arm that can say "the model would not have answered
with this letter". No grammar by construction.

## D. Measured, not assumed: the noise band and the duplication

The f axis re-renders the same per-field prompt (the option RNG key excludes the field
condition), so **60.2% of arm B1 is re-computation** (2388 unique (prompt_sha, field) among
6000 records). Those duplicates land in different batches, which makes them a free
batch-composition control, and it is declared as one:

| measured on r1/B1, 3612 duplicate pairs | value |
|---|---|
| candidate logits bit-identical | 2790/3612 = 0.772 |
| differ by more than one bf16 step (0.125) | 787/3612 = 0.218 |
| p99 / max absolute difference | 1.26 / 4.79 |
| **groups whose chosen position disagrees** | **3/2329 = 0.0013** |

Mechanism: left padding differs with batch composition (same 201 real tokens, pad 7 vs pad 0),
and eager attention on MPS is not numerically invariant to it. **The noise floor for any paired
Δ is therefore 0.13%, measured.** Per-field arms run with `--unique-prompts` from here on; the f
axis is meaningful for A1joint only, and the per-field cost at f>1 is f × the per-field cost
arithmetically.

## E. Unchanged

P1, P2, P3, the STOP rules and the corpus are unchanged. S3 and S7 stay discharged
(64/64; 808.4 tok/s). Post-tag diffs are listed in RESULT.md.

---

# AMENDMENT prereg-v1.2 (2026-09-16, after external review)

Written after an external reviewer read the protocol. Six changes; three of them are the
reviewer's findings, not ours. No arm's accuracy had been read on the new corpus.

## A. The claim is renamed — "generate vs read" was causally wrong

A1letter → B1 is an identity (already recorded in v1.1). A1label → B1 changes TWO things at
once: string labels vs a symbolic verbalizer, AND autoregressive greedy vs one-step scoring.
The question it answers is therefore **"does direct verbalizer scoring preserve classification
quality compared with constrained label generation?"** — not "generate vs read".
Cost of read-vs-generate is measured on **A1letter ↔ B1**, where quality is an identity by
construction and only latency differs. End-to-end quality is **A1label ↔ B1**. B1score isolates
the third question: constrained greedy decoding of a label string vs scoring the full candidate
sequence.

## B. Candidate mass is now recorded — the normalized numbers were under-described

softmax over k candidates is P(option | the answer is one of the options). It says nothing
about how much mass the model put on the candidates at all: a set can hold the top token and
still carry a small fraction of the distribution, and the normalized numbers look confident
anyway. Every B1 record now carries `logZ`, `candidate_mass` and `candidate_mass_with_space`.
Numerator and denominator are read from ONE distribution — the model's own head. The first
implementation mixed the fp32 recompute into the numerator and produced **mass = 1.0192**,
i.e. a probability above 1, which is how the mismatch announced itself; an assert now holds it.
Measured immediately on the smoke model: median mass 0.9913 but **minimum 0.0158**.
The outputs are called **normalized candidate scores**, never "confidence that the answer is correct".

## C. Off-by-one in k — the pure within-domain ceiling is 15, not 14

Each CLINC domain holds 15 intents, so gold + all 14 same-domain distractors is k=15. K_LEVELS
was (2,4,8,14), which left one distractor unused for no stated reason. Now (2,4,8,15); k=16
remains the mixed row.

## D. The corpus covers every intent, and is larger

Stratification moved from 10 domains to **150 intents, 3 texts each, N=450** (domain balance
follows for free; the reverse does not). The experiment is about label sets assembled at request
time, so coverage of the label space is the thing that had to be guaranteed. Narrower intervals
are a side effect and NOT the reason: the −3.0 pp margin is unchanged, and it stays a
non-inferiority decision about acceptable quality loss, which is a different kind of quantity
from the 0.13% measurement noise floor. Tying the margin to the noise floor is explicitly
rejected. An inconclusive verdict stays inconclusive.

## E. Distribution stability, not only answer flips

B1 is sold as a distribution, so counting argmax flips understates its instability. Measured on
the duplicate prompts already collected (total variation between the normalized candidate
distributions):

| | within one host | across hosts (Mac ↔ GPU) |
|---|---|---|
| median TV | 0.00000 | 0.00000 |
| p99 TV | 0.00040 | 0.20191 |
| max TV | 0.552 | 0.882 |
| share with TV > 0.05 | 0.25% | 2.6% |
| answer flips | 0.17% | 0.42% |

The reviewer's common-mode hypothesis holds for the bulk and fails in the tail: the share of
pairs whose distribution moves materially is 2–6× the share whose answer flips. Both are now
reported; the flip rate alone is not called "the noise band".

## F. Cost is now measured against INPUT LENGTH

CLINC utterances are ~15 tokens — the case most favourable to repeating the prefill per field.
`scripts/run_cost.py` sweeps 32 / 128 / 512 / 2048 / 4096 input tokens × 1,2,3 fields and reports
wall and prefill ratios. Accuracy is not measured there and must not be read from it. Batching
fields into one call is one batched CALL, not one forward: the work still scales with the number
of sequences, and the protocol no longer implies otherwise.

## G. Also recorded

`pred_pos_native_bf16` and `native_matches_fp32` per B1 record: how often the model's own bf16
head and our fp32 recompute disagree on the winner. B1perm now reports three things separately:
the chosen-position histogram, accuracy by true-answer position, and agreement across shifts.
The dependent fields (`domain`, the boolean) are named a **dependency stress test**, not a
second classification task.

## Unchanged

P1/P2/P3 thresholds, the STOP rules, the ladder, the controls, and the refusal to claim
calibration. What the experiment may claim at the end: a frozen causal LM can consume an
arbitrary per-request label schema and classify by scoring single-token verbalizers without
emitting a token; we measure what that costs in quality against constrained structured
generation, and where the compute crossover sits. It may not claim to have reproduced Jev.

---

# Amendment v1.3 — 2026-09-17 evening, BEFORE reading run b4: a harness defect in every generation arm

## What was found, and how

The demo bench (`demo/`) produced `"callback_phone": "55555555555555555555"` and
`"people_waiting": 4555555…` for a text that says "555-0102" and "4 more people". A two-sided
placebo (`scripts/smoke_positions.py`) localised it: `Engine.generate` passed an explicit
`position_ids` tensor for the prompt, and transformers 4.57 does not extend a caller-supplied
tensor across decode steps — `prepare_inputs_for_generation` slices its last column for every new
token, `_update_model_kwargs_for_generation` extends `attention_mask` and `cache_position` only.
Every generated token therefore sat at the LAST PROMPT POSITION. Without the tensor the same
prompt gives `{"topic": "bleeding", "callback_phone": "555-0102", "people_waiting": 4}` in 29 tokens;
with it, a 96-token loop. The first generated token is bit-identical either way (the prefill
positions are correct in both) — measured on the Mac, batch 1.

This is the class "the guard checked the side it could see": positions were asserted explicit
*for prefill* and celebrated as a virtue in the module docstring; the decode path was never
given an input on which the convention was forced to fail.

## Which numbers are affected

| unaffected (first-token or no-generation arms) | under review (multi-token generation) |
|---|---|
| B1, B1cache, B1perm (logit reads); A1letter (one token); B1score (teacher-forced full forward with correct positions); the letter read from B1free; all noise bands; the B1cache gate | A1joint (all subsets), A1split, A1label, A1prob (all three attempts), A0, the joint arm of the cost sweep, B1free beyond its first token |

Claims at risk, named before b4 is read: (1) **symbol vs label, +10.00 pp** — A1label wrote
multi-token names under frozen positions; (2) the TypeSafe-form verdict and the "number without a
bound loops" / "≈25 tokens per key" explanations — both may be the defect, not the schema;
(3) the cost table's joint-JSON column (token counts and whitespace under frozen positions);
(4) direct P1 −0.22 pp — probably robust (intent is the first field, positions off by a constant
few), but re-measured, not assumed; (5) joint-vs-split conditioning on the dependent fields.

## Decision rules, written now

- **Symbol vs label** survives only if corrected A1label still trails A1letter with a 95 % CI
  excluding zero on the same 5369 units; the length mechanism is then re-checked on the corrected
  arm (chosen-vs-gold token length) and remains supported independently by B1score, which never
  generated. If the deficit vanishes, the note says "the letter advantage was a harness artefact"
  and the length story is retracted for A1label and kept only for sequence likelihood.
- **TypeSafe form**: re-run at cap 640, stride 4, corrected engine. It is judged on finish
  reasons first (if `length` reappears, the schema explanation is back on the table), then on
  accuracy against B1 on the same units.
- **P1 direct**: the b4 A1joint main replaces b2's in §7/§8; parity is claimed only from the
  corrected number.
- **Cost**: the b4 rows replace b2/b3 rows for all three arms in one table with one engine.
- Everything in RESULT/NOTE that depends on a generation arm is marked "under review, b4" until
  replaced; retired numbers are kept in RESULT §9 with the reason.

## Re-run (b4, corrected engine, RTX 4090): gate first

`smoke_positions.py` must print `ENGINE_OK | PLACEBO_LOOPS` on the box or the program stops
(a check that cannot fail on the host proves nothing). Then A1joint main/easy(stride 4)/oos,
A1label main, A1split main, B1free, A0 (stride 7), cost 32/128/512 and 2048 (batches 4/2),
A1prob (cap 640, stride 4). Unaffected arms are not re-run; their b2 records stand.
