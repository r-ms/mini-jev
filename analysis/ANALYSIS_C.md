# Analysis C — two questions on mini-jev run data (runs/b1 RTX 4090, runs/r1 Mac MPS)

Read-only analysis, 2026-09-17. Scripts and raw outputs: `q1.py` (voided first pass), `q1b.py`, `q1c.py`, `q2.py`, `q2b.py` and their `*_out.txt` in this directory. All joins below were checked by byte-identical `prompt_sha` before any number was read.

---

## QUESTION 1 — the free-text control (B1free vs the logit read)

### Finding 0 (changes the question): runs/b1 has NO logit read of the control prompts in `B1.jsonl`

- `runs/b1/run.log` 20:14:08: `scripts/run.py --arm B1 --subset main --run-id b1` crashed after **607 / 13 500** records (`AssertionError: candidate mass 1.0000038 > 1: mixed distributions again`) and was never resumed. The later `oos` run confirms it: `"already": 607`.
- Conditions present in `runs/b1/B1.jsonl` (11 547 records): `easy:k{2,4,8,15}:f{1,2,3}` 10 800, `main:k15/k16` partial 607, `oos:*` 100, `ctl:nogold` 40. **`main:k2/k4/k8` do not exist.** Main-grid letter numbers on b1 come from `A1letter` (greedy under a letter grammar, `--unique-prompts`) and `B1score` (label likelihood).
- The `ctl:textanswer` units draw their options with the same RNG key as `main:k4:f1` (default policy `within_domain`), so their logit-read twin would have been `main:k4:f1` — which is absent on b1 and present on r1.

### The published 189/200 is a wrong-prompt join, and it reproduces exactly

| pairing tried on b1 | agreement |
|---|---|
| lead's recipe: first `B1.jsonl` record by (text_id, intent, k=4) — lands on `easy:k4:f1` (150) / `oos:in_scope` (50) | 190/200 |
| last `B1.jsonl` record by the same key — lands on `easy:k4:f3` (200) | 189/200 |
| `B1 easy:k4:f1` by text_id | **189/200** |
| `B1 easy:k4:f2` / `f3` by text_id (other k=4 records for the same texts) | 29/200, 0/200 |
| join control: options identical / prompt_sha identical, any of the above | **0/200 / 0/200** |

Anatomy of 189: B1free correct on its own hard within-domain set **189/200**; the easy-set logit read correct on the same texts **198/200**; both correct **189/200**; same wrong label **0/200** → "agreement" = 189. The gold letter position differs between the two option sets on 153/200 texts; the two option lists share only the gold label on 193/200. **0.945 is B1free's accuracy on the hard k=4 set, not an agreement rate.** A1letter on the byte-identical prompts is also 189/200 correct.

### Valid same-prompt comparisons (byte-identical `prompt_sha`)

| comparison | run | n | agreement |
|---|---|---|---|
| B1free vs A1letter (greedy under letter grammar) | b1 | 200 | **199/200 = 0.995** |
| B1free vs B1perm shift0 (the only TRUE logit read of these prompts on b1) | b1 | 50 | **49/50 = 0.980** |
| A1letter vs B1perm shift0 | b1 | 50 | 50/50 |
| A1letter vs any b1 logit read with identical prompt (B1 partial main + B1perm), all fields | b1 | 1156 | 1155/1156 = 0.9991 (one: clinc-002502 k16, gap 4.9) |
| B1free vs B1 `main:k4:f1` | r1 | 200 | 200/200 (prompt_sha identical 200/200) |
| A1letter vs B1 | r1 | 912 | 912/912 |
| B1score (label likelihood) vs B1free — different mechanism, listed for completeness | b1 | 200 | 108/200 |

The pre-registered bar (mismatch ≤ 2%) is met on every same-prompt comparison: 1/200 = 0.5 % against A1letter, 1/50 = 2.0 % against the true logit read.

### The one genuine disagreement (class a: a different valid letter)

clinc-000404 `is my replacement card due to arrive in the mail today`; A=redeem_rewards, B=replacement_card_duration (gold), C=expiration_date, D=damaged_card.
- B1free raw `'A'` → redeem_rewards (batch of 8, batch_id 7, pad 0); A1letter raw `'D'` → damaged_card; B1perm shift0 logit read → damaged_card, `cand_logits_bare=[53.48, 36.39, 35.73, 57.63]`, **gap 4.15**, p_cand D=0.984, `native_matches_fp32=True`, candidate_mass 1.0.
- Under option rotation (B1perm k4, 4 shifts) this text's logit-read choice flips (`damaged_card, replacement_card_duration, damaged_card, damaged_card`); 3/50 control texts flip under rotation.
- Neither answer is correct. Malformed/verbose (class b): **0/200** on both hosts — every raw_response is a single capital letter, generated_tokens 2, finish `eos`, state `ok`.

Gap on the 50 units with a true same-prompt logit read: disagreement gap **4.15**; agreements n=49 min 8.42, q1 27.40, median 28.83, q3 30.56, max 33.42. candidate_mass is 1.0 on all 200 (saturated — it cannot distinguish anything here). The first-pass "gap on the 11" (median 25.3 vs 28.1) was measured on the wrong prompt's logits and is void.

The 11 "disagreements" of the published number = B1free's 11 errors; A1letter gives the same wrong letter on 10/11 (list with texts and options in `q1d_out.txt`).

### Cross-host (r1 Mac vs b1 4090), byte-identical prompts only

| quantity | value |
|---|---|
| logit-read prompts common to both hosts (prompt_sha identical) | 58 (by k: 2→14, 4→21, 8→9, 15→3, 16→11) |
| chosen option changed between hosts | **0/58** |
| max \|candidate logit diff\| per unit | min 0.043, q1 0.26, median 0.385, q3 0.67, max 1.11 |
| B1free prompts common to both hosts | 12; free answer changed **0/12** |
| of the 11 published "disagreements", present in r1 | 1 (clinc-001450) — in r1 it was a valid join and agreed |

The earlier cross-host figure "B1 pred changed 22/209" in the first pass paired records with different options (only 43/209 identical prompts) and is void.

### Conclusion Q1

0.945 is a harness problem, specifically a missing-comparator join: on b1 the main-grid logit read crashed at 607 records, so any join of the control by (text_id, intent, k=4) into `B1.jsonl` silently landed on the easy-set (cross-domain) records — a different prompt with a different letter for the gold — and the resulting "agreement" is arithmetically B1free's own accuracy (189/200 correct, 198/200 easy-read correct, both 189). It is not a numerics-across-hosts effect: on the 58 byte-identical logit-read prompts shared by both hosts the chosen option never changed and candidate logits moved by 0.4 median, 1.1 max; the 12 shared control prompts gave identical free answers; r1 reached 200/200 only because its `main:k4:f1` records exist and the join was valid there. Against byte-identical prompts on b1 the control passes the bar: 199/200 against A1letter and 49/50 against the one true logit read available, and the single genuine disagreement is a real near-tie property of the model on that text (gap 4.1 vs an agreement median of 28.8; rotation-unstable), amplified by batch-composition numerics of the kind PREREG §D already measured. RESULT.md §1 line 15 and the two sentences after it (the two quoted sentences about "5.5 % of cases" and "the numerical side") should be replaced by the same-prompt numbers, and the b1 report should state that the main-grid B1 logit read is absent for k=2/4/8.

---

## QUESTION 2 — joint vs separate on the dependent fields (runs/b1 main)

### Denominators

- A1joint main: 6 750 records, all `state=ok` → unit rows intent 6 750, domain 4 500 (f2 2 250 + f3 2 250), boolean 2 250 (f3).
- A1split main (`--unique-prompts`): 4 950 records, all ok → intent 2 250 (f1), domain 2 250 (f2), boolean **450** (f3, k=2 only: the boolean prompt renders `true | false (JSON booleans)` and is byte-identical across k, so one copy per text; 450 distinct prompt_sha, 450 texts).
- Joins by (text_id, field, k): split has exactly one row per key (0 duplicates); joint has one row per (key, field-condition), reported separately.

### Paired accuracy, delta = acc(split) − acc(joint), cluster bootstrap over text_id, 20 000 resamples

**domain, joint@f2 vs split** — n = 2 250; joint 1 684/2 250 = 0.748; split 1 563/2 250 = 0.695; **Δ −5.38 pp [−7.47, −3.33]**; b (split right, joint wrong) = 115, c (joint right, split wrong) = 236.

| k | n | joint | split | Δ split−joint | b / c |
|---|---|---|---|---|---|
| 2 | 450 | 408 (0.907) | 404 (0.898) | −0.89 [−3.78, +2.00] | 21 / 25 |
| 4 | 450 | 369 (0.820) | 349 (0.776) | −4.44 [−8.00, −1.11] | 21 / 41 |
| 8 | 450 | 301 (0.669) | 276 (0.613) | −5.56 [−9.56, −1.56] | 31 / 56 |
| 15 | 450 | 300 (0.667) | 267 (0.593) | −7.33 [−11.33, −3.56] | 25 / 58 |
| 16 | 450 | 306 (0.680) | 267 (0.593) | −8.67 [−12.44, −5.11] | 17 / 56 |

**domain, joint@f3 vs split** — n = 2 250; joint 1 694/2 250 = 0.753; **Δ −5.82 pp [−8.18, −3.51]**; b = 144, c = 275; per k: −0.22, −5.11, −7.11, −8.89, −7.78 pp (k = 2, 4, 8, 15, 16).

**is_banking_or_credit_cards, joint@f3 vs split, strict same-k pairing (k=2)** — n = 450; joint 431/450 = 0.958; split 407/450 = 0.904; **Δ −5.33 pp [−7.56, −3.11]**; b = 2, c = 26. Gold balance true 90/450. **Balanced accuracy joint 0.903, split 0.765.** gold=true: joint 73/90 = 0.811, split 48/90 = 0.533; gold=false: joint 358/360, split 359/360.

Extended pairing (the split answer to the identical boolean prompt reused per text against joint at every k; clusters = text_id): n = 2 250; joint 2 171/2 250 = 0.965, split 2 035/2 250 = 0.904, Δ −6.04 pp [−8.40, −3.78], b = 20, c = 156; balanced 0.932 vs 0.765; joint true-recall rises with k: 0.811 (k2), 0.856, 0.878, 0.911, 0.933 (k16) while split stays 0.533.

### Cross-field cue inside the joint record

| stratum | domain acc (joint) |
|---|---|
| own intent RIGHT | 3 088/4 090 = 0.755 (f2 0.754, f3 0.756) |
| own intent WRONG | 290/410 = 0.707 (f2 0.691, f3 0.724) |

| stratum | boolean acc (joint, f3) |
|---|---|
| own domain RIGHT | 1 660/1 694 = 0.980 |
| own domain WRONG | 511/556 = 0.919 |
| own intent RIGHT / WRONG | 1 976/2 047 = 0.965 / 195/203 = 0.961 |

Split arm stratified by its own independent intent request (text-difficulty baseline, no conditioning): domain acc 1 431/2 047 = 0.699 (intent right) vs 132/203 = 0.650 (intent wrong); boolean 373/404 = 0.923 (domain right) vs 34/46 = 0.739 (domain wrong).

Decomposition of the paired domain delta by the joint's own intent correctness (joint@f2 vs split): intent-right stratum 2 043 pairs (90.8 %): joint 0.754 vs split 0.700 → −4.89 pp of the −5.38; intent-wrong stratum 207 pairs: joint 0.691 vs split 0.638 → −0.49 pp. The joint edge is ~5 pp in both strata. Boolean (joint@f3 vs split, n=450) by own domain: domain-right 405 pairs: joint 0.968 vs split 0.909 → the whole −5.33 pp; domain-wrong 45 pairs: 0.867 vs 0.867 → 0.00 pp.

### Is the joint arm reading its own earlier answer? Self-consistency

Under the within-domain policy every intent option list at k ≤ 15 is single-domain (1 350/1 350 records per k; k=16: 0/1 350), so `domain(own intent answer) == gold` on **2 248/2 250** for both arms — copying one's own answer would score ~100 %. Measured: joint `pred_domain == domain(pred_intent)` **1 682/2 250 = 0.748** (f2; f3 0.752) — identical to its domain accuracy by construction; split cross-request consistency 1 563/2 250 = 0.695. Boolean: joint `pred_bool == f(pred_domain)` 2 174/2 250 = 0.966 (f(pred_domain)==gold 0.961); split cross-request 416/450 = 0.924.

### The cue isolated: A1joint MAIN vs A1joint EASY on the same (text, k, field-condition) — same model, same text, same domain/boolean question; only the intent list differs (within-domain leaks the domain; cross-domain does not)

| field | n | easy (no leak) | main (leak) | Δ main−easy | b / c |
|---|---|---|---|---|---|
| domain | 900 | 647/900 = 0.719 | 684/900 = 0.760 | **+4.11 pp [+1.11, +7.22]** | 108 / 71 |
| domain per k=2/4/8/15 | 225 each | 0.876 / 0.778 / 0.613 / 0.609 | 0.907 / 0.804 / 0.667 / 0.662 | +3.1 / +2.7 / +5.3 / +5.3 (each CI crosses 0) | |
| is_banking_or_credit_cards | 450 | 432/450 = 0.960 | 435/450 = 0.967 | +0.67 pp [−0.67, +2.22] | 8 / 5 |

Joint-EASY vs split on the same (text, k) — the joint arm without the leak against separate requests: domain f2 +2.00 pp [−2.22, +6.22] (n 450), f3 −2.00 pp [−6.44, +2.22]; boolean **−5.78 pp [−9.56, −2.22]** (joint-easy 432/450 = 0.960 vs split 406/450 = 0.902, b 7, c 33).

Does the domain follow the model's own WRONG intent? Easy records where `domain(pred_intent) != gold domain`: 15 → followed own answer 2, answered gold 10, neither 3. Main k=16 (one cross-domain intent in the list): 4 such records → followed 0, gold 4.

### Conclusion Q2

The joint arm beats separate requests on both dependent fields by about the same amount (domain −5.4/−5.8 pp, boolean −5.3/−6.0 pp for split−joint, all CIs excluding zero from k=4 up; at k=2 the domain delta is ~0), but the two advantages have different sources and only one of them is "conditioning on its own earlier answer". For `domain` the advantage is a property of the prompt, not of the sequential answer: it is the same size whether the joint's intent answer was right (−4.9 of −5.4 pp comes from that 91 % stratum) or wrong; the joint answers a domain inconsistent with the one implied by its own intent in 25 % of records even though that implied domain is gold 99.9 % of the time under within-domain distractors; when its own intent implies a wrong domain it follows it in 2/15 (easy) and 0/4 (k=16) cases; and swapping only the intent option list from within-domain to cross-domain on the same texts removes 4.1 pp [1.1, 7.2] of the gain, after which the joint arm is indistinguishable from split (+2.0 / −2.0 pp, CIs ±4). So essentially all of the domain edge is the 15 same-domain intent names sitting above the domain question — a leak the split arm's domain request never sees — and the residual attributable to reading its own answer is within ±4 pp of zero. For the boolean the picture inverts: the leak contributes nothing (main vs easy +0.7 pp [−0.7, +2.2]), the edge survives against split without it (−5.8 pp [−9.6, −2.2]), it lives entirely in pairs where the joint's own domain answer was right (−5.33 of −5.33 pp; 0.00 when that answer was wrong), and the boolean agrees with f(own domain) 96.6 % of the time. The mechanism there is recall on the 20 % positives: asked alone, the model answers "false" on half of the banking/credit-card texts (48/90), while with its domain answer already written in the same JSON it answers "true" 73–84/90, so balanced accuracy goes 0.77 → 0.90–0.95. That boolean gain is explained by conditioning on the earlier answer, and it inherits that answer's errors (0.87 when own domain is wrong, same as split).

---
# APPENDIX A — q1b_out.txt (same-prompt comparators, gap, cross-host)
A1letter: records 5369, distinct prompt_sha 5369, shas with >1 record 0
B1score: records 5369, distinct prompt_sha 5369, shas with >1 record 0
B1perm shift0: records 50, distinct prompt_sha 50, shas with >1 record 0

## Q1 corrected: on runs/b1 NO B1 logit read exists for the main:k4 prompts (B1 main crashed at 607/13500 records, run.log 20:14:38 AssertionError "candidate mass > 1"; never resumed).
## Same-prompt comparators on b1 (byte-identical prompt_sha): A1letter (greedy under letter grammar, 200), B1score (label likelihood, 200), B1perm shift0 (TRUE logit read, 50).
B1free vs A1letter (same prompt): agree 199/200 = 0.9950
B1free vs B1score (same prompt): agree 108/200 = 0.5400
B1free vs B1perm shift0 (same prompt, TRUE logit read): agree 49/50 = 0.9800
A1letter vs B1perm shift0 (M14 on the 50 same-prompt units): agree 50/50

### Reproduction of the published 189/200: which pairing gives it?
  join to FIRST B1.jsonl record by (text_id,intent,4) [options differ in 200/200]: 190/200
  join to LAST B1.jsonl record by (text_id,intent,4): 189/200
  join to B1 easy:k4:f1 by text_id [different distractor policy = different options]: 189/200
  join to B1 easy:k4:f2 by text_id [different distractor policy = different options]: 29/200
  join to B1 easy:k4:f3 by text_id [different distractor policy = different options]: 0/200
  compare LETTER POSITION only vs first B1 record: 42/200
  paired() on (condition,text_id,field): B1 has condition ctl:textanswer? False -> 0 pairs

## Disagreements B1free vs A1letter on b1: 1/200
--- #1 clinc-000404  TEXT='is my replacement card due to arrive in the mail today'
    OPTIONS: A=redeem_rewards | B=replacement_card_duration | C=expiration_date | D=damaged_card   gold=replacement_card_duration (B)
    B1free raw='A' -> 'redeem_rewards' (correct=False) | A1letter raw='D' -> 'damaged_card' (correct=False) allowed_step1=[4, 4, 4, 4, 4, 4, 4, 4] batch(free)=8/7 batch(letter)=8/357 pad free=0 letter=1
    B1perm shift0 (logit read): pred='damaged_card' gap=4.145687 logits=[53.48, 36.39, 35.73, 57.63] native_matches_fp32=True p_cand=[0.015586, 0.0, 0.0, 0.984414]
    B1score (likelihood): pred='replacement_card_duration'

## Disagreements B1free vs B1perm shift0 (true logit read) on b1: 1/50
--- #1 clinc-000404 TEXT='is my replacement card due to arrive in the mail today'
    OPTIONS: A=redeem_rewards | B=replacement_card_duration | C=expiration_date | D=damaged_card   gold=replacement_card_duration
    free raw='A' -> 'redeem_rewards'; logit pred='damaged_card' gap=4.145687 logits=[53.48, 36.39, 35.73, 57.63] native_matches_fp32=True; A1letter -> 'damaged_card'

## Gap analysis on the 50 units with a true same-prompt logit read (B1perm shift0)
  free==logit: 49/50; disagree gap: [4.146]; agree gap quartiles: {'n': 49, 'min': 8.418, 'q1': 27.401, 'med': 28.828, 'q3': 30.559, 'max': 33.415}
  free==A1letter on these 50: 49/50; gap of the free!=A1letter units: [4.146]; gap quartiles where free==A1letter: {'n': 49, 'min': 8.418, 'q1': 27.401, 'med': 28.828, 'q3': 30.559, 'max': 33.415}
  native_matches_fp32 on the 50: Counter({True: 50})
  rotation stability (B1perm k4, 4 shifts) for the free!=A1letter units among the 50:
    clinc-000404: preds by shift ['damaged_card', 'replacement_card_duration', 'damaged_card', 'damaged_card']; free=redeem_rewards; A1letter=damaged_card
  texts whose logit-read choice changes under rotation (k4, all 50): 3/50

## Cross-host (r1 = Mac/MPS) restricted to BYTE-IDENTICAL prompts
B1 logit-read prompts common to both hosts (identical prompt_sha): 58 (r1 distinct 2388, b1 distinct 5394)
  chosen option changed between hosts: 0/58
  gap(b1) changed: []; gap(r1) changed: []
  gap(b1) unchanged quartiles: {'n': 58, 'min': 0.685, 'q1': 21.576, 'med': 25.662, 'q3': 27.921, 'max': 33.389}
  by k: {2: (0, 14), 4: (0, 21), 8: (0, 9), 15: (0, 3), 16: (0, 11)}
  by b1 condition: Counter({'easy:k4:f3': 18, 'easy:k2:f3': 14, 'main:k16:f1': 11, 'easy:k8:f3': 6, 'perm:k4:shift0': 3, 'easy:k15:f3': 3, 'perm:k8:shift0': 3})
  max |candidate logit diff| across hosts, same prompt: quartiles {'n': 58, 'min': 0.043, 'q1': 0.263, 'med': 0.385, 'q3': 0.67, 'max': 1.113}
B1free prompts common to both hosts: 12; free answer changed: 0/12
r1: B1free vs B1 same prompt: 200/200; r1 B1 gap quartiles on these: {'n': 200, 'min': 0.296, 'q1': 25.179, 'med': 27.896, 'q3': 29.773, 'max': 35.72}; units with gap<1: 1
r1: A1letter vs B1 same prompt (M14 on Mac): 502/912
b1: A1letter vs B1/B1perm same prompt (M14 on 4090): 303/1156; gaps of disagreements: [<list of 853 stale numbers from the buggy field join, removed>]; by condition: Counter({'easy:k2:f3': 373, 'easy:k4:f3': 271, 'easy:k8:f3': 135, 'easy:k15:f3': 64, 'easy:k2:f2': 9, 'main:k16:f1': 1})

# APPENDIX B — q1c_out.txt (M14 recomputed, anatomy of 189)
## M14 recomputed with each record's OWN field (previous 303/1156 was a bug: helper read "intent" on domain/boolean records)
b1: A1letter vs B1/B1perm, byte-identical prompt: 1155/1156 = 0.9991; by field: Counter({'is_banking_or_credit_cards': 821, 'intent': 304, 'domain': 31})
   disagreements: [('clinc-002502', 'intent', 'main:k16:f1', 'schedule_meeting', 'meeting_schedule', 4.912)]
r1: A1letter vs B1, byte-identical prompt: 912/912 = 1.0000; disagreements gaps: []

## Anatomy of the published 189/200
  condition of the FIRST B1.jsonl k=4 intent record per control text: Counter({'easy:k4:f1': 150, 'oos:in_scope': 50})
  condition of the LAST  B1.jsonl k=4 intent record per control text: Counter({'easy:k4:f3': 200})
  B1free correct (hard within-domain k4 set): 189/200; B1 easy:k4:f1 correct (cross-domain set, same texts): 198/200; both correct: 189/200; same wrong label: 0/200 -> "agreement" 189/200
  gold letter position differs between the two option sets on 153/200 texts; options lists share only the gold label on 193/200
  A1letter correct on the same 200 prompts: 189/200; B1free correct: 189/200

# APPENDIX C — q1d_out.txt (the 11 B1free errors)
## The 11 texts the published number counted as "disagreements" = B1free's own errors on the hard k=4 set: 11/200
--- #1 clinc-000295  TEXT='i have a revolving store card and defaulted so will my fico score be affected'
    OPTIONS(hard, the prompt B1free saw): A=application_status | B=expiration_date | C=credit_limit_change | D=improve_credit_score  gold=improve_credit_score (D)
    B1free raw='A' -> application_status   | A1letter (same prompt, grammar) raw='A' -> application_status  SAME
    true logit read of the same prompt (B1perm shift0): pred=application_status gap=17.561 logits=[61.39, 32.79, 35.57, 43.83]
    what the published join compared it with: B1 easy:k4:f1, options=['interest_rate', 'oil_change_when', 'improve_credit_score', 'timezone'], pred=improve_credit_score (correct=True), gap=27.96
--- #2 clinc-000650  TEXT='tell me what you can do for me'
    OPTIONS(hard, the prompt B1free saw): A=thank_you | B=do_you_have_pets | C=what_can_i_ask_you | D=fun_fact  gold=what_can_i_ask_you (C)
    B1free raw='A' -> thank_you   | A1letter (same prompt, grammar) raw='A' -> thank_you  SAME
    true logit read of the same prompt (B1perm shift0): none exists on b1 (B1 main crashed)
    what the published join compared it with: B1 easy:k4:f1, options=['account_blocked', 'last_maintenance', 'international_visa', 'what_can_i_ask_you'], pred=what_can_i_ask_you (correct=True), gap=29.15
--- #3 clinc-001450  TEXT='what sort of health benefits do i have'
    OPTIONS(hard, the prompt B1free saw): A=pto_request_status | B=rollover_401k | C=payday | D=insurance  gold=insurance (D)
    B1free raw='A' -> pto_request_status   | A1letter (same prompt, grammar) raw='A' -> pto_request_status  SAME
    true logit read of the same prompt (B1perm shift0): none exists on b1 (B1 main crashed)
    what the published join compared it with: B1 easy:k4:f1, options=['min_payment', 'insurance', 'credit_score', 'update_playlist'], pred=insurance (correct=True), gap=5.95
--- #4 clinc-000404  TEXT='is my replacement card due to arrive in the mail today'
    OPTIONS(hard, the prompt B1free saw): A=redeem_rewards | B=replacement_card_duration | C=expiration_date | D=damaged_card  gold=replacement_card_duration (B)
    B1free raw='A' -> redeem_rewards   | A1letter (same prompt, grammar) raw='D' -> damaged_card  DIFFERENT
    true logit read of the same prompt (B1perm shift0): pred=damaged_card gap=4.146 logits=[53.48, 36.39, 35.73, 57.63]
    what the published join compared it with: B1 easy:k4:f1, options=['translate', 'exchange_rate', 'meaning_of_life', 'replacement_card_duration'], pred=replacement_card_duration (correct=True), gap=6.28
--- #5 clinc-001077  TEXT='i need reviews for places serving tacos in chicago'
    OPTIONS(hard, the prompt B1free saw): A=confirm_reservation | B=accept_reservations | C=restaurant_suggestion | D=restaurant_reviews  gold=restaurant_suggestion (C)
    B1free raw='D' -> restaurant_reviews   | A1letter (same prompt, grammar) raw='D' -> restaurant_reviews  SAME
    true logit read of the same prompt (B1perm shift0): none exists on b1 (B1 main crashed)
    what the published join compared it with: B1 easy:k4:f1, options=['restaurant_suggestion', 'ingredient_substitution', 'card_declined', 'reset_settings'], pred=restaurant_suggestion (correct=True), gap=31.52
--- #6 clinc-000393  TEXT='how can i request a new credit card'
    OPTIONS(hard, the prompt B1free saw): A=report_lost_card | B=replacement_card_duration | C=improve_credit_score | D=credit_score  gold=replacement_card_duration (B)
    B1free raw='A' -> report_lost_card   | A1letter (same prompt, grammar) raw='A' -> report_lost_card  SAME
    true logit read of the same prompt (B1perm shift0): pred=report_lost_card gap=8.418 logits=[56.52, 32.53, 48.1, 36.02]
    what the published join compared it with: B1 easy:k4:f1, options=['definition', 'pto_used', 'share_location', 'replacement_card_duration'], pred=definition (correct=False), gap=24.65
--- #7 clinc-000594  TEXT='a hidden government facility'
    OPTIONS(hard, the prompt B1free saw): A=where_are_you_from | B=who_do_you_work_for | C=are_you_a_bot | D=fun_fact  gold=where_are_you_from (A)
    B1free raw='D' -> fun_fact   | A1letter (same prompt, grammar) raw='D' -> fun_fact  SAME
    true logit read of the same prompt (B1perm shift0): none exists on b1 (B1 main crashed)
    what the published join compared it with: B1 easy:k4:f1, options=['schedule_meeting', 'credit_score', 'where_are_you_from', 'rewards_balance'], pred=schedule_meeting (correct=False), gap=17.12
--- #8 clinc-000695  TEXT='give me instructions for an oil change'
    OPTIONS(hard, the prompt B1free saw): A=last_maintenance | B=gas_type | C=directions | D=oil_change_how  gold=oil_change_how (D)
    B1free raw='C' -> directions   | A1letter (same prompt, grammar) raw='C' -> directions  SAME
    true logit read of the same prompt (B1perm shift0): none exists on b1 (B1 main crashed)
    what the published join compared it with: B1 easy:k4:f1, options=['change_user_name', 'maybe', 'oil_change_how', 'food_last'], pred=oil_change_how (correct=True), gap=27.36
--- #9 clinc-000916  TEXT="what's the average time to boston when riding a bus"
    OPTIONS(hard, the prompt B1free saw): A=gas | B=uber | C=distance | D=tire_pressure  gold=distance (C)
    B1free raw='A' -> gas   | A1letter (same prompt, grammar) raw='A' -> gas  SAME
    true logit read of the same prompt (B1perm shift0): none exists on b1 (B1 main crashed)
    what the published join compared it with: B1 easy:k4:f1, options=['smart_home', 'change_accent', 'cancel', 'distance'], pred=distance (correct=True), gap=23.35
--- #10 clinc-001944  TEXT='unsync yourself from my device'
    OPTIONS(hard, the prompt B1free saw): A=sync_device | B=cancel | C=change_accent | D=whisper_mode  gold=sync_device (A)
    B1free raw='B' -> cancel   | A1letter (same prompt, grammar) raw='B' -> cancel  SAME
    true logit read of the same prompt (B1perm shift0): none exists on b1 (B1 main crashed)
    what the published join compared it with: B1 easy:k4:f1, options=['gas_type', 'interest_rate', 'sync_device', 'card_declined'], pred=sync_device (correct=True), gap=27.31
--- #11 clinc-000530  TEXT='flip a coin, i call heads!'
    OPTIONS(hard, the prompt B1free saw): A=make_call | B=flip_coin | C=share_location | D=weather  gold=flip_coin (B)
    B1free raw='A' -> make_call   | A1letter (same prompt, grammar) raw='A' -> make_call  SAME
    true logit read of the same prompt (B1perm shift0): none exists on b1 (B1 main crashed)
    what the published join compared it with: B1 easy:k4:f1, options=['shopping_list', 'flip_coin', 'food_last', 'spelling'], pred=flip_coin (correct=True), gap=29.37

summary: A1letter same answer as B1free on 10 / 11 ; easy-set logit read correct on 9 / 11

# APPENDIX D — q2_out.txt
# Q2 — A1joint vs A1split on the DEPENDENT fields (runs/b1, main grid)

## Denominators
A1joint main records: 6750; states: {'ok': 6750}
A1split main records: 4950; states: {'ok': 4950}
A1joint unit rows: 13500; by field: {'intent': 6750, 'domain': 4500, 'is_banking_or_credit_cards': 2250}
A1split unit rows: 4950; by field: {'intent': 2250, 'domain': 2250, 'is_banking_or_credit_cards': 450}
A1split rows by (field, field-condition) — --unique-prompts kept ONE copy of each per-field prompt, under the first field-condition that renders it:
   intent: {'f1': 2250}
   domain: {'f2': 2250}
   is_banking_or_credit_cards: {'f3': 450}
A1joint rows by (field, field-condition):
   intent: {'f1': 2250, 'f2': 2250, 'f3': 2250}
   domain: {'f2': 2250, 'f3': 2250}
   is_banking_or_credit_cards: {'f3': 2250}

## Pairing by (text_id, field, k): split has one row per key (any f-condition); joint has one row per (key, f-condition)
### field=domain: split keys 2250, keys with >1 split row: 0
  joint@f2 vs split: joint keys 2250, common 2250, joint-only 0, split-only 0
  n=2250  acc(joint)=1684/2250 = 0.7484  acc(split)=1563/2250 = 0.6947  delta split-joint=-5.38 pp [-7.47, -3.33] (cluster bootstrap over text_id, 20000)  discordant b(split right,joint wrong)=115 c(joint right,split wrong)=236
  per k:
    k=2   n=450  joint 408/450 = 0.9067  split 404/450 = 0.8978  delta -0.89 pp [-3.78,+2.00]  b=21 c=25
    k=4   n=450  joint 369/450 = 0.8200  split 349/450 = 0.7756  delta -4.44 pp [-8.00,-1.11]  b=21 c=41
    k=8   n=450  joint 301/450 = 0.6689  split 276/450 = 0.6133  delta -5.56 pp [-9.56,-1.56]  b=31 c=56
    k=15  n=450  joint 300/450 = 0.6667  split 267/450 = 0.5933  delta -7.33 pp [-11.33,-3.56]  b=25 c=58
    k=16  n=450  joint 306/450 = 0.6800  split 267/450 = 0.5933  delta -8.67 pp [-12.44,-5.11]  b=17 c=56
  joint@f3 vs split: joint keys 2250, common 2250, joint-only 0, split-only 0
  n=2250  acc(joint)=1694/2250 = 0.7529  acc(split)=1563/2250 = 0.6947  delta split-joint=-5.82 pp [-8.18, -3.51] (cluster bootstrap over text_id, 20000)  discordant b(split right,joint wrong)=144 c(joint right,split wrong)=275
  per k:
    k=2   n=450  joint 405/450 = 0.9000  split 404/450 = 0.8978  delta -0.22 pp [-3.11,+2.89]  b=24 c=25
    k=4   n=450  joint 372/450 = 0.8267  split 349/450 = 0.7756  delta -5.11 pp [-8.67,-1.78]  b=21 c=44
    k=8   n=450  joint 308/450 = 0.6844  split 276/450 = 0.6133  delta -7.11 pp [-11.56,-2.67]  b=38 c=70
    k=15  n=450  joint 307/450 = 0.6822  split 267/450 = 0.5933  delta -8.89 pp [-13.33,-4.44]  b=32 c=72
    k=16  n=450  joint 302/450 = 0.6711  split 267/450 = 0.5933  delta -7.78 pp [-12.00,-3.78]  b=29 c=64
### field=is_banking_or_credit_cards: split keys 450, keys with >1 split row: 0
  joint@f3 vs split: joint keys 2250, common 450, joint-only 1800, split-only 0
  n=450  acc(joint)=431/450 = 0.9578  acc(split)=407/450 = 0.9044  delta split-joint=-5.33 pp [-7.56, -3.11] (cluster bootstrap over text_id, 20000)  discordant b(split right,joint wrong)=2 c(joint right,split wrong)=26
    boolean gold balance: true 90/450; balanced acc joint=0.9028 split=0.7653
    gold=true: joint 73/90 = 0.8111  split 48/90 = 0.5333
    gold=false: joint 358/360 = 0.9944  split 359/360 = 0.9972
  per k:
    k=2   n=450  joint 431/450 = 0.9578  split 407/450 = 0.9044  delta -5.33 pp [-7.56,-3.11]  b=2 c=26

## Cross-field cue inside A1joint records (same record: was its OWN earlier answer right?)
### A1joint main: domain accuracy by whether the same record's intent was right
  all  intent RIGHT: domain acc 3088/4090 = 0.7550
  all  intent WRONG: domain acc 290/410 = 0.7073
  f2   intent RIGHT: domain acc 1541/2043 = 0.7543
  f2   intent WRONG: domain acc 143/207 = 0.6908
  f3   intent RIGHT: domain acc 1547/2047 = 0.7557
  f3   intent WRONG: domain acc 147/203 = 0.7241
### A1joint main (f3): is_banking_or_credit_cards accuracy by whether the same record's domain was right
  all  domain RIGHT: is_banking_or_credit_cards acc 1660/1694 = 0.9799
  all  domain WRONG: is_banking_or_credit_cards acc 511/556 = 0.9191
  f3   domain RIGHT: is_banking_or_credit_cards acc 1660/1694 = 0.9799
  f3   domain WRONG: is_banking_or_credit_cards acc 511/556 = 0.9191
### A1joint main (f3), boolean by INTENT: is_banking_or_credit_cards accuracy by whether the same record's intent was right
  all  intent RIGHT: is_banking_or_credit_cards acc 1976/2047 = 0.9653
  all  intent WRONG: is_banking_or_credit_cards acc 195/203 = 0.9606
  f3   intent RIGHT: is_banking_or_credit_cards acc 1976/2047 = 0.9653
  f3   intent WRONG: is_banking_or_credit_cards acc 195/203 = 0.9606

### Self-consistency: is the joint domain simply domain(own intent answer)? and boolean = (own domain in banking/credit_cards)?
#### A1joint main
  domain  f2 k=15: n=450 pred_domain==domain(pred_intent) 300/450 = 0.6667; domain(pred_intent)==gold 450/450 = 1.0000
  domain  f2 k=16: n=450 pred_domain==domain(pred_intent) 304/450 = 0.6756; domain(pred_intent)==gold 448/450 = 0.9956
  domain  f2 k=2: n=450 pred_domain==domain(pred_intent) 408/450 = 0.9067; domain(pred_intent)==gold 450/450 = 1.0000
  domain  f2 k=4: n=450 pred_domain==domain(pred_intent) 369/450 = 0.8200; domain(pred_intent)==gold 450/450 = 1.0000
  domain  f2 k=8: n=450 pred_domain==domain(pred_intent) 301/450 = 0.6689; domain(pred_intent)==gold 450/450 = 1.0000
  domain  f2 k=all: n=2250 pred_domain==domain(pred_intent) 1682/2250 = 0.7476; domain(pred_intent)==gold 2248/2250 = 0.9991
  domain  f3 k=15: n=450 pred_domain==domain(pred_intent) 307/450 = 0.6822; domain(pred_intent)==gold 450/450 = 1.0000
  domain  f3 k=16: n=450 pred_domain==domain(pred_intent) 300/450 = 0.6667; domain(pred_intent)==gold 448/450 = 0.9956
  domain  f3 k=2: n=450 pred_domain==domain(pred_intent) 405/450 = 0.9000; domain(pred_intent)==gold 450/450 = 1.0000
  domain  f3 k=4: n=450 pred_domain==domain(pred_intent) 372/450 = 0.8267; domain(pred_intent)==gold 450/450 = 1.0000
  domain  f3 k=8: n=450 pred_domain==domain(pred_intent) 308/450 = 0.6844; domain(pred_intent)==gold 450/450 = 1.0000
  domain  f3 k=all: n=2250 pred_domain==domain(pred_intent) 1692/2250 = 0.7520; domain(pred_intent)==gold 2248/2250 = 0.9991
  boolean f3 k=15: n=450 pred_bool==f(pred_domain) 435/450 = 0.9667; f(pred_domain)==gold 430/450 = 0.9556
  boolean f3 k=16: n=450 pred_bool==f(pred_domain) 433/450 = 0.9622; f(pred_domain)==gold 424/450 = 0.9422
  boolean f3 k=2: n=450 pred_bool==f(pred_domain) 432/450 = 0.9600; f(pred_domain)==gold 439/450 = 0.9756
  boolean f3 k=4: n=450 pred_bool==f(pred_domain) 440/450 = 0.9778; f(pred_domain)==gold 440/450 = 0.9778
  boolean f3 k=8: n=450 pred_bool==f(pred_domain) 434/450 = 0.9644; f(pred_domain)==gold 430/450 = 0.9556
  boolean f3 k=all: n=2250 pred_bool==f(pred_domain) 2174/2250 = 0.9662; f(pred_domain)==gold 2163/2250 = 0.9613
#### A1split main — same quantities across its INDEPENDENT requests (text, k)
  split: pred_domain==domain(pred_intent) 1563/2250 = 0.6947; domain(pred_intent)==gold 2248/2250 = 0.9991; pred_bool==f(pred_domain) 416/450 = 0.9244

## Why the intent list leaks the domain under the main (within-domain) policy: share of joint records whose intent OPTIONS all come from ONE domain
  {2: '1350/1350', 4: '1350/1350', 8: '1350/1350', 15: '1350/1350', 16: '0/1350'}

## Control where the leak is absent: A1joint on the EASY subset (cross-domain intent distractors) — domain acc by own intent right/wrong
A1joint easy records: 1350; fields conditions: {'f3': 450, 'f2': 450, 'f1': 450}
### A1joint easy: domain accuracy by whether the same record's intent was right
  all  intent RIGHT: domain acc 629/876 = 0.7180
  all  intent WRONG: domain acc 18/24 = 0.7500
  f2   intent RIGHT: domain acc 308/437 = 0.7048
  f2   intent WRONG: domain acc 9/13 = 0.6923
  f3   intent RIGHT: domain acc 321/439 = 0.7312
  f3   intent WRONG: domain acc 9/11 = 0.8182
#### A1joint easy
  domain  f2 k=15: n=113 pred_domain==domain(pred_intent) 68/113 = 0.6018; domain(pred_intent)==gold 109/113 = 0.9646
  domain  f2 k=2: n=112 pred_domain==domain(pred_intent) 97/112 = 0.8661; domain(pred_intent)==gold 111/112 = 0.9911
  domain  f2 k=4: n=113 pred_domain==domain(pred_intent) 86/113 = 0.7611; domain(pred_intent)==gold 113/113 = 1.0000
  domain  f2 k=8: n=112 pred_domain==domain(pred_intent) 64/112 = 0.5714; domain(pred_intent)==gold 110/112 = 0.9821
  domain  f2 k=all: n=450 pred_domain==domain(pred_intent) 315/450 = 0.7000; domain(pred_intent)==gold 443/450 = 0.9844
  domain  f3 k=15: n=112 pred_domain==domain(pred_intent) 66/112 = 0.5893; domain(pred_intent)==gold 110/112 = 0.9821
  domain  f3 k=2: n=113 pred_domain==domain(pred_intent) 100/113 = 0.8850; domain(pred_intent)==gold 113/113 = 1.0000
  domain  f3 k=4: n=112 pred_domain==domain(pred_intent) 86/112 = 0.7679; domain(pred_intent)==gold 109/112 = 0.9732
  domain  f3 k=8: n=113 pred_domain==domain(pred_intent) 72/113 = 0.6372; domain(pred_intent)==gold 110/113 = 0.9735
  domain  f3 k=all: n=450 pred_domain==domain(pred_intent) 324/450 = 0.7200; domain(pred_intent)==gold 442/450 = 0.9822
  boolean f3 k=15: n=112 pred_bool==f(pred_domain) 103/112 = 0.9196; f(pred_domain)==gold 104/112 = 0.9286
  boolean f3 k=2: n=113 pred_bool==f(pred_domain) 107/113 = 0.9469; f(pred_domain)==gold 109/113 = 0.9646
  boolean f3 k=4: n=112 pred_bool==f(pred_domain) 109/112 = 0.9732; f(pred_domain)==gold 112/112 = 1.0000
  boolean f3 k=8: n=113 pred_bool==f(pred_domain) 107/113 = 0.9469; f(pred_domain)==gold 103/113 = 0.9115
  boolean f3 k=all: n=450 pred_bool==f(pred_domain) 426/450 = 0.9467; f(pred_domain)==gold 428/450 = 0.9511
  easy: intent options all one domain: {2: '30/338', 4: '0/337', 8: '0/338', 15: '0/337'}

## Split-arm stratified the same way (its intent answer is from an INDEPENDENT request on the same text/k) — text-difficulty baseline, not conditioning
  split intent RIGHT: domain acc 1431/2047 = 0.6991
  split intent WRONG: domain acc 132/203 = 0.6502
  split domain RIGHT: boolean acc 373/404 = 0.9233
  split domain WRONG: boolean acc 34/46 = 0.7391

## Decomposition of the paired delta by the JOINT record's own intent correctness (domain, joint@f2 vs split; joint@f3 vs split) and by own domain correctness (boolean, joint@f3 vs split)
### domain, joint@f2 vs split, n=2250
  joint intent RIGHT: n=2043 (0.908 of pairs)  joint domain 1541/2043 = 0.7543  split domain 1431/2043 = 0.7004  contribution to total delta(split-joint) = -4.89 pp
  joint intent WRONG: n=207 (0.092 of pairs)  joint domain 143/207 = 0.6908  split domain 132/207 = 0.6377  contribution to total delta(split-joint) = -0.49 pp
  total: joint 1684/2250 = 0.7484 split 1563/2250 = 0.6947 delta -5.38 pp
### domain, joint@f3 vs split, n=2250
  joint intent RIGHT: n=2047 (0.910 of pairs)  joint domain 1547/2047 = 0.7557  split domain 1435/2047 = 0.7010  contribution to total delta(split-joint) = -4.98 pp
  joint intent WRONG: n=203 (0.090 of pairs)  joint domain 147/203 = 0.7241  split domain 128/203 = 0.6305  contribution to total delta(split-joint) = -0.84 pp
  total: joint 1694/2250 = 0.7529 split 1563/2250 = 0.6947 delta -5.82 pp
### is_banking_or_credit_cards, joint@f3 vs split, n=450
  joint domain RIGHT: n=405 (0.900 of pairs)  joint is_banking_or_credit_cards 392/405 = 0.9679  split is_banking_or_credit_cards 368/405 = 0.9086  contribution to total delta(split-joint) = -5.33 pp
  joint domain WRONG: n=45 (0.100 of pairs)  joint is_banking_or_credit_cards 39/45 = 0.8667  split is_banking_or_credit_cards 39/45 = 0.8667  contribution to total delta(split-joint) = +0.00 pp
  total: joint 431/450 = 0.9578 split 407/450 = 0.9044 delta -5.33 pp
### is_banking_or_credit_cards, joint@f3 vs split, n=450
  joint intent RIGHT: n=439 (0.976 of pairs)  joint is_banking_or_credit_cards 422/439 = 0.9613  split is_banking_or_credit_cards 398/439 = 0.9066  contribution to total delta(split-joint) = -5.33 pp
  joint intent WRONG: n=11 (0.024 of pairs)  joint is_banking_or_credit_cards 9/11 = 0.8182  split is_banking_or_credit_cards 9/11 = 0.8182  contribution to total delta(split-joint) = +0.00 pp
  total: joint 431/450 = 0.9578 split 407/450 = 0.9044 delta -5.33 pp

# APPENDIX E — q2b_out.txt
## (a) Boolean: split prompt is identical across k?
  split boolean records 450; distinct prompt_sha 450; conditions {'main:k2:f3': 450}; distinct text_id 450
## (a) Boolean extended pairing: joint@f3 at every k vs the ONE split answer per text (same question, split row reused; bootstrap clusters on text_id)
  pairs 2250 (joint boolean rows 2250)
  all k: n=2250 joint 2171/2250 = 0.9649 split 2035/2250 = 0.9044 delta split-joint -6.04 pp [-8.40,-3.78] b=20 c=156 | gold=true recall joint 395/450 = 0.8778 split 240/450 = 0.5333; gold=false joint 1776/1800 = 0.9867 split 1795/1800 = 0.9972; balanced joint 0.9322 split 0.7653
  k=2: n=450 joint 431/450 = 0.9578 split 407/450 = 0.9044 delta split-joint -5.33 pp [-7.56,-3.11] b=2 c=26 | gold=true recall joint 73/90 = 0.8111 split 48/90 = 0.5333; gold=false joint 358/360 = 0.9944 split 359/360 = 0.9972; balanced joint 0.9028 split 0.7653
  k=4: n=450 joint 434/450 = 0.9644 split 407/450 = 0.9044 delta split-joint -6.00 pp [-8.44,-3.78] b=2 c=29 | gold=true recall joint 77/90 = 0.8556 split 48/90 = 0.5333; gold=false joint 357/360 = 0.9917 split 359/360 = 0.9972; balanced joint 0.9236 split 0.7653
  k=8: n=450 joint 436/450 = 0.9689 split 407/450 = 0.9044 delta split-joint -6.44 pp [-8.89,-4.22] b=2 c=31 | gold=true recall joint 79/90 = 0.8778 split 48/90 = 0.5333; gold=false joint 357/360 = 0.9917 split 359/360 = 0.9972; balanced joint 0.9347 split 0.7653
  k=15: n=450 joint 435/450 = 0.9667 split 407/450 = 0.9044 delta split-joint -6.22 pp [-8.89,-3.56] b=6 c=34 | gold=true recall joint 82/90 = 0.9111 split 48/90 = 0.5333; gold=false joint 353/360 = 0.9806 split 359/360 = 0.9972; balanced joint 0.9458 split 0.7653
  k=16: n=450 joint 435/450 = 0.9667 split 407/450 = 0.9044 delta split-joint -6.22 pp [-9.11,-3.56] b=8 c=36 | gold=true recall joint 84/90 = 0.9333 split 48/90 = 0.5333; gold=false joint 351/360 = 0.9750 split 359/360 = 0.9972; balanced joint 0.9542 split 0.7653

## (b) The cross-field cue isolated: A1joint MAIN vs A1joint EASY on the SAME (text_id, k, field-condition). Same model, same text, same domain/boolean question; only the INTENT option list differs (within-domain = leaks the domain; cross-domain = does not)
### domain: main keys 4500, easy keys 900, common 900
  all: n=900 easy(no leak) 647/900 = 0.7189 main(leak) 684/900 = 0.7600 delta main-easy +4.11 pp [+1.11,+7.22] b(main right, easy wrong)=108 c(easy right, main wrong)=71
    k=2   n=225 easy 197/225 = 0.8756 main 204/225 = 0.9067 delta +3.11 pp [-1.33,+8.00]
    k=4   n=225 easy 175/225 = 0.7778 main 181/225 = 0.8044 delta +2.67 pp [-3.11,+8.44]
    k=8   n=225 easy 138/225 = 0.6133 main 150/225 = 0.6667 delta +5.33 pp [-1.33,+12.00]
    k=15  n=225 easy 137/225 = 0.6089 main 149/225 = 0.6622 delta +5.33 pp [-0.44,+11.11]
### is_banking_or_credit_cards: main keys 2250, easy keys 450, common 450
  all: n=450 easy(no leak) 432/450 = 0.9600 main(leak) 435/450 = 0.9667 delta main-easy +0.67 pp [-0.67,+2.22] b(main right, easy wrong)=8 c(easy right, main wrong)=5
    k=2   n=113 easy 107/113 = 0.9469 main 109/113 = 0.9646 delta +1.77 pp [-1.77,+5.31]
    k=4   n=112 easy 109/112 = 0.9732 main 109/112 = 0.9732 delta +0.00 pp [-2.68,+2.68]
    k=8   n=113 easy 107/113 = 0.9469 main 107/113 = 0.9469 delta +0.00 pp [-4.42,+4.42]
    k=15  n=112 easy 109/112 = 0.9732 main 110/112 = 0.9821 delta +0.89 pp [+0.00,+2.68]
  gold=true: easy 71/82 = 0.8659 main 69/82 = 0.8415

## (c) A1joint EASY vs A1split (main) on the same (text_id, k): the joint arm WITHOUT the option-list leak against separate requests
  domain joint-easy@f2 vs split: n=450 joint-easy 317/450 = 0.7044 split 326/450 = 0.7244 delta split-joint +2.00 pp [-2.22,+6.22] b=51 c=42
  domain joint-easy@f3 vs split: n=450 joint-easy 330/450 = 0.7333 split 321/450 = 0.7133 delta split-joint -2.00 pp [-6.44,+2.22] b=44 c=53
  is_banking_or_credit_cards joint-easy@f3 vs split: n=450 joint-easy 432/450 = 0.9600 split 406/450 = 0.9022 delta split-joint -5.78 pp [-9.56,-2.22] b=7 c=33

## (d) EASY: when the joint arm's OWN intent answer implies a WRONG domain, does its domain follow the answer or the text?
  records where domain(pred_intent) != gold domain: 15; domain FOLLOWED own wrong intent: 2; domain == gold (ignored own answer): 10; neither: 3
    ('clinc-003427', 15, 'insurance', 'work', 'travel', 'travel')
    ('clinc-002780', 15, 'maybe', 'meta', 'kitchen_and_dining', 'kitchen_and_dining')
    ('clinc-002763', 15, 'time', 'utility', 'kitchen_and_dining', 'kitchen_and_dining')
    ('clinc-000919', 15, 'travel_suggestion', 'travel', 'travel', 'auto_and_commute')
    ('clinc-004322', 15, 'jump_start', 'auto_and_commute', 'kitchen_and_dining', 'kitchen_and_dining')
    ('clinc-003850', 15, 'calculator', 'utility', 'home', 'banking')
    ('clinc-003497', 8, 'interest_rate', 'banking', 'auto_and_commute', 'auto_and_commute')
    ('clinc-000919', 8, 'travel_notification', 'travel', 'travel', 'auto_and_commute')
    ('clinc-002881', 8, 'text', 'utility', 'small_talk', 'small_talk')
    ('clinc-000910', 8, 'travel_alert', 'travel', 'auto_and_commute', 'auto_and_commute')
    ('clinc-000594', 8, 'whisper_mode', 'meta', 'work', 'small_talk')
    ('clinc-000594', 4, 'rewards_balance', 'credit_cards', 'small_talk', 'small_talk')
    ('clinc-002662', 4, 'time', 'utility', 'work', 'work')
    ('clinc-003427', 4, 'transactions', 'banking', 'travel', 'travel')
    ('clinc-000594', 2, 'timer', 'utility', 'meta', 'small_talk')

## (e) MAIN k=16 (mixed list: 14 same-domain + 1 cross-domain intent): same question
  k=16 records with domain(pred_intent) != gold: 4; followed 0; gold 4; neither 0

# APPENDIX F — VOIDED first pass q1_out.txt (join by (text_id,intent,4) into B1.jsonl: 0/200 identical prompts; kept only as evidence of what the naive join produces; its gap and cross-host numbers are NOT valid)
# Q1 — B1free vs B1 on runs/b1 (and cross-host with runs/r1)

## Denominators
b1/B1free records: 200; conditions: Counter({'ctl:textanswer': 200})
b1/B1 records: 11547; conditions with k=4 intent: Counter({'easy:k4:f1': 450, 'easy:k4:f2': 450, 'easy:k4:f3': 450, 'oos:in_scope': 50, 'oos:oos': 50, 'ctl:nogold': 40})
r1/B1free records: 200; r1/B1 records: 6000
b1 states of B1free: Counter({'ok': 200}); finish_reason: Counter({'eos': 200}); generated_tokens: Counter({2: 200})
r1 states of B1free: Counter({'ok': 200}); finish_reason: Counter({'eos': 200}); generated_tokens: Counter({2: 200})
b1 B1free raw_response distribution: [('C', 52), ('A', 51), ('D', 50), ('B', 47)]
r1 B1free raw_response distribution: [('C', 59), ('D', 50), ('B', 47), ('A', 44)]

### runs/b1: join B1free -> B1 by (text_id, field, k)
paired: 200 / 200; unmatched B1free: 0
JOIN CONTROL — options list identical: 0/200; prompt_sha identical: 0/200
B1 duplicate records per unit (f1/f2/f3 re-renders): units with >1 record 200/200; units whose duplicates disagree among themselves: 41/200
agreement free==logit: 190/200 = 0.9500
### runs/r1: join B1free -> B1 by (text_id, field, k)
paired: 200 / 200; unmatched B1free: 0
JOIN CONTROL — options list identical: 200/200; prompt_sha identical: 200/200
B1 duplicate records per unit (f1/f2/f3 re-renders): units with >1 record 200/200; units whose duplicates disagree among themselves: 0/200
agreement free==logit: 200/200 = 1.0000

## The 10 disagreements on runs/b1 (denominator 200)
--- #1 text_id=clinc-000650 class=[c: other]
    TEXT: 'tell me what you can do for me'
    OPTIONS: A=account_blocked | B=last_maintenance | C=international_visa | D=what_can_i_ask_you
    gold='what_can_i_ask_you' (pos 3=D)
    B1free raw_response='A' state=ok finish=eos gen_tokens=2 parsed_pred='thank_you'
    B1 pred='what_can_i_ask_you' (pos 3=D) argmax_class=bare_candidate tie=False native_matches_fp32=True
    B1 cand_logits_bare=[26.928, 25.825, 26.309, 56.079]  ranking=['D', 'A', 'C', 'B']
    B1 gap=29.150627 candidate_mass=1.0 p_cand=[0.0, 0.0, 0.0, 1.0]
    free letter rank in B1 logits: None  | free_correct=False b1_correct=True
    B1 dup records for unit=3 dup preds=['what_can_i_ask_you']  options_eq=False prompt_sha_eq=False
--- #2 text_id=clinc-001450 class=[c: other]
    TEXT: 'what sort of health benefits do i have'
    OPTIONS: A=min_payment | B=insurance | C=credit_score | D=update_playlist
    gold='insurance' (pos 1=B)
    B1free raw_response='A' state=ok finish=eos gen_tokens=2 parsed_pred='pto_request_status'
    B1 pred='insurance' (pos 1=B) argmax_class=bare_candidate tie=False native_matches_fp32=True
    B1 cand_logits_bare=[51.374, 57.321, 34.855, 33.125]  ranking=['B', 'A', 'C', 'D']
    B1 gap=5.946438 candidate_mass=1.0 p_cand=[0.002608, 0.997392, 0.0, 0.0]
    free letter rank in B1 logits: None  | free_correct=False b1_correct=True
    B1 dup records for unit=3 dup preds=['insurance']  options_eq=False prompt_sha_eq=False
--- #3 text_id=clinc-000404 class=[a: different VALID letter]
    TEXT: 'is my replacement card due to arrive in the mail today'
    OPTIONS: A=expiration_date | B=none of the above | C=replacement_card_duration | D=redeem_rewards | E=damaged_card
    gold='replacement_card_duration' (pos 2=C)
    B1free raw_response='A' state=ok finish=eos gen_tokens=2 parsed_pred='redeem_rewards'
    B1 pred='damaged_card' (pos 4=E) argmax_class=bare_candidate tie=False native_matches_fp32=True
    B1 cand_logits_bare=[51.112, 34.784, 32.408, 29.718, 55.646]  ranking=['E', 'A', 'B', 'C', 'D']
    B1 gap=4.534054 candidate_mass=1.0 p_cand=[0.010623, 0.0, 0.0, 0.0, 0.989377]
    free letter rank in B1 logits: 5  | free_correct=False b1_correct=False
    B1 dup records for unit=4 dup preds=['damaged_card', 'replacement_card_duration']  options_eq=False prompt_sha_eq=False
--- #4 text_id=clinc-001077 class=[c: other]
    TEXT: 'i need reviews for places serving tacos in chicago'
    OPTIONS: A=restaurant_suggestion | B=ingredient_substitution | C=card_declined | D=reset_settings
    gold='restaurant_suggestion' (pos 0=A)
    B1free raw_response='D' state=ok finish=eos gen_tokens=2 parsed_pred='restaurant_reviews'
    B1 pred='restaurant_suggestion' (pos 0=A) argmax_class=bare_candidate tie=False native_matches_fp32=True
    B1 cand_logits_bare=[59.727, 28.211, 27.539, 27.755]  ranking=['A', 'B', 'D', 'C']
    B1 gap=31.516081 candidate_mass=1.0 p_cand=[1.0, 0.0, 0.0, 0.0]
    free letter rank in B1 logits: None  | free_correct=False b1_correct=True
    B1 dup records for unit=3 dup preds=['restaurant_suggestion']  options_eq=False prompt_sha_eq=False
--- #5 text_id=clinc-000393 class=[a: different VALID letter]
    TEXT: 'how can i request a new credit card'
    OPTIONS: A=replacement_card_duration | B=improve_credit_score | C=none of the above | D=report_lost_card | E=credit_score
    gold='replacement_card_duration' (pos 0=A)
    B1free raw_response='A' state=ok finish=eos gen_tokens=2 parsed_pred='report_lost_card'
    B1 pred='none of the above' (pos 2=C) argmax_class=bare_candidate tie=False native_matches_fp32=True
    B1 cand_logits_bare=[54.214, 32.177, 54.621, 32.06, 33.071]  ranking=['C', 'A', 'E', 'B', 'D']
    B1 gap=0.406639 candidate_mass=1.0 p_cand=[0.399718, 0.0, 0.600282, 0.0, 0.0]
    free letter rank in B1 logits: 5  | free_correct=False b1_correct=False
    B1 dup records for unit=5 dup preds=['definition', 'international_fees', 'none of the above']  options_eq=False prompt_sha_eq=False
--- #6 text_id=clinc-000594 class=[c: other]
    TEXT: 'a hidden government facility'
    OPTIONS: A=schedule_meeting | B=credit_score | C=where_are_you_from | D=rewards_balance
    gold='where_are_you_from' (pos 2=C)
    B1free raw_response='D' state=ok finish=eos gen_tokens=2 parsed_pred='fun_fact'
    B1 pred='schedule_meeting' (pos 0=A) argmax_class=bare_candidate tie=False native_matches_fp32=True
    B1 cand_logits_bare=[54.338, 30.487, 35.693, 37.214]  ranking=['A', 'D', 'C', 'B']
    B1 gap=17.123978 candidate_mass=1.0 p_cand=[1.0, 0.0, 0.0, 0.0]
    free letter rank in B1 logits: None  | free_correct=False b1_correct=False
    B1 dup records for unit=3 dup preds=['schedule_meeting']  options_eq=False prompt_sha_eq=False
--- #7 text_id=clinc-000695 class=[c: other]
    TEXT: 'give me instructions for an oil change'
    OPTIONS: A=change_user_name | B=maybe | C=oil_change_how | D=food_last
    gold='oil_change_how' (pos 2=C)
    B1free raw_response='C' state=ok finish=eos gen_tokens=2 parsed_pred='directions'
    B1 pred='oil_change_how' (pos 2=C) argmax_class=bare_candidate tie=False native_matches_fp32=True
    B1 cand_logits_bare=[26.699, 27.996, 55.353, 26.767]  ranking=['C', 'B', 'D', 'A']
    B1 gap=27.356743 candidate_mass=1.0 p_cand=[0.0, 0.0, 1.0, 0.0]
    free letter rank in B1 logits: None  | free_correct=False b1_correct=True
    B1 dup records for unit=3 dup preds=['oil_change_how']  options_eq=False prompt_sha_eq=False
--- #8 text_id=clinc-000916 class=[c: other]
    TEXT: "what's the average time to boston when riding a bus"
    OPTIONS: A=smart_home | B=change_accent | C=cancel | D=distance
    gold='distance' (pos 3=D)
    B1free raw_response='A' state=ok finish=eos gen_tokens=2 parsed_pred='gas'
    B1 pred='distance' (pos 3=D) argmax_class=bare_candidate tie=False native_matches_fp32=True
    B1 cand_logits_bare=[35.697, 28.966, 27.573, 59.045]  ranking=['D', 'A', 'B', 'C']
    B1 gap=23.347157 candidate_mass=1.0 p_cand=[0.0, 0.0, 0.0, 1.0]
    free letter rank in B1 logits: None  | free_correct=False b1_correct=True
    B1 dup records for unit=3 dup preds=['distance']  options_eq=False prompt_sha_eq=False
--- #9 text_id=clinc-001944 class=[c: other]
    TEXT: 'unsync yourself from my device'
    OPTIONS: A=gas_type | B=interest_rate | C=sync_device | D=card_declined
    gold='sync_device' (pos 2=C)
    B1free raw_response='B' state=ok finish=eos gen_tokens=2 parsed_pred='cancel'
    B1 pred='sync_device' (pos 2=C) argmax_class=bare_candidate tie=False native_matches_fp32=True
    B1 cand_logits_bare=[33.863, 32.231, 62.226, 34.914]  ranking=['C', 'D', 'A', 'B']
    B1 gap=27.311485 candidate_mass=1.0 p_cand=[0.0, 0.0, 1.0, 0.0]
    free letter rank in B1 logits: None  | free_correct=False b1_correct=True
    B1 dup records for unit=3 dup preds=['sync_device']  options_eq=False prompt_sha_eq=False
--- #10 text_id=clinc-000530 class=[c: other]
    TEXT: 'flip a coin, i call heads!'
    OPTIONS: A=shopping_list | B=flip_coin | C=food_last | D=spelling
    gold='flip_coin' (pos 1=B)
    B1free raw_response='A' state=ok finish=eos gen_tokens=2 parsed_pred='make_call'
    B1 pred='flip_coin' (pos 1=B) argmax_class=bare_candidate tie=False native_matches_fp32=True
    B1 cand_logits_bare=[28.532, 59.218, 29.843, 28.172]  ranking=['B', 'C', 'A', 'D']
    B1 gap=29.374861 candidate_mass=1.0 p_cand=[0.0, 1.0, 0.0, 0.0]
    free letter rank in B1 logits: None  | free_correct=False b1_correct=True
    B1 dup records for unit=3 dup preds=['flip_coin']  options_eq=False prompt_sha_eq=False

classification: {'c: other': 8, 'a: different VALID letter': 2}
free correct on the 11: 0/10; B1 correct on the 11: 7/10
free letter is B1 runner-up (rank 2): 0/10

## B1 gap and candidate_mass: disagreements vs agreements (runs/b1)
gap  disagree (n=10): {'n': 10, 'min': 0.407, 'q1': 8.741, 'med': 25.329, 'q3': 28.702, 'max': 31.516, 'mean': 19.607}
gap  agree    (n=190): {'n': 190, 'min': 4.849, 'q1': 25.482, 'med': 28.056, 'q3': 30.197, 'max': 33.683, 'mean': 27.277}
cand_mass disagree: {'n': 10, 'min': 1.0, 'q1': 1.0, 'med': 1.0, 'q3': 1.0, 'max': 1.0, 'mean': 1.0}
cand_mass agree   : {'n': 190, 'min': 1.0, 'q1': 1.0, 'med': 1.0, 'q3': 1.0, 'max': 1.0, 'mean': 1.0}
full gap list disagree: [0.407, 4.534, 5.946, 17.124, 23.347, 27.311, 27.357, 29.151, 29.375, 31.516]
agreements with gap <= max disagreement gap (31.516): 167/190
P(gap_disagree < gap_agree) over all 1900 cross pairs = 0.683 (0.5 = no difference)
disagreement rate by B1 gap bin:
  gap in [0,1): 1/1
  gap in [1,2): 0/0
  gap in [2,4): 0/0
  gap in [4,8): 2/4
  gap in [8,16): 0/3
  gap in [16,100): 7/192

## Cross-host: units present in BOTH runs/b1 and runs/r1 (same text_id+field+k)
B1free units: b1 200, r1 200, overlap 12
B1 units (all k, intent): b1 5583, r1 3000, overlap 209
CROSS-HOST JOIN CONTROL on B1free overlap: same options 12/12; same prompt_sha 12/12
CROSS-HOST JOIN CONTROL on B1 overlap: same options 43/209; same prompt_sha 43/209
B1 pred changed between hosts on overlapping B1 units: 22/209
  gap (b1) of the changed units: {'n': 22, 'min': 1.024, 'q1': 4.965, 'med': 10.746, 'q3': 20.154, 'max': 27.192, 'mean': 12.798}
  gap (b1) of the unchanged units: {'n': 187, 'min': 0.685, 'q1': 22.794, 'med': 26.14, 'q3': 29.01, 'max': 33.243, 'mean': 24.53}
  k=2: changed 5/66
  k=4: changed 9/66
  k=8: changed 8/66
  k=16: changed 0/11
B1free parsed pred changed between hosts on overlapping B1free units: 0/12
max |logit diff| per unit across hosts (B1 overlap, n=206): {'n': 206, 'min': 0.043, 'q1': 1.038, 'med': 23.95, 'q3': 30.18, 'max': 37.092, 'mean': 17.902}

### The 11 b1 disagreements, in r1?
  clinc-000650: NOT in r1 (B1free in r1: False, B1 in r1: False)
  clinc-001450: b1 B1=insurance free=pto_request_status | r1 B1=pto_request_status free=pto_request_status raw='A' | B1 pred changed=True free changed=False | gap b1=5.946 r1=10.588 | logits b1=[51.37, 57.32, 34.86, 33.13] r1=[56.27, 29.46, 31.67, 45.68] same_opts=False
  clinc-000404: NOT in r1 (B1free in r1: False, B1 in r1: False)
  clinc-001077: NOT in r1 (B1free in r1: False, B1 in r1: False)
  clinc-000393: NOT in r1 (B1free in r1: False, B1 in r1: False)
  clinc-000594: NOT in r1 (B1free in r1: False, B1 in r1: False)
  clinc-000695: NOT in r1 (B1free in r1: False, B1 in r1: False)
  clinc-000916: NOT in r1 (B1free in r1: False, B1 in r1: False)
  clinc-001944: NOT in r1 (B1free in r1: False, B1 in r1: False)
  clinc-000530: NOT in r1 (B1free in r1: False, B1 in r1: False)
  of the 10: in both runs 1; B1 pred changed between hosts 1/1; free answer changed between hosts 0/1; r1 free==r1 B1 on these 1/1

### r1 B1 gap on the same 200-unit control population vs b1
r1 gap on r1 control units (n=200): {'n': 200, 'min': 0.296, 'q1': 25.179, 'med': 27.896, 'q3': 29.773, 'max': 35.72, 'mean': 26.435}
b1 gap on b1 control units (n=200): {'n': 200, 'min': 0.407, 'q1': 25.356, 'med': 27.965, 'q3': 30.1, 'max': 33.683, 'mean': 26.894}
r1 control units with gap<2: 1/200; b1: 1/200
