import sys, json, statistics, collections
sys.path.insert(0, '.')
from minijev.records import read_jsonl

SP = 'analysis/out'

def load(run):
    d = {}
    d['B1free'] = read_jsonl(f'runs/{run}/B1free.jsonl')
    d['B1'] = read_jsonl(f'runs/{run}/B1.jsonl')
    return d

texts = {}
for line in open('data/clinc/sample.jsonl', encoding='utf-8'):
    if line.strip():
        r = json.loads(line); texts[r['text_id']] = r['text']

def b1_index(recs):
    """(text_id, field, k) -> FIRST B1 record; also all records per unit."""
    first, allr = {}, collections.defaultdict(list)
    for r in recs:
        kk = (r['text_id'], r['field'], r['k'])
        allr[kk].append(r)
        first.setdefault(kk, r)
    return first, allr

def quart(xs):
    xs = sorted(xs)
    if not xs: return None
    qs = statistics.quantiles(xs, n=4, method='inclusive') if len(xs) >= 2 else [xs[0]]*3
    return dict(n=len(xs), min=round(xs[0],3), q1=round(qs[0],3), med=round(statistics.median(xs),3), q3=round(qs[2],3), max=round(xs[-1],3), mean=round(statistics.mean(xs),3))

out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)

b1 = load('b1'); r1 = load('r1')
P('# Q1 — B1free vs B1 on runs/b1 (and cross-host with runs/r1)')
P()
P('## Denominators')
P(f'b1/B1free records: {len(b1["B1free"])}; conditions: {collections.Counter(r["condition"] for r in b1["B1free"])}')
P(f'b1/B1 records: {len(b1["B1"])}; conditions with k=4 intent: {collections.Counter(r["condition"] for r in b1["B1"] if r["k"]==4 and r["field"]=="intent")}')
P(f'r1/B1free records: {len(r1["B1free"])}; r1/B1 records: {len(r1["B1"])}')
P(f'b1 states of B1free: {collections.Counter(r["state"] for r in b1["B1free"])}; finish_reason: {collections.Counter(r["finish_reason"] for r in b1["B1free"])}; generated_tokens: {collections.Counter(r["generated_tokens"] for r in b1["B1free"])}')
P(f'r1 states of B1free: {collections.Counter(r["state"] for r in r1["B1free"])}; finish_reason: {collections.Counter(r["finish_reason"] for r in r1["B1free"])}; generated_tokens: {collections.Counter(r["generated_tokens"] for r in r1["B1free"])}')
P(f'b1 B1free raw_response distribution: {collections.Counter(r["raw_response"] for r in b1["B1free"]).most_common(12)}')
P(f'r1 B1free raw_response distribution: {collections.Counter(r["raw_response"] for r in r1["B1free"]).most_common(12)}')
P()

def pair(run, d, label):
    first, allr = b1_index(d['B1'])
    rows = []
    n_opt_eq = n_sha_eq = n_missing = 0
    dup_disagree = 0; dup_units = 0
    for f in d['B1free']:
        kk = (f['text_id'], f['field'], f['k'])
        b = first.get(kk)
        if b is None:
            n_missing += 1; continue
        opt_eq = (f['options'][f['field']] == b['options'])
        sha_eq = (f['prompt_sha'] == b['prompt_sha'])
        n_opt_eq += opt_eq; n_sha_eq += sha_eq
        # duplicates of the B1 unit across f1/f2/f3: do they agree with each other?
        preds = {x['pred'] for x in allr[kk]}
        if len(allr[kk]) > 1:
            dup_units += 1
            if len(preds) > 1: dup_disagree += 1
        fp = f['per_field_pred'].get(f['field'])
        rows.append(dict(tid=f['text_id'], free=f, b1=b, free_pred=fp, b1_pred=b['pred'],
                         agree=(fp == b['pred']), opt_eq=opt_eq, sha_eq=sha_eq,
                         b1_dups=len(allr[kk]), b1_dup_preds=sorted(preds)))
    P(f'### {label}: join B1free -> B1 by (text_id, field, k)')
    P(f'paired: {len(rows)} / {len(d["B1free"])}; unmatched B1free: {n_missing}')
    P(f'JOIN CONTROL — options list identical: {n_opt_eq}/{len(rows)}; prompt_sha identical: {n_sha_eq}/{len(rows)}')
    P(f'B1 duplicate records per unit (f1/f2/f3 re-renders): units with >1 record {dup_units}/{len(rows)}; units whose duplicates disagree among themselves: {dup_disagree}/{dup_units}')
    agree = sum(r['agree'] for r in rows)
    P(f'agreement free==logit: {agree}/{len(rows)} = {agree/len(rows):.4f}')
    return rows

rows_b1 = pair('b1', b1, 'runs/b1')
rows_r1 = pair('r1', r1, 'runs/r1')
P()

# ---- the 11 disagreements
dis = [r for r in rows_b1 if not r['agree']]
P(f'## The {len(dis)} disagreements on runs/b1 (denominator {len(rows_b1)})')
classes = collections.Counter()
for i, r in enumerate(dis, 1):
    f, b = r['free'], r['b1']
    opts = b['options']; letters = [chr(65+j) for j in range(len(opts))]
    gold = b['gold']
    raw = f['raw_response']
    fp = r['free_pred']
    logits = b['cand_logits_bare']
    order = sorted(range(len(opts)), key=lambda j: -logits[j])
    # where does the free letter rank in B1 logits?
    if f['state'] == 'ok' and fp in opts:
        free_pos = opts.index(fp); free_rank = order.index(free_pos) + 1
        cls = 'a: different VALID letter'
    elif f['state'] != 'ok':
        free_pos = None; free_rank = None
        cls = 'b: malformed / unparsable'
    else:
        free_pos = None; free_rank = None; cls = 'c: other'
    classes[cls] += 1
    P(f'--- #{i} text_id={r["tid"]} class=[{cls}]')
    P(f'    TEXT: {texts.get(r["tid"], "<not in current sample>")!r}')
    P('    OPTIONS: ' + ' | '.join(f'{L}={o}' for L, o in zip(letters, opts)))
    P(f'    gold={gold!r} (pos {b["gold_pos"]}={letters[b["gold_pos"]]})')
    P(f'    B1free raw_response={raw!r} state={f["state"]} finish={f["finish_reason"]} gen_tokens={f["generated_tokens"]} parsed_pred={fp!r}')
    P(f'    B1 pred={b["pred"]!r} (pos {b["pred_pos"]}={letters[b["pred_pos"]]}) argmax_class={b["argmax_class"]} tie={b["tie"]} native_matches_fp32={b.get("native_matches_fp32")}')
    P(f'    B1 cand_logits_bare={[round(x,3) for x in logits]}  ranking={[letters[j] for j in order]}')
    P(f'    B1 gap={b["gap"]} candidate_mass={b["candidate_mass"]} p_cand={b["p_cand"]}')
    P(f'    free letter rank in B1 logits: {free_rank}  | free_correct={fp==gold} b1_correct={b["pred"]==gold}')
    P(f'    B1 dup records for unit={r["b1_dups"]} dup preds={r["b1_dup_preds"]}  options_eq={r["opt_eq"]} prompt_sha_eq={r["sha_eq"]}')
P()
P(f'classification: {dict(classes)}')
P(f'free correct on the 11: {sum(1 for r in dis if r["free_pred"]==r["b1"]["gold"])}/{len(dis)}; B1 correct on the 11: {sum(1 for r in dis if r["b1_pred"]==r["b1"]["gold"])}/{len(dis)}')
P(f'free letter is B1 runner-up (rank 2): {sum(1 for r in dis if r["free_pred"] in r["b1"]["options"] and sorted(range(len(r["b1"]["options"])), key=lambda j:-r["b1"]["cand_logits_bare"][j]).index(r["b1"]["options"].index(r["free_pred"]))==1)}/{len(dis)}')
P()

# ---- gap / candidate_mass distributions
ag = [r for r in rows_b1 if r['agree']]
P('## B1 gap and candidate_mass: disagreements vs agreements (runs/b1)')
P(f'gap  disagree (n={len(dis)}): {quart([r["b1"]["gap"] for r in dis])}')
P(f'gap  agree    (n={len(ag)}): {quart([r["b1"]["gap"] for r in ag])}')
P(f'cand_mass disagree: {quart([r["b1"]["candidate_mass"] for r in dis])}')
P(f'cand_mass agree   : {quart([r["b1"]["candidate_mass"] for r in ag])}')
P(f'full gap list disagree: {sorted(round(r["b1"]["gap"],3) for r in dis)}')
# rank-based: how many agreements have gap below the max disagreement gap
mx = max(r['b1']['gap'] for r in dis)
P(f'agreements with gap <= max disagreement gap ({mx:.3f}): {sum(1 for r in ag if r["b1"]["gap"] <= mx)}/{len(ag)}')
# Mann-Whitney U (exact-ish via rank)
import itertools
g_d = [r['b1']['gap'] for r in dis]; g_a = [r['b1']['gap'] for r in ag]
u = sum(1 for x in g_d for y in g_a if x < y) + 0.5*sum(1 for x in g_d for y in g_a if x == y)
P(f'P(gap_disagree < gap_agree) over all {len(g_d)*len(g_a)} cross pairs = {u/(len(g_d)*len(g_a)):.3f} (0.5 = no difference)')
# by gap bins: disagreement rate
bins = [(0,1),(1,2),(2,4),(4,8),(8,16),(16,100)]
P('disagreement rate by B1 gap bin:')
for lo,hi in bins:
    inb = [r for r in rows_b1 if lo <= r['b1']['gap'] < hi]
    nd = sum(1 for r in inb if not r['agree'])
    P(f'  gap in [{lo},{hi}): {nd}/{len(inb)}')
P()

# ---- cross host
P('## Cross-host: units present in BOTH runs/b1 and runs/r1 (same text_id+field+k)')
first_b1, _ = b1_index(b1['B1']); first_r1, _ = b1_index(r1['B1'])
free_b1 = {(f['text_id'], f['field'], f['k']): f for f in b1['B1free']}
free_r1 = {(f['text_id'], f['field'], f['k']): f for f in r1['B1free']}
P(f'B1free units: b1 {len(free_b1)}, r1 {len(free_r1)}, overlap {len(set(free_b1)&set(free_r1))}')
P(f'B1 units (all k, intent): b1 {len(first_b1)}, r1 {len(first_r1)}, overlap {len(set(first_b1)&set(first_r1))}')
ov_free = sorted(set(free_b1) & set(free_r1))
# controls on overlap: same options / prompt across hosts?
same_opts = sum(1 for kk in ov_free if free_b1[kk]['options'] == free_r1[kk]['options'])
same_sha = sum(1 for kk in ov_free if free_b1[kk]['prompt_sha'] == free_r1[kk]['prompt_sha'])
P(f'CROSS-HOST JOIN CONTROL on B1free overlap: same options {same_opts}/{len(ov_free)}; same prompt_sha {same_sha}/{len(ov_free)}')
ov_b1 = sorted(set(first_b1) & set(first_r1))
same_opts_b = sum(1 for kk in ov_b1 if first_b1[kk]['options'] == first_r1[kk]['options'])
same_sha_b = sum(1 for kk in ov_b1 if first_b1[kk]['prompt_sha'] == first_r1[kk]['prompt_sha'])
P(f'CROSS-HOST JOIN CONTROL on B1 overlap: same options {same_opts_b}/{len(ov_b1)}; same prompt_sha {same_sha_b}/{len(ov_b1)}')
# B1 pred stability across hosts on all overlapping B1 units
chg = [kk for kk in ov_b1 if first_b1[kk]['pred'] != first_r1[kk]['pred']]
P(f'B1 pred changed between hosts on overlapping B1 units: {len(chg)}/{len(ov_b1)}')
P(f'  gap (b1) of the changed units: {quart([first_b1[kk]["gap"] for kk in chg])}')
P(f'  gap (b1) of the unchanged units: {quart([first_b1[kk]["gap"] for kk in ov_b1 if kk not in chg])}')
# by k
for k in sorted({kk[2] for kk in ov_b1}):
    sub = [kk for kk in ov_b1 if kk[2]==k]; c = sum(1 for kk in sub if first_b1[kk]['pred'] != first_r1[kk]['pred'])
    P(f'  k={k}: changed {c}/{len(sub)}')
# B1free stability across hosts
chg_f = [kk for kk in ov_free if free_b1[kk]['per_field_pred'].get('intent') != free_r1[kk]['per_field_pred'].get('intent')]
P(f'B1free parsed pred changed between hosts on overlapping B1free units: {len(chg_f)}/{len(ov_free)}')
# logit differences across hosts on overlapping B1 units
diffs = []
for kk in ov_b1:
    a, b = first_b1[kk]['cand_logits_bare'], first_r1[kk]['cand_logits_bare']
    if len(a)==len(b): diffs.append(max(abs(x-y) for x,y in zip(a,b)))
P(f'max |logit diff| per unit across hosts (B1 overlap, n={len(diffs)}): {quart(diffs)}')
P()
P('### The 11 b1 disagreements, in r1?')
n_in = 0; b1pred_changed = 0; free_changed = 0; r1_agree = 0
for r in dis:
    kk = (r['tid'], 'intent', 4)
    inr = kk in free_r1 and kk in first_r1
    if not inr:
        P(f'  {r["tid"]}: NOT in r1 (B1free in r1: {kk in free_r1}, B1 in r1: {kk in first_r1})'); continue
    n_in += 1
    rb = first_r1[kk]; rf = free_r1[kk]
    bc = rb['pred'] != r['b1_pred']; fc = rf['per_field_pred'].get('intent') != r['free_pred']
    b1pred_changed += bc; free_changed += fc
    r1_agree += (rf['per_field_pred'].get('intent') == rb['pred'])
    P(f'  {r["tid"]}: b1 B1={r["b1_pred"]} free={r["free_pred"]} | r1 B1={rb["pred"]} free={rf["per_field_pred"].get("intent")} raw={rf["raw_response"]!r} | B1 pred changed={bc} free changed={fc} | gap b1={r["b1"]["gap"]:.3f} r1={rb["gap"]:.3f} | logits b1={[round(x,2) for x in r["b1"]["cand_logits_bare"]]} r1={[round(x,2) for x in rb["cand_logits_bare"]]} same_opts={rb["options"]==r["b1"]["options"]}')
P(f'  of the {len(dis)}: in both runs {n_in}; B1 pred changed between hosts {b1pred_changed}/{n_in}; free answer changed between hosts {free_changed}/{n_in}; r1 free==r1 B1 on these {r1_agree}/{n_in}')
P()
# r1 within-host: gap distribution vs b1 for the overlapping textanswer units
P('### r1 B1 gap on the same 200-unit control population vs b1')
P(f'r1 gap on r1 control units (n={len(rows_r1)}): {quart([r["b1"]["gap"] for r in rows_r1])}')
P(f'b1 gap on b1 control units (n={len(rows_b1)}): {quart([r["b1"]["gap"] for r in rows_b1])}')
P(f'r1 control units with gap<2: {sum(1 for r in rows_r1 if r["b1"]["gap"]<2)}/{len(rows_r1)}; b1: {sum(1 for r in rows_b1 if r["b1"]["gap"]<2)}/{len(rows_b1)}')
open(f'{SP}/q1_out.txt','w').write('\n'.join(out)+'\n')
