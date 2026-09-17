import sys, json, statistics, collections
sys.path.insert(0, '.')
from minijev.records import read_jsonl
SP = 'analysis/out'
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)
def quart(xs):
    xs = sorted(xs)
    if not xs: return None
    qs = statistics.quantiles(xs, n=4, method='inclusive') if len(xs) >= 2 else [xs[0]]*3
    return dict(n=len(xs), min=round(xs[0],3), q1=round(qs[0],3), med=round(statistics.median(xs),3), q3=round(qs[2],3), max=round(xs[-1],3))
texts = {}
for line in open('data/clinc/sample.jsonl', encoding='utf-8'):
    if line.strip():
        r = json.loads(line); texts[r['text_id']] = r['text']

free = read_jsonl('runs/b1/B1free.jsonl')
B1 = read_jsonl('runs/b1/B1.jsonl'); letter = read_jsonl('runs/b1/A1letter.jsonl'); score = read_jsonl('runs/b1/B1score.jsonl'); perm = read_jsonl('runs/b1/B1perm.jsonl')
by_sha = lambda recs: {r['prompt_sha']: r for r in recs}
L = by_sha(letter); S = by_sha(score); Pm = by_sha([r for r in perm if r['condition']=='perm:k4:shift0'])
# sanity: any sha collision inside each file? (same prompt in two records)
for nm, recs in [('A1letter', letter), ('B1score', score), ('B1perm shift0', [r for r in perm if r['condition']=='perm:k4:shift0'])]:
    c = collections.Counter(r['prompt_sha'] for r in recs)
    P(f'{nm}: records {len(recs)}, distinct prompt_sha {len(c)}, shas with >1 record {sum(1 for v in c.values() if v>1)}')
P()
P('## Q1 corrected: on runs/b1 NO B1 logit read exists for the main:k4 prompts (B1 main crashed at 607/13500 records, run.log 20:14:38 AssertionError "candidate mass > 1"; never resumed).')
P('## Same-prompt comparators on b1 (byte-identical prompt_sha): A1letter (greedy under letter grammar, 200), B1score (label likelihood, 200), B1perm shift0 (TRUE logit read, 50).')
fp = lambda r: r['per_field_pred'].get('intent')
def agree(name, comp, getpred):
    n = a = 0; dis = []
    for f in free:
        c = comp.get(f['prompt_sha'])
        if c is None: continue
        n += 1
        if fp(f) == getpred(c): a += 1
        else: dis.append((f, c))
    P(f'B1free vs {name}: agree {a}/{n} = {a/n:.4f}' if n else f'B1free vs {name}: 0 pairs')
    return dis
dis_L = agree('A1letter (same prompt)', L, fp)
dis_S = agree('B1score (same prompt)', S, lambda c: c['pred'])
dis_P = agree('B1perm shift0 (same prompt, TRUE logit read)', Pm, lambda c: c['pred'])
# A1letter vs B1perm shift0 on the 50
n=a=0; dl=[]
for sha, p in Pm.items():
    l = L.get(sha)
    if l is None: continue
    n+=1; a += (fp(l)==p['pred'])
    if fp(l)!=p['pred']: dl.append((l,p))
P(f'A1letter vs B1perm shift0 (M14 on the 50 same-prompt units): agree {a}/{n}')
P()
P('### Reproduction of the published 189/200: which pairing gives it?')
# lead recipe: first B1 record by (text_id, field, k) in B1.jsonl order
first = {}
for r in B1: first.setdefault((r['text_id'], r['field'], r['k']), r)
a = sum(1 for f in free if fp(f) == first[(f['text_id'],'intent',4)]['pred']); P(f'  join to FIRST B1.jsonl record by (text_id,intent,4) [options differ in 200/200]: {a}/200')
# other orderings: last record, or any condition-specific
last = {}
for r in B1: last[(r['text_id'], r['field'], r['k'])] = r
a = sum(1 for f in free if fp(f) == last[(f['text_id'],'intent',4)]['pred']); P(f'  join to LAST B1.jsonl record by (text_id,intent,4): {a}/200')
for cond in ['easy:k4:f1','easy:k4:f2','easy:k4:f3']:
    idx = {(r['text_id']): r for r in B1 if r['condition']==cond}
    a = sum(1 for f in free if f['text_id'] in idx and fp(f)==idx[f['text_id']]['pred']); n = sum(1 for f in free if f['text_id'] in idx)
    P(f'  join to B1 {cond} by text_id [different distractor policy = different options]: {a}/{n}')
# pred_pos-based (letter) comparisons
a = sum(1 for f in free if f['options']['intent'].index(fp(f)) == first[(f['text_id'],'intent',4)]['pred_pos']); P(f'  compare LETTER POSITION only vs first B1 record: {a}/200')
# via unit_rows/paired style keyed on (condition, text_id, field) -> ctl:textanswer has no B1 -> 0
P(f'  paired() on (condition,text_id,field): B1 has condition ctl:textanswer? {any(r["condition"]=="ctl:textanswer" for r in B1)} -> 0 pairs')
P()
P(f'## Disagreements B1free vs A1letter on b1: {len(dis_L)}/200')
for i,(f,c) in enumerate(dis_L,1):
    opts = f['options']['intent']; Ls=[chr(65+j) for j in range(len(opts))]
    p = Pm.get(f['prompt_sha'])
    P(f'--- #{i} {f["text_id"]}  TEXT={texts.get(f["text_id"])!r}')
    P('    OPTIONS: ' + ' | '.join(f'{a}={o}' for a,o in zip(Ls,opts)) + f'   gold={f["gold"]["intent"]} ({Ls[f["gold_pos"]["intent"]]})')
    P(f'    B1free raw={f["raw_response"]!r} -> {fp(f)!r} (correct={fp(f)==f["gold"]["intent"]}) | A1letter raw={c["raw_response"]!r} -> {fp(c)!r} (correct={fp(c)==c["gold"]["intent"]}) allowed_step1={c["allowed_step1"]} batch(free)={f["batch_size"]}/{f["batch_id"]} batch(letter)={c["batch_size"]}/{c["batch_id"]} pad free={f["pad_tokens"]} letter={c["pad_tokens"]}')
    if p:
        P(f'    B1perm shift0 (logit read): pred={p["pred"]!r} gap={p["gap"]} logits={[round(x,2) for x in p["cand_logits_bare"]]} native_matches_fp32={p.get("native_matches_fp32")} p_cand={p["p_cand"]}')
    else:
        P('    B1perm shift0: not among the 50')
    s = S.get(f['prompt_sha']); P(f'    B1score (likelihood): pred={s["pred"]!r}')
P()
P(f'## Disagreements B1free vs B1perm shift0 (true logit read) on b1: {len(dis_P)}/50')
for i,(f,p) in enumerate(dis_P,1):
    opts = f['options']['intent']; Ls=[chr(65+j) for j in range(len(opts))]
    P(f'--- #{i} {f["text_id"]} TEXT={texts.get(f["text_id"])!r}')
    P('    OPTIONS: ' + ' | '.join(f'{a}={o}' for a,o in zip(Ls,opts)) + f'   gold={f["gold"]["intent"]}')
    P(f'    free raw={f["raw_response"]!r} -> {fp(f)!r}; logit pred={p["pred"]!r} gap={p["gap"]} logits={[round(x,2) for x in p["cand_logits_bare"]]} native_matches_fp32={p.get("native_matches_fp32")}; A1letter -> {fp(L[f["prompt_sha"]])!r}')
P()
P('## Gap analysis on the 50 units with a true same-prompt logit read (B1perm shift0)')
rows50 = [(f, Pm[f['prompt_sha']]) for f in free if f['prompt_sha'] in Pm]
ag = [(f,p) for f,p in rows50 if fp(f)==p['pred']]; dg = [(f,p) for f,p in rows50 if fp(f)!=p['pred']]
P(f'  free==logit: {len(ag)}/50; disagree gap: {[round(p["gap"],3) for f,p in dg]}; agree gap quartiles: {quart([p["gap"] for f,p in ag])}')
# using A1letter disagreement as the marker on the 50 that have logits
agL = [(f,p) for f,p in rows50 if fp(f)==fp(L[f['prompt_sha']])]; dgL = [(f,p) for f,p in rows50 if fp(f)!=fp(L[f['prompt_sha']])]
P(f'  free==A1letter on these 50: {len(agL)}/50; gap of the free!=A1letter units: {[round(p["gap"],3) for f,p in dgL]}; gap quartiles where free==A1letter: {quart([p["gap"] for f,p in agL])}')
P(f'  native_matches_fp32 on the 50: {collections.Counter(p.get("native_matches_fp32") for f,p in rows50)}')
# rotation stability from B1perm: for each of the 50 texts, do the 4 shifts agree on the chosen OPTION?
byt = collections.defaultdict(dict)
for r in perm:
    if r['k']==4: byt[r['text_id']][r['condition']] = r
P('  rotation stability (B1perm k4, 4 shifts) for the free!=A1letter units among the 50:')
for f,p in dgL:
    d = byt[f['text_id']]; preds = [d[f"perm:k4:shift{s}"]['pred'] for s in range(4)]
    P(f'    {f["text_id"]}: preds by shift {preds}; free={fp(f)}; A1letter={fp(L[f["prompt_sha"]])}')
unst = sum(1 for tid,d in byt.items() if len({d[f"perm:k4:shift{s}"]['pred'] for s in range(4)})>1)
P(f'  texts whose logit-read choice changes under rotation (k4, all 50): {unst}/{len(byt)}')
P()
P('## Cross-host (r1 = Mac/MPS) restricted to BYTE-IDENTICAL prompts')
r1B1 = read_jsonl('runs/r1/B1.jsonl'); r1free = read_jsonl('runs/r1/B1free.jsonl')
R = {}
for r in r1B1: R.setdefault(r['prompt_sha'], r)
# b1 logit reads available: B1 (easy/oos/nogold/main k15,k16 partial), B1perm
b1reads = {}
for r in B1 + perm: b1reads.setdefault(r['prompt_sha'], r)
common = sorted(set(R) & set(b1reads))
P(f'B1 logit-read prompts common to both hosts (identical prompt_sha): {len(common)} (r1 distinct {len(R)}, b1 distinct {len(b1reads)})')
chg = [s for s in common if R[s]['pred'] != b1reads[s]['pred']]
P(f'  chosen option changed between hosts: {len(chg)}/{len(common)}')
P(f'  gap(b1) changed: {[round(b1reads[s]["gap"],3) for s in chg]}; gap(r1) changed: {[round(R[s]["gap"],3) for s in chg]}')
P(f'  gap(b1) unchanged quartiles: {quart([b1reads[s]["gap"] for s in common if s not in chg])}')
P(f'  by k: ' + str({k: (sum(1 for s in chg if b1reads[s]["k"]==k), sum(1 for s in common if b1reads[s]["k"]==k)) for k in sorted({b1reads[s]["k"] for s in common})}))
P(f'  by b1 condition: ' + str(collections.Counter(b1reads[s]["condition"] for s in common)))
dif = [max(abs(x-y) for x,y in zip(R[s]['cand_logits_bare'], b1reads[s]['cand_logits_bare'])) for s in common]
P(f'  max |candidate logit diff| across hosts, same prompt: quartiles {quart(dif)}')
# free vs free across hosts, identical prompts
RF = {r['prompt_sha']: r for r in r1free}; cf = sorted(set(RF) & {f['prompt_sha'] for f in free})
P(f'B1free prompts common to both hosts: {len(cf)}; free answer changed: {sum(1 for s in cf if fp(RF[s]) != fp(next(f for f in free if f["prompt_sha"]==s)))}/{len(cf)}')
# on r1: the 200 control units — gap distribution and the r1 A1letter? (no k4 in r1 A1letter)
r1pairs = [(f, R[f['prompt_sha']]) for f in r1free if f['prompt_sha'] in R]
P(f'r1: B1free vs B1 same prompt: {sum(1 for f,b in r1pairs if fp(f)==b["pred"])}/{len(r1pairs)}; r1 B1 gap quartiles on these: {quart([b["gap"] for f,b in r1pairs])}; units with gap<1: {sum(1 for f,b in r1pairs if b["gap"]<1)}')
# r1 M14 (A1letter vs B1) as the comparable harness control there
r1L = read_jsonl('runs/r1/A1letter.jsonl'); n=a=0
for l in r1L:
    b = R.get(l['prompt_sha'])
    if b is None: continue
    n+=1; a += (fp(l)==b['pred'])
P(f'r1: A1letter vs B1 same prompt (M14 on Mac): {a}/{n}')
# b1 M14: A1letter vs any b1 logit read with identical prompt (B1 partial main k15/k16 + perm)
n=a=0; dis_m14=[]
for l in letter:
    b = b1reads.get(l['prompt_sha'])
    if b is None: continue
    n+=1; a += (fp(l)==b['pred'])
    if fp(l)!=b['pred']: dis_m14.append((l,b))
P(f'b1: A1letter vs B1/B1perm same prompt (M14 on 4090): {a}/{n}; gaps of disagreements: {[round(b["gap"],3) for l,b in dis_m14]}; by condition: {collections.Counter(b["condition"] for l,b in dis_m14)}')
open(f'{SP}/q1b_out.txt','w').write('\n'.join(out)+'\n')
