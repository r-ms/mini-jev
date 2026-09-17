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
def fpred(r):  # per-field arms: the record's own single field
    return (r.get('per_field_pred') or {}).get(r['fields_order'][0])

free = read_jsonl('runs/b1/B1free.jsonl'); B1 = read_jsonl('runs/b1/B1.jsonl'); letter = read_jsonl('runs/b1/A1letter.jsonl'); perm = read_jsonl('runs/b1/B1perm.jsonl')
b1reads = {}
for r in B1 + perm: b1reads.setdefault(r['prompt_sha'], r)
P('## M14 recomputed with each record\'s OWN field (previous 303/1156 was a bug: helper read "intent" on domain/boolean records)')
n=a=0; dis=[]
for l in letter:
    b = b1reads.get(l['prompt_sha'])
    if b is None: continue
    n+=1; ok = (fpred(l)==b['pred']); a += ok
    if not ok: dis.append((l,b))
P(f'b1: A1letter vs B1/B1perm, byte-identical prompt: {a}/{n} = {a/n:.4f}; by field: {collections.Counter(l["fields_order"][0] for l,b in [(l,b1reads[l["prompt_sha"]]) for l in letter if l["prompt_sha"] in b1reads])}')
P(f'   disagreements: {[(l["text_id"], l["fields_order"][0], b["condition"], fpred(l), b["pred"], round(b["gap"],3)) for l,b in dis]}')
r1B1 = read_jsonl('runs/r1/B1.jsonl'); r1L = read_jsonl('runs/r1/A1letter.jsonl')
R = {}
for r in r1B1: R.setdefault(r['prompt_sha'], r)
n=a=0; dis=[]
for l in r1L:
    b = R.get(l['prompt_sha'])
    if b is None: continue
    n+=1; ok=(fpred(l)==b['pred']); a+=ok
    if not ok: dis.append((l,b))
P(f'r1: A1letter vs B1, byte-identical prompt: {a}/{n} = {a/n:.4f}; disagreements gaps: {[round(b["gap"],3) for l,b in dis]}')
P()
P('## Anatomy of the published 189/200')
easy = {r['text_id']: r for r in B1 if r['condition']=='easy:k4:f1'}
first, last = {}, {}
for r in B1:
    kk=(r['text_id'], r['field'], r['k']); first.setdefault(kk, r); last[kk]=r
P(f'  condition of the FIRST B1.jsonl k=4 intent record per control text: {collections.Counter(first[(f["text_id"],"intent",4)]["condition"] for f in free)}')
P(f'  condition of the LAST  B1.jsonl k=4 intent record per control text: {collections.Counter(last[(f["text_id"],"intent",4)]["condition"] for f in free)}')
fp = lambda f: f['per_field_pred'].get('intent')
free_ok = sum(1 for f in free if fp(f)==f['gold']['intent']); easy_ok = sum(1 for f in free if easy[f['text_id']]['pred']==easy[f['text_id']]['gold'])
both = sum(1 for f in free if fp(f)==f['gold']['intent'] and easy[f['text_id']]['pred']==easy[f['text_id']]['gold'])
same_wrong = sum(1 for f in free if fp(f)!=f['gold']['intent'] and fp(f)==easy[f['text_id']]['pred'])
P(f'  B1free correct (hard within-domain k4 set): {free_ok}/200; B1 easy:k4:f1 correct (cross-domain set, same texts): {easy_ok}/200; both correct: {both}/200; same wrong label: {same_wrong}/200 -> "agreement" {both+same_wrong}/200')
P(f'  gold letter position differs between the two option sets on {sum(1 for f in free if f["gold_pos"]["intent"]!=easy[f["text_id"]]["gold_pos"])}/200 texts; options lists share only the gold label on {sum(1 for f in free if set(f["options"]["intent"]) & set(easy[f["text_id"]]["options"]) == {f["gold"]["intent"]})}/200')
L = {r['prompt_sha']: r for r in letter}
letter_ok = sum(1 for f in free if fpred(L[f['prompt_sha']])==f['gold']['intent'])
P(f'  A1letter correct on the same 200 prompts: {letter_ok}/200; B1free correct: {free_ok}/200')
open(f'{SP}/q1c_out.txt','w').write('\n'.join(out)+'\n')
