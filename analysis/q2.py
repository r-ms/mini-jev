import sys, json, collections, statistics, re
sys.path.insert(0, '.')
from minijev.records import read_jsonl
from minijev.scoring import unit_rows, accuracy, paired, cluster_bootstrap_delta, correct, balanced_accuracy
from minijev.data import fetch_domains
SP = 'analysis/out'
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)
def pct(c, n): return f'{c}/{n} = {c/n:.4f}' if n else f'{c}/{n} = n/a'
i2d = fetch_domains()
BOOLD = {'banking', 'credit_cards'}
def kof(cond): return int(re.search(r':k(\d+):', cond).group(1))
def fcof(cond): return cond.split(':')[-1]

joint = [r for r in read_jsonl('runs/b1/A1joint.jsonl') if r['condition'].startswith('main')]
split = [r for r in read_jsonl('runs/b1/A1split.jsonl') if r['condition'].startswith('main')]
P('# Q2 — A1joint vs A1split on the DEPENDENT fields (runs/b1, main grid)')
P()
P('## Denominators')
P(f'A1joint main records: {len(joint)}; states: {dict(collections.Counter(r["state"] for r in joint))}')
P(f'A1split main records: {len(split)}; states: {dict(collections.Counter(r["state"] for r in split))}')
jr = unit_rows(joint); sr = unit_rows(split)
P(f'A1joint unit rows: {len(jr)}; by field: {dict(collections.Counter(r["field"] for r in jr))}')
P(f'A1split unit rows: {len(sr)}; by field: {dict(collections.Counter(r["field"] for r in sr))}')
P('A1split rows by (field, field-condition) — --unique-prompts kept ONE copy of each per-field prompt, under the first field-condition that renders it:')
for f in ('intent','domain','is_banking_or_credit_cards'):
    P(f'   {f}: {dict(sorted(collections.Counter(fcof(r["condition"]) for r in sr if r["field"]==f).items()))}')
P('A1joint rows by (field, field-condition):')
for f in ('intent','domain','is_banking_or_credit_cards'):
    P(f'   {f}: {dict(sorted(collections.Counter(fcof(r["condition"]) for r in jr if r["field"]==f).items()))}')
P()
# join key uniqueness for the (text_id, field, k) pairing per field-condition
def idx(rows, field):
    d = collections.defaultdict(list)
    for r in rows:
        if r['field']==field: d[(r['text_id'], r['field'], kof(r['condition']), fcof(r['condition']))].append(r)
    return d
P('## Pairing by (text_id, field, k): split has one row per key (any f-condition); joint has one row per (key, f-condition)')
results = {}
for field in ('domain', 'is_banking_or_credit_cards'):
    sidx = collections.defaultdict(list)
    for r in sr:
        if r['field']==field: sidx[(r['text_id'], kof(r['condition']))].append(r)
    dup = sum(1 for v in sidx.values() if len(v)>1)
    P(f'### field={field}: split keys {len(sidx)}, keys with >1 split row: {dup}')
    for fc in sorted({fcof(r['condition']) for r in jr if r['field']==field}):
        jidx = {(r['text_id'], kof(r['condition'])): r for r in jr if r['field']==field and fcof(r['condition'])==fc}
        common = sorted(set(jidx) & set(sidx))
        P(f'  joint@{fc} vs split: joint keys {len(jidx)}, common {len(common)}, joint-only {len(set(jidx)-set(sidx))}, split-only {len(set(sidx)-set(jidx))}')
        if not common:
            P('  0 pairs — keys tried: (text_id, k) e.g.', list(jidx)[:3], list(sidx)[:3]); continue
        pairs = [((f'{fc}', tid, field), correct(jidx[(tid,k)]), correct(sidx[(tid,k)][0])) for tid,k in common]
        # cluster_bootstrap_delta groups by the middle element of the key = text_id -> pass (cond,tid,field)
        pairs_cb = [((fc, tid, field), ca, cb) for ((_, tid, _), ca, cb) in pairs]
        cj = sum(ca for _,ca,_ in pairs); cs = sum(cb for _,_,cb in pairs); n = len(pairs)
        b = sum(1 for _,ca,cb in pairs if cb and not ca); c = sum(1 for _,ca,cb in pairs if ca and not cb)
        point, (lo, hi) = cluster_bootstrap_delta(pairs_cb, resamples=20000)
        P(f'  n={n}  acc(joint)={pct(cj,n)}  acc(split)={pct(cs,n)}  delta split-joint={point*100:+.2f} pp [{lo*100:+.2f}, {hi*100:+.2f}] (cluster bootstrap over text_id, 20000)  discordant b(split right,joint wrong)={b} c(joint right,split wrong)={c}')
        if field=='is_banking_or_credit_cards':
            jrows = [jidx[(tid,k)] for tid,k in common]; srows = [sidx[(tid,k)][0] for tid,k in common]
            gold_pos = sum(1 for r in jrows if r['gold']=='true')
            P(f'    boolean gold balance: true {gold_pos}/{n}; balanced acc joint={balanced_accuracy(jrows):.4f} split={balanced_accuracy(srows):.4f}')
            for g in ('true','false'):
                jj=[r for r in jrows if r['gold']==g]; ss=[r for r in srows if r['gold']==g]
                P(f'    gold={g}: joint {pct(sum(map(correct,jj)),len(jj))}  split {pct(sum(map(correct,ss)),len(ss))}')
        P('  per k:')
        for k in sorted({kk for _,kk in common}):
            sub = [(key,ca,cb) for key,ca,cb in pairs_cb if any(True for _ in [0]) and (key[1],k) in jidx and kof(jidx[(key[1],k)]['condition'])==k]
            sub = [((fc,tid,field), correct(jidx[(tid,kk)]), correct(sidx[(tid,kk)][0])) for tid,kk in common if kk==k]
            nn=len(sub); cj=sum(ca for _,ca,_ in sub); cs=sum(cb for _,_,cb in sub)
            b=sum(1 for _,ca,cb in sub if cb and not ca); c=sum(1 for _,ca,cb in sub if ca and not cb)
            pt,(l,h) = cluster_bootstrap_delta(sub, resamples=20000)
            P(f'    k={k:<3} n={nn:<4} joint {pct(cj,nn)}  split {pct(cs,nn)}  delta {pt*100:+.2f} pp [{l*100:+.2f},{h*100:+.2f}]  b={b} c={c}')
        results[(field, fc)] = (n, cj, cs)
P()
P('## Cross-field cue inside A1joint records (same record: was its OWN earlier answer right?)')
def strat(recs, dep, cond_field, label):
    P(f'### {label}: {dep} accuracy by whether the same record\'s {cond_field} was right')
    rows = collections.defaultdict(list)
    for r in recs:
        if dep not in r['fields_order'] or cond_field not in r['fields_order']: continue
        pp = r.get('per_field_pred') or {}
        cond_ok = pp.get(cond_field) == r['gold'][cond_field]
        dep_ok = pp.get(dep) == r['gold'][dep]
        rows[(fcof(r['condition']), cond_ok)].append(dep_ok)
        rows[('all', cond_ok)].append(dep_ok)
    for fc in sorted({k[0] for k in rows}):
        for ok in (True, False):
            v = rows.get((fc, ok), [])
            P(f'  {fc:<4} {cond_field} {"RIGHT" if ok else "WRONG"}: {dep} acc {pct(sum(v), len(v))}')
strat(joint, 'domain', 'intent', 'A1joint main')
strat(joint, 'is_banking_or_credit_cards', 'domain', 'A1joint main (f3)')
strat(joint, 'is_banking_or_credit_cards', 'intent', 'A1joint main (f3), boolean by INTENT')
P()
P('### Self-consistency: is the joint domain simply domain(own intent answer)? and boolean = (own domain in banking/credit_cards)?')
def consist(recs, label):
    dd = collections.defaultdict(lambda: [0,0,0,0])  # [n, consistent, consistent&right, implied_right]
    bb = collections.defaultdict(lambda: [0,0,0])
    for r in recs:
        pp = r.get('per_field_pred') or {}
        fc = fcof(r['condition']); k = kof(r['condition'])
        if 'domain' in r['fields_order'] and pp.get('intent') in i2d and pp.get('domain'):
            imp = i2d[pp['intent']]
            for key in ((fc,'all'), (fc,k)):
                d = dd[key]; d[0]+=1; d[1]+= (pp['domain']==imp); d[2] += (pp['domain']==imp and pp['domain']==r['gold']['domain']); d[3] += (imp==r['gold']['domain'])
        if 'is_banking_or_credit_cards' in r['fields_order'] and pp.get('domain') and pp.get('is_banking_or_credit_cards'):
            impb = 'true' if pp['domain'] in BOOLD else 'false'
            for key in ((fc,'all'), (fc,k)):
                b = bb[key]; b[0]+=1; b[1]+=(pp['is_banking_or_credit_cards']==impb); b[2]+=(impb==r['gold']['is_banking_or_credit_cards'])
    P(f'#### {label}')
    for key in sorted(dd, key=lambda t:(t[0], str(t[1]))):
        n,cns,cr,ir = dd[key]; P(f'  domain  {key[0]} k={key[1]}: n={n} pred_domain==domain(pred_intent) {pct(cns,n)}; domain(pred_intent)==gold {pct(ir,n)}')
    for key in sorted(bb, key=lambda t:(t[0], str(t[1]))):
        n,cns,ir = bb[key]; P(f'  boolean {key[0]} k={key[1]}: n={n} pred_bool==f(pred_domain) {pct(cns,n)}; f(pred_domain)==gold {pct(ir,n)}')
consist(joint, 'A1joint main')
# split: implied-from-own-independent-answers (different requests)
P('#### A1split main — same quantities across its INDEPENDENT requests (text, k)')
sp = collections.defaultdict(dict)
for r in sr: sp[(r['text_id'], kof(r['condition']))][r['field']] = r
dn=dc=dr=0; bn=bc=0
for key, d in sp.items():
    if 'intent' in d and 'domain' in d and d['intent']['pred'] in i2d and d['domain']['pred']:
        dn+=1; imp=i2d[d['intent']['pred']]; dc += (d['domain']['pred']==imp); dr += (imp==d['domain']['gold'])
    if 'domain' in d and 'is_banking_or_credit_cards' in d and d['domain']['pred'] and d['is_banking_or_credit_cards']['pred']:
        bn+=1; impb = 'true' if d['domain']['pred'] in BOOLD else 'false'; bc += (d['is_banking_or_credit_cards']['pred']==impb)
P(f'  split: pred_domain==domain(pred_intent) {pct(dc,dn)}; domain(pred_intent)==gold {pct(dr,dn)}; pred_bool==f(pred_domain) {pct(bc,bn)}')
P()
P('## Why the intent list leaks the domain under the main (within-domain) policy: share of joint records whose intent OPTIONS all come from ONE domain')
one = collections.Counter(); tot = collections.Counter()
for r in joint:
    k = kof(r['condition']); doms = {i2d.get(o) for o in r['options']['intent']}
    tot[k]+=1; one[k]+= (len(doms)==1)
P('  ' + str({k: f'{one[k]}/{tot[k]}' for k in sorted(tot)}))
P()
P('## Control where the leak is absent: A1joint on the EASY subset (cross-domain intent distractors) — domain acc by own intent right/wrong')
easy = [r for r in read_jsonl('runs/b1/A1joint.jsonl') if r['condition'].startswith('easy')]
P(f'A1joint easy records: {len(easy)}; fields conditions: {dict(collections.Counter(fcof(r["condition"]) for r in easy))}')
strat(easy, 'domain', 'intent', 'A1joint easy')
consist(easy, 'A1joint easy')
one = collections.Counter(); tot = collections.Counter()
for r in easy:
    k = kof(r['condition']); doms = {i2d.get(o) for o in r['options']['intent']}; tot[k]+=1; one[k]+=(len(doms)==1)
P('  easy: intent options all one domain: ' + str({k: f'{one[k]}/{tot[k]}' for k in sorted(tot)}))
P()
P('## Split-arm stratified the same way (its intent answer is from an INDEPENDENT request on the same text/k) — text-difficulty baseline, not conditioning')
rows = collections.defaultdict(list)
for key, d in sp.items():
    if 'intent' in d and 'domain' in d:
        rows[correct(d['intent'])].append(correct(d['domain']))
for ok in (True, False):
    v = rows[ok]; P(f'  split intent {"RIGHT" if ok else "WRONG"}: domain acc {pct(sum(v),len(v))}')
rows = collections.defaultdict(list)
for key, d in sp.items():
    if 'domain' in d and 'is_banking_or_credit_cards' in d:
        rows[correct(d['domain'])].append(correct(d['is_banking_or_credit_cards']))
for ok in (True, False):
    v = rows[ok]; P(f'  split domain {"RIGHT" if ok else "WRONG"}: boolean acc {pct(sum(v),len(v))}')
P()
P('## Decomposition of the paired delta by the JOINT record\'s own intent correctness (domain, joint@f2 vs split; joint@f3 vs split) and by own domain correctness (boolean, joint@f3 vs split)')
def decomp(field, cond_field, fc):
    jidx = {}
    for r in joint:
        if fcof(r['condition'])!=fc or field not in r['fields_order']: continue
        jidx[(r['text_id'], kof(r['condition']))] = r
    sidx = {}
    for r in sr:
        if r['field']==field: sidx[(r['text_id'], kof(r['condition']))] = r
    common = sorted(set(jidx)&set(sidx))
    st = collections.defaultdict(lambda: [0,0,0])
    for key in common:
        r = jidx[key]; pp = r.get('per_field_pred') or {}
        ok = pp.get(cond_field)==r['gold'][cond_field]
        jr_ok = pp.get(field)==r['gold'][field]; sr_ok = correct(sidx[key])
        s = st[ok]; s[0]+=1; s[1]+=jr_ok; s[2]+=sr_ok
    n = len(common)
    P(f'### {field}, joint@{fc} vs split, n={n}')
    for ok in (True, False):
        m, j, s = st[ok]
        P(f'  joint {cond_field} {"RIGHT" if ok else "WRONG"}: n={m} ({m/n:.3f} of pairs)  joint {field} {pct(j,m)}  split {field} {pct(s,m)}  contribution to total delta(split-joint) = {(s-j)/n*100:+.2f} pp')
    tj = sum(v[1] for v in st.values()); ts = sum(v[2] for v in st.values())
    P(f'  total: joint {pct(tj,n)} split {pct(ts,n)} delta {(ts-tj)/n*100:+.2f} pp')
decomp('domain','intent','f2'); decomp('domain','intent','f3'); decomp('is_banking_or_credit_cards','domain','f3'); decomp('is_banking_or_credit_cards','intent','f3')
open(f'{SP}/q2_out.txt','w').write('\n'.join(out)+'\n')
