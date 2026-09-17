import sys, json, collections, re
sys.path.insert(0, '.')
from minijev.records import read_jsonl
from minijev.scoring import unit_rows, correct, cluster_bootstrap_delta, balanced_accuracy
from minijev.data import fetch_domains
SP = 'analysis/out'
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)
def pct(c, n): return f'{c}/{n} = {c/n:.4f}' if n else f'{c}/{n} = n/a'
i2d = fetch_domains(); BOOLD = {'banking','credit_cards'}
def kof(c): return int(re.search(r':k(\d+):', c).group(1))
def fcof(c): return c.split(':')[-1]
J = read_jsonl('runs/b1/A1joint.jsonl'); S = read_jsonl('runs/b1/A1split.jsonl')
jm = [r for r in J if r['condition'].startswith('main')]; je = [r for r in J if r['condition'].startswith('easy')]
sm = [r for r in S if r['condition'].startswith('main')]
sr = unit_rows(sm); jmr = unit_rows(jm); jer = unit_rows(je)
P('## (a) Boolean: split prompt is identical across k?')
sb = [r for r in sm if r['fields_order']==['is_banking_or_credit_cards']]
P(f'  split boolean records {len(sb)}; distinct prompt_sha {len({r["prompt_sha"] for r in sb})}; conditions {dict(collections.Counter(r["condition"] for r in sb))}; distinct text_id {len({r["text_id"] for r in sb})}')
# does the joint record's rendered boolean line depend on k? by construction user_a renders "true | false (JSON booleans)"; the joint prompt differs by intent list only
sbt = {r['text_id']: r for r in sb}
P('## (a) Boolean extended pairing: joint@f3 at every k vs the ONE split answer per text (same question, split row reused; bootstrap clusters on text_id)')
jb = [r for r in jmr if r['field']=='is_banking_or_credit_cards']
pairs = [((fcof(r['condition']), r['text_id'], 'bool'), correct(r), sbt[r['text_id']]['per_field_pred'].get('is_banking_or_credit_cards')==r['gold'], kof(r['condition']), r) for r in jb if r['text_id'] in sbt]
P(f'  pairs {len(pairs)} (joint boolean rows {len(jb)})')
def rep(sub, label):
    n=len(sub); cj=sum(p[1] for p in sub); cs=sum(p[2] for p in sub); b=sum(1 for p in sub if p[2] and not p[1]); c=sum(1 for p in sub if p[1] and not p[2])
    pt,(lo,hi)=cluster_bootstrap_delta([(p[0],p[1],p[2]) for p in sub], resamples=20000)
    jrows=[p[4] for p in sub]; srows=[{'gold':p[4]['gold'], 'pred': p[4]['gold'] if p[2] else ('true' if p[4]['gold']=='false' else 'false')} for p in sub]
    tp_j=sum(1 for p in sub if p[4]['gold']=='true' and p[1]); tp_s=sum(1 for p in sub if p[4]['gold']=='true' and p[2]); npos=sum(1 for p in sub if p[4]['gold']=='true')
    tn_j=sum(1 for p in sub if p[4]['gold']=='false' and p[1]); tn_s=sum(1 for p in sub if p[4]['gold']=='false' and p[2]); nneg=n-npos
    P(f'  {label}: n={n} joint {pct(cj,n)} split {pct(cs,n)} delta split-joint {pt*100:+.2f} pp [{lo*100:+.2f},{hi*100:+.2f}] b={b} c={c} | gold=true recall joint {pct(tp_j,npos)} split {pct(tp_s,npos)}; gold=false joint {pct(tn_j,nneg)} split {pct(tn_s,nneg)}; balanced joint {(tp_j/npos+tn_j/nneg)/2:.4f} split {(tp_s/npos+tn_s/nneg)/2:.4f}')
rep(pairs, 'all k')
for k in sorted({p[3] for p in pairs}): rep([p for p in pairs if p[3]==k], f'k={k}')
P()
P('## (b) The cross-field cue isolated: A1joint MAIN vs A1joint EASY on the SAME (text_id, k, field-condition). Same model, same text, same domain/boolean question; only the INTENT option list differs (within-domain = leaks the domain; cross-domain = does not)')
for field in ('domain','is_banking_or_credit_cards'):
    mi = {(r['text_id'], kof(r['condition']), fcof(r['condition'])): r for r in jmr if r['field']==field}
    ei = {(r['text_id'], kof(r['condition']), fcof(r['condition'])): r for r in jer if r['field']==field}
    common = sorted(set(mi)&set(ei))
    P(f'### {field}: main keys {len(mi)}, easy keys {len(ei)}, common {len(common)}')
    prs = [((k[2], k[0], field), correct(ei[k]), correct(mi[k])) for k in common]  # delta = main - easy
    n=len(prs); ce=sum(p[1] for p in prs); cm=sum(p[2] for p in prs); pt,(lo,hi)=cluster_bootstrap_delta(prs, resamples=20000)
    P(f'  all: n={n} easy(no leak) {pct(ce,n)} main(leak) {pct(cm,n)} delta main-easy {pt*100:+.2f} pp [{lo*100:+.2f},{hi*100:+.2f}] b(main right, easy wrong)={sum(1 for p in prs if p[2] and not p[1])} c(easy right, main wrong)={sum(1 for p in prs if p[1] and not p[2])}')
    for k in sorted({kk[1] for kk in common}):
        sub=[p for p,kk in zip(prs,common) if kk[1]==k]; nn=len(sub); ce=sum(p[1] for p in sub); cm=sum(p[2] for p in sub); pt,(lo,hi)=cluster_bootstrap_delta(sub, resamples=20000)
        P(f'    k={k:<3} n={nn} easy {pct(ce,nn)} main {pct(cm,nn)} delta {pt*100:+.2f} pp [{lo*100:+.2f},{hi*100:+.2f}]')
    if field=='is_banking_or_credit_cards':
        npos=sum(1 for k in common if mi[k]['gold']=='true'); P(f'  gold=true: easy {pct(sum(1 for k in common if mi[k]["gold"]=="true" and correct(ei[k])),npos)} main {pct(sum(1 for k in common if mi[k]["gold"]=="true" and correct(mi[k])),npos)}')
P()
P('## (c) A1joint EASY vs A1split (main) on the same (text_id, k): the joint arm WITHOUT the option-list leak against separate requests')
for field in ('domain','is_banking_or_credit_cards'):
    if field=='domain':
        si = {(r['text_id'], kof(r['condition'])): r for r in sr if r['field']=='domain'}
        ei = collections.defaultdict(dict)
        for r in jer:
            if r['field']=='domain': ei[(r['text_id'], kof(r['condition']))][fcof(r['condition'])] = r
    else:
        si = {(r['text_id'], k): sbt[r['text_id']] for r in jer if r['field']==field for k in [kof(r['condition'])] if r['text_id'] in sbt}
        ei = collections.defaultdict(dict)
        for r in jer:
            if r['field']==field: ei[(r['text_id'], kof(r['condition']))][fcof(r['condition'])] = r
    for fc in ('f2','f3'):
        common = sorted(k for k in ei if fc in ei[k] and k in si)
        if not common: continue
        if field=='domain':
            prs = [((fc,k[0],field), correct(ei[k][fc]), correct(si[k])) for k in common]
        else:
            prs = [((fc,k[0],field), correct(ei[k][fc]), si[k]['per_field_pred'].get(field)==ei[k][fc]['gold']) for k in common]
        n=len(prs); cj=sum(p[1] for p in prs); cs=sum(p[2] for p in prs); pt,(lo,hi)=cluster_bootstrap_delta(prs, resamples=20000)
        P(f'  {field} joint-easy@{fc} vs split: n={n} joint-easy {pct(cj,n)} split {pct(cs,n)} delta split-joint {pt*100:+.2f} pp [{lo*100:+.2f},{hi*100:+.2f}] b={sum(1 for p in prs if p[2] and not p[1])} c={sum(1 for p in prs if p[1] and not p[2])}')
P()
P('## (d) EASY: when the joint arm\'s OWN intent answer implies a WRONG domain, does its domain follow the answer or the text?')
fol=ign=oth=0; ex=[]
for r in je:
    pp=r.get('per_field_pred') or {}
    if 'domain' not in r['fields_order'] or pp.get('intent') not in i2d: continue
    imp=i2d[pp['intent']]
    if imp==r['gold']['domain']: continue
    if pp.get('domain')==imp: fol+=1
    elif pp.get('domain')==r['gold']['domain']: ign+=1
    else: oth+=1
    ex.append((r['text_id'], kof(r['condition']), pp['intent'], imp, pp.get('domain'), r['gold']['domain']))
P(f'  records where domain(pred_intent) != gold domain: {fol+ign+oth}; domain FOLLOWED own wrong intent: {fol}; domain == gold (ignored own answer): {ign}; neither: {oth}')
for e in ex: P('   ', e)
P()
P('## (e) MAIN k=16 (mixed list: 14 same-domain + 1 cross-domain intent): same question')
fol=ign=oth=0
for r in jm:
    pp=r.get('per_field_pred') or {}
    if kof(r['condition'])!=16 or 'domain' not in r['fields_order'] or pp.get('intent') not in i2d: continue
    imp=i2d[pp['intent']]
    if imp==r['gold']['domain']: continue
    if pp.get('domain')==imp: fol+=1
    elif pp.get('domain')==r['gold']['domain']: ign+=1
    else: oth+=1
P(f'  k=16 records with domain(pred_intent) != gold: {fol+ign+oth}; followed {fol}; gold {ign}; neither {oth}')
open(f'{SP}/q2b_out.txt','w').write('\n'.join(out)+'\n')
