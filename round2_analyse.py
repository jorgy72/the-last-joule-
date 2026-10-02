import numpy as np, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, sim2
ev = json.load(open('results/evolved2.json'))
test = sim.make_world(3000, 9090)
bins = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20.01]
out = []
def curve(r, key):
    E = np.concatenate([d[key][d['act']] for d in r['rec']]); P = np.concatenate([d['p'][d['act']] for d in r['rec']])
    RD = np.concatenate([d['rd'][d['act']] for d in r['rec']]); ST = np.concatenate([d['steps'][d['act']] for d in r['rec']])
    res = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (E >= lo) & (E < hi)
        res.append(dict(lo=lo, hi=min(hi, 20), n=int(m.sum()), read=float(RD[m].mean()) if m.sum() > 100 else None,
                        steps=float(ST[m].mean()) if m.sum() > 100 else None))
    return res
for fee, seed, p in ev:
    r = sim2.run2(p, test, fee, record=True)
    out.append(dict(fee=fee, seed=seed, params=np.round(p, 3).tolist(), correct=float(r['correct'].mean()),
                    survival=float(r['survived'].mean()), reads=float(r['reads'].mean()), fee_spent=float(r['fees'].mean()),
                    reads_per_q=float(r['reads'].sum() / r['lifetime'].sum()),
                    by_true=curve(r, 'e'), by_belief=curve(r, 'belief')))
blind = sim2.run2((0.283, 0.0, -30, 0, 0), test, 0.0)
free = sim2.run2((0.827, -1.692, 30, 0, 0), test, 0.0)
base = dict(blind=dict(correct=float(blind['correct'].mean()), survival=float(blind['survived'].mean())),
            free_v1=dict(correct=float(free['correct'].mean()), survival=float(free['survived'].mean())))
# per-lifetime correct for CI of best-per-fee vs blind
json.dump(dict(runs=out, base=base), open('results/results2.json', 'w'), indent=1)
print(base)
for o in out:
    print(f"fee {o['fee']:.3f} s{o['seed']} correct {o['correct']:6.1f} surv {o['survival']:.3f} reads/q {o['reads_per_q']:.3f} fee/life {o['fee_spent']:.2f} a=({o['params'][0]:.2f},{o['params'][1]:.2f})")
    print('    read|true  ', [None if b['read'] is None else round(b['read'], 2) for b in o['by_true']])
    print('    read|belief', [None if b['read'] is None else round(b['read'], 2) for b in o['by_belief']])
