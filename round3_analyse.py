import numpy as np, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, sim2
ev = json.load(open('results/evolved3.json')); ev2 = json.load(open('results/evolved2.json'))
test = sim.make_world(3000, 9090)
edges = [0, 4, 8, 12, 16, 20.01]
out = []
for fee, seed, p in ev:
    r = sim2.run2_band(p[:2], p[2:], test, fee, record=True)
    B = np.concatenate([d['belief'][d['act']] for d in r['rec']]); RD = np.concatenate([d['rd'][d['act']] for d in r['rec']])
    E = np.concatenate([d['e'][d['act']] for d in r['rec']])
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (B >= lo) & (B < hi); mt = (E >= lo) & (E < hi)
        bands.append(dict(lo=lo, hi=min(hi, 20), share=float(m.mean()), read=float(RD[m].mean()) if m.sum() > 300 else None,
                          read_true=float(RD[mt].mean()) if mt.sum() > 300 else None,
                          rule=float(1 / (1 + np.exp(-p[2 + len(bands)])))))
    out.append(dict(fee=fee, seed=seed, params=np.round(p, 3).tolist(), correct=float(r['correct'].mean()), survival=float(r['survived'].mean()),
                    reads_per_q=float(r['reads'].sum() / r['lifetime'].sum()), bands=bands))
v2best = {}
for fee, seed, p in ev2:
    if fee in (0.05, 0.15):
        c = float(sim2.run2(p, test, fee)['correct'].mean())
        v2best.setdefault(fee, []).append(c)
json.dump(dict(runs=out, v2=v2best), open('results/results3.json', 'w'), indent=1)
print('v2 quadratic same test world', v2best)
for o in out:
    print(f"fee {o['fee']} s{o['seed']} correct {o['correct']:.1f} surv {o['survival']:.3f} reads/q {o['reads_per_q']:.3f}")
    print('   rule  ', [round(b['rule'], 2) for b in o['bands']])
    print('   share ', [round(b['share'], 3) for b in o['bands']])
    print('   true  ', [None if b['read_true'] is None else round(b['read_true'], 2) for b in o['bands']])
