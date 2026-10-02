import numpy as np, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim

ev = json.load(open('results/evolved.json'))
FC = {'blind': 'stakes', 'sensing': 'stakes', 'counter': 'counter', 'linear': 'linear', 'scrambled': 'scrambled'}

# 1. pick the best of the three runs per condition on a validation world
val = sim.make_world(1500, 777)
best = {}
for cond, seed, p in ev:
    f = sim.fitness(p, val, FC[cond])
    if cond not in best or f > best[cond][1]:
        best[cond] = (p, f)
pol = {k: v[0] for k, v in best.items()}
runs_fit = {}
for cond, seed, p in ev:
    runs_fit.setdefault(cond, []).append(dict(seed=seed, params=np.round(p, 3).tolist(), val_fitness=round(float(sim.fitness(p, val, FC[cond])), 2)))

# 2. held-out test in the viability world
test = sim.make_world(5000, 4242)
rng = np.random.default_rng(0)
def ci(x, B=1000):
    x = np.asarray(x, float)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
def ratio_ci(a, b, B=1000):
    a = np.asarray(a, float); b = np.asarray(b, float)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(a), len(a)); bs.append(a[i].sum() / b[i].sum())
    return [float(a.sum() / b.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

def summarise(r):
    return dict(survival=ci(r['survived']), lifetime=ci(r['lifetime']), correct=ci(r['correct']),
                useful_share=ratio_ci(r['useful'], r['spent']), settled_share=ratio_ci(r['coherent'], r['spent']),
                steps_per_task=float(r['spent'].sum() / sim.C / r['lifetime'].sum()))

results = {}
for cond in ['blind', 'sensing', 'scrambled', 'linear', 'counter']:
    r = sim.run(pol[cond], test, stakes=True, scrambled=(cond == 'scrambled'))
    results[cond] = summarise(r)
# lesion: the evolved sensing agent with its meter cut (fed another lifetime's reading)
results['sensing_lesioned'] = summarise(sim.run(pol['sensing'], test, stakes=True, scrambled=True))
# lesion 2: meter frozen at full
p = list(pol['sensing'])
r_full = sim.run([p[0], 0.0, p[2], 0.0], test, stakes=True)
results['sensing_meter_stuck_full'] = summarise(r_full)

# 3. policy curves
u = np.linspace(0, 1, 21)
curves = {}
for cond in pol:
    a0, a1, s0, s1 = pol[cond]
    curves[cond] = dict(energy=(sim.E_REF * (1 - u)).round(2).tolist(),
                        threshold=np.exp(a0 + a1 * u).round(3).tolist(),
                        skip=(s0 + s1 * u).round(3).tolist())

# 4. behaviour by energy level (viability world, own condition)
def by_energy(params, scrambled=False):
    r = sim.run(params, test, stakes=True, scrambled=scrambled, record=True)
    E = np.concatenate([d['e_before'][d['act']] for d in r['rec']])
    st = np.concatenate([d['steps'][d['act']] for d in r['rec']])
    sk = np.concatenate([d['skip'][d['act']] for d in r['rec']])
    ok = np.concatenate([d['ok'][d['act']] for d in r['rec']])
    hd = np.concatenate([d['hard'][d['act']] for d in r['rec']])
    bins = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20.01]
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (E >= lo) & (E < hi)
        if m.sum() < 200: continue
        th = m & ~sk
        spent = st[m].sum()
        out.append(dict(lo=lo, hi=hi, n=int(m.sum()), steps=float(st[th].mean()) if th.any() else None,
                        skip=float(sk[m].mean()), skip_easy=float(sk[m & ~hd].mean()), skip_hard=float(sk[m & hd].mean()),
                        acc_attempted=float(ok[th].mean()) if th.any() else None,
                        useful_share=float((st[m] * ok[m]).sum() / spent)))
    return out, r
beh = {}
recs = {}
for cond in ['sensing', 'blind', 'linear']:
    beh[cond], recs[cond] = by_energy(pol[cond])

# 5. last joule: what happens below 2 J; recover or die
def last_joule(r):
    n = len(r['lifetime'])
    E = np.stack([d['e_before'] for d in r['rec']], 1)   # (n,T)
    A = np.stack([d['act'] for d in r['rec']], 1)
    low = (E < 2) & A
    ever = low.any(1)
    died = ~r['survived']
    return dict(reached_low=float(ever.mean()), died_given_low=float(died[ever].mean()) if ever.any() else None,
                recovered_given_low=float((~died[ever]).mean()) if ever.any() else None)
lj = {c: last_joule(recs[c]) for c in recs}

# 6. example traces: same 3 lifetimes for sensing vs blind
traces = {}
for c in ['sensing', 'blind']:
    E = np.stack([d['e_before'] for d in recs[c]['rec']], 1)
    A = np.stack([d['act'] for d in recs[c]['rec']], 1)
    sel = [11, 23, 57]
    traces[c] = [[round(float(E[i, t]), 2) if A[i, t] else None for t in range(sim.T)] for i in sel]

out = dict(params={k: np.round(v, 3).tolist() for k, v in pol.items()}, runs=runs_fit, results=results, curves=curves,
           by_energy=beh, last_joule=lj, traces=traces,
           world=dict(T=sim.T, KMAX=sim.KMAX, C=sim.C, M=sim.M, R=sim.R, E0=sim.E0, E_MAX=sim.E_MAX, BLOCK=sim.BLOCK,
                      EASY=sim.EASY, HARD=sim.HARD, n_test=5000))
json.dump(out, open('results/results.json', 'w'), indent=1)
for k, v in results.items():
    print(f"{k:26s} surv {v['survival'][0]:.3f} life {v['lifetime'][0]:6.1f} correct {v['correct'][0]:6.1f} [{v['correct'][1]:.1f},{v['correct'][2]:.1f}] useful {v['useful_share'][0]:.3f} settled {v['settled_share'][0]:.3f} steps {v['steps_per_task']:.2f}")
print(json.dumps(out['params'], indent=0))
print(json.dumps(lj, indent=0))
for c in beh:
    print(c)
    for b in beh[c]: print('  ', b)
