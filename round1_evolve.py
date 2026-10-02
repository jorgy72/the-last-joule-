import numpy as np, sys, os, json, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim

LO = np.array([-2.0, -4.0, -1.0, -3.0]); HI = np.array([3.0, 4.0, 2.5, 3.0])
CONDS = {'blind': [0, 2], 'counter': [0, 1, 2, 3], 'linear': [0, 1, 2, 3], 'sensing': [0, 1, 2, 3], 'scrambled': [0, 1, 2, 3]}

def evolve(args):
    cond, seed = args
    free = CONDS[cond]
    rng = np.random.default_rng(seed)
    mean = rng.uniform(LO, HI); mean[[i for i in range(4) if i not in free]] = 0
    std = (HI - LO) / 3
    best = (None, -1e9)
    fcond = 'stakes' if cond in ('blind', 'sensing') else cond
    for g in range(28):
        world = sim.make_world(500, seed * 1000 + g)
        pop = mean + rng.normal(0, 1, (24, 4)) * std
        pop = np.clip(pop, LO, HI)
        for i in range(4):
            if i not in free: pop[:, i] = 0
        fit = np.array([sim.fitness(p, world, fcond) for p in pop])
        el = pop[np.argsort(fit)[-6:]]
        mean = el.mean(0); std = np.maximum(el.std(0), 0.03 * (HI - LO)) * 0.9 + 0.1 * std
        for i in range(4):
            if i not in free: mean[i] = 0; std[i] = 0
    return cond, seed, mean.tolist()

if __name__ == '__main__':
    jobs = [(c, s) for c in CONDS for s in (1, 2, 3)]
    t0 = time.time()
    with Pool(2) as p:
        res = p.map(evolve, jobs)
    json.dump(res, open('results/evolved.json', 'w'), indent=1)
    print('time', time.time() - t0)
    for r in res: print(r[0], r[1], np.round(r[2], 2))
