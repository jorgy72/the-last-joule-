import numpy as np, sys, os, json, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, sim2

LO = np.array([-2.0, -4.0, -8.0, -16.0, -16.0]); HI = np.array([3.0, 4.0, 8.0, 16.0, 16.0])
FEES = [0.0, 0.015, 0.05, 0.15, 0.3]

def evolve(args):
    fee, seed = args
    rng = np.random.default_rng(seed)
    mean = rng.uniform(LO, HI) * 0.5; mean[0] = rng.uniform(-0.5, 1.5); mean[1] = rng.uniform(-2, 0.5)
    std = (HI - LO) / 4
    for g in range(30):
        world = sim.make_world(500, seed * 1000 + g)
        pop = np.clip(mean + rng.normal(0, 1, (24, 5)) * std, LO, HI)
        fit = np.array([sim2.fitness2(p, world, fee) for p in pop])
        el = pop[np.argsort(fit)[-6:]]
        mean = el.mean(0); std = np.maximum(el.std(0), 0.03 * (HI - LO)) * 0.9 + 0.1 * std
    return fee, seed, mean.tolist()

if __name__ == '__main__':
    jobs = [(f, s) for f in FEES for s in (1, 2, 3)]
    t0 = time.time()
    with Pool(2) as p:
        res = p.map(evolve, jobs)
    json.dump(res, open('results/evolved2.json', 'w'), indent=1)
    print('time', time.time() - t0, flush=True)
    for r in res: print(r[0], r[1], np.round(r[2], 2))
