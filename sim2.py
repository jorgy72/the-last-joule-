"""The Last Joule v2: the meter costs energy to read.

Same world as sim.py. The agent no longer sees its energy for free. It holds the last
reading it took (starting with an accurate reading of E0). Before each question it
decides whether to read the meter again, with probability
    p_read = sigmoid(c0 + c1*u + c2*u^2),   u = deficit implied by its LAST reading,
so the rule can come out rising, falling, U-shaped or inverted-U. Each reading costs FEE J.
Thinking threshold uses the believed deficit: theta = exp(a0 + a1*u_belief).
Skipping is switched off (it never evolved in v1).
"""
import numpy as np
from sim import make_world, T, KMAX, C, M, R, E0, E_REF, E_MAX


def run2(params, world, fee, record=False):
    a0, a1, c0, c1, c2 = params
    def pfun(belief):
        u_b = np.clip(1 - belief / E_REF, 0, 1)
        z = c0 + c1 * u_b + c2 * u_b * u_b
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    return _run(a0, a1, pfun, world, fee, record)


def run2_band(a, logits, world, fee, record=False):
    edges = np.array([4.0, 8.0, 12.0, 16.0]); lg = np.asarray(logits, float)
    def pfun(belief):
        z = lg[np.searchsorted(edges, belief, side='right')]
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    return _run(a[0], a[1], pfun, world, fee, record)


def _run(a0, a1, pfun, world, fee, record=False):
    n = world['n']
    mu, sign, noise = world['mu'], world['sign'], world['noise']
    rng = np.random.default_rng(world['perm'][0] + 17)   # read coin flips, fixed per world
    coins = rng.random((n, T))
    e = np.full(n, E0)
    belief = np.full(n, E0)
    alive = np.ones(n, bool)
    correct = np.zeros(n); spent = np.zeros(n); useful = np.zeros(n); reads = np.zeros(n); fees = np.zeros(n)
    lifetime = np.full(n, T)
    rec = [] if record else None
    for t in range(T):
        p = pfun(belief)
        rd = (coins[:, t] < p) & alive
        e = e - fee * rd
        belief = np.where(rd, e, belief)
        u = np.clip(1 - belief / E_REF, 0, 1)
        theta = np.exp(a0 + a1 * u)
        x = mu[:, t, None] + noise[:, t, :]
        S = np.cumsum(x, axis=1)
        hit = np.abs(S) >= theta[:, None]
        k_stop = np.where(hit.any(1), hit.argmax(1), KMAX - 1)
        steps = k_stop + 1
        S_final = S[np.arange(n), k_stop]
        ans = np.sign(S_final); ans[ans == 0] = 1
        ok = ans == sign[:, t]
        e_prev = e.copy()
        cost = M + C * steps
        e = np.where(alive, np.minimum(e - cost + R * ok, E_MAX), e)
        act = alive.copy()
        correct += ok * act; spent += C * steps * act; useful += C * steps * ok * act
        reads += rd; fees += fee * rd
        if record:
            rec.append(dict(e=e_prev, p=p, rd=rd, steps=steps, act=act, belief=belief.copy()))
        died = alive & (e <= 0)
        lifetime[died] = t + 1
        alive &= ~died
    out = dict(correct=correct, spent=spent, useful=useful, reads=reads, fees=fees, lifetime=lifetime, survived=alive)
    if record:
        out['rec'] = rec
    return out


def fitness2(params, world, fee):
    return run2(params, world, fee)['correct'].mean()
