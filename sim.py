"""Last-joule toy world.

Each lifetime is a run of up to T tasks. A task hides a sign (+/-) behind noisy evidence.
Thinking = drawing one evidence sample, costing C joules. The agent stops when its running
evidence |S| reaches a threshold theta, or at KMAX samples, and answers sign(S).
After the first sample it may abandon the task ("skip") if |S| < s.
Living costs M joules per task. A correct answer pays R joules.
Viability: in the stakes world, energy <= 0 means death: no further tasks.

Policy (4 numbers): theta = exp(a0 + a1*u), skip if |S1| < s0 + s1*u,
where u = deficit = clip(1 - e/E_REF, 0, 1) is what the meter reports.
Blind policies have a1 = s1 = 0. A scrambled meter feeds u from a random other lifetime.
"""
import numpy as np

T = 400          # tasks per lifetime
KMAX = 20        # max thinking steps per task
C = 0.15         # J per thinking step
M = 1.10         # J living cost per task
R = 2.00         # J reward for a correct answer
E0 = 10.0        # starting energy
E_REF = 20.0     # meter scale = storage cap
E_MAX = 20.0     # energy can't be stockpiled above this
BLOCK = 50       # lean / rich seasons alternate every 50 tasks
EASY, HARD = 0.8, 0.25


def make_world(n, seed):
    rng = np.random.default_rng(seed)
    season_lean = (np.arange(T) // BLOCK) % 2 == 1                     # (T,)
    p_hard = np.where(season_lean, 0.7, 0.3)
    hard = rng.random((n, T)) < p_hard[None]
    mag = np.where(hard, HARD, EASY)
    sign = np.where(rng.random((n, T)) < 0.5, -1.0, 1.0)
    noise = rng.normal(0, 1, (n, T, KMAX))
    perm = rng.permutation(n)                                         # for scrambled meter
    return dict(mu=mag * sign, sign=sign, noise=noise, hard=hard, perm=perm, n=n)


def run(params, world, stakes=True, scrambled=False, record=False):
    a0, a1, s0, s1 = params
    n = world['n']
    mu, sign, noise = world['mu'], world['sign'], world['noise']
    e = np.full(n, E0)
    alive = np.ones(n, bool)
    correct = np.zeros(n)
    spent_think = np.zeros(n)
    coherent = np.zeros(n)
    useful = np.zeros(n)
    lifetime = np.full(n, T)
    rec = [] if record else None
    for t in range(T):
        meter_e = e[world['perm']] if scrambled else e
        u = np.clip(1 - meter_e / E_REF, 0, 1)
        theta = np.exp(a0 + a1 * u)
        skip_thr = s0 + s1 * u
        x = mu[:, t, None] + noise[:, t, :]                            # (n,K)
        S = np.cumsum(x, axis=1)
        hit = np.abs(S) >= theta[:, None]
        k_stop = np.where(hit.any(1), hit.argmax(1), KMAX - 1)         # 0-based index of last sample used
        skip = np.abs(S[:, 0]) < skip_thr
        steps = np.where(skip, 1, k_stop + 1)
        S_final = S[np.arange(n), k_stop]
        ans = np.sign(S_final); ans[ans == 0] = 1
        ok = (~skip) & (ans == sign[:, t])
        # coherent steps: samples up to the last change in the running answer (only on correct answers)
        run_sign = np.sign(S); run_sign[run_sign == 0] = 1
        idx = np.arange(KMAX)[None]
        used = idx <= k_stop[:, None]
        changes = (run_sign[:, 1:] != run_sign[:, :-1]) & used[:, 1:]
        last_change = np.where(changes.any(1), KMAX - 1 - np.argmax(changes[:, ::-1], 1), 0)  # 0-based index
        coh_steps = np.where(ok, last_change + 1, 0)
        act = alive if stakes else np.ones(n, bool)
        e_prev = e.copy()
        cost = M + C * steps
        e = np.where(act, np.minimum(e - cost + R * ok, E_MAX), e)
        correct += ok * act
        spent_think += C * steps * act
        coherent += C * coh_steps * act
        useful += C * steps * ok * act
        if record:
            rec.append(dict(e_before=e_prev, steps=steps, skip=skip,
                            ok=ok, coh=coh_steps, act=act.copy(), hard=world['hard'][:, t]))
        if stakes:
            died = alive & (e <= 0)
            lifetime[died] = t + 1
            alive &= ~died
    net = e - E0
    out = dict(correct=correct, spent=spent_think, coherent=coherent, useful=useful, lifetime=lifetime,
               survived=alive if stakes else np.ones(n, bool), net=net)
    if record:
        out['rec'] = rec
    return out


def fitness(params, world, cond):
    if cond == 'counter':        # meter visible, energy has no consequence: score = correct answers
        r = run(params, world, stakes=False)
        return r['correct'].mean()
    if cond == 'linear':         # meter visible, energy is a cost but no death: score = net energy
        r = run(params, world, stakes=False)
        return (R * r['correct'] - r['spent'] - M * T).mean()
    r = run(params, world, stakes=True, scrambled=(cond == 'scrambled'))
    return r['correct'].mean()   # viability worlds: correct answers over a lifetime (death forfeits the rest)
