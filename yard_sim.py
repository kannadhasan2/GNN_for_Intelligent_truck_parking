
import math
import time
import numpy as np

SLOT_MIN, SPD = 5, 288                      # 5-minute slots, 288 slots per day
T_IN, H = 12, 12                            # 1 h history, 1 h forecast horizon
Z, BPZ = 4, 6
N = Z * BPZ                                 # bays
V = N + Z + 1                               # bays + zones + gate
MAX_WAIT = 12                               # trucks leave (overflow) after 60 min of waiting
ZONE_NAMES = ["Z0 Standard-S", "Z1 Standard-L", "Z2 Reefer", "Z3 Hazmat"]
CARGO_NAMES = ["General", "Refrigerated", "Hazardous"]

BAY_ZONE = np.repeat(np.arange(Z), BPZ)
BAY_LEN = np.where(BAY_ZONE == 0, 14.0, 18.0)
ZONE_BASE = np.array([55, 20, 90, 125])                                # Z1 sits closest to the gate
BAY_DIST = ZONE_BASE[BAY_ZONE] + 6 * np.tile(np.array([3, 0, 5, 1, 4, 2]), Z)   # metres from gate
SPECIAL = BAY_ZONE >= 2                                              # reefer / hazmat bays


def rate(slot, scale=1.3):
    """Arrival intensity (trucks per slot): daily peak + quieter weekends."""
    tod, dow = (slot % SPD) / SPD, (slot // SPD) % 7
    return scale * (1 + 0.55 * np.sin(2 * np.pi * (tod - 0.3))) * (0.8 if dow >= 5 else 1.0)


def new_truck(rng, t):
    return dict(t=t, len=12.0 if rng.random() < 0.55 else 16.5,
                cargo=int(rng.choice(3, p=[0.8, 0.12, 0.08])),
                dwell=int(np.clip(rng.lognormal(math.log(14), 0.5), 3, 72)))


def feasible(tr, occ):
    ok = (~occ) & (BAY_LEN >= tr["len"])
    if tr["cargo"] == 1:
        ok &= BAY_ZONE == 2
    elif tr["cargo"] == 2:
        ok &= BAY_ZONE == 3
    return ok


class FCFS:
    name, uses_fc = "FCFS (first free bay)", False
    def choose(self, tr, feas, occ, fq, off): return int(np.flatnonzero(feas)[0])

class Nearest:
    name, uses_fc = "Nearest available", False
    def choose(self, tr, feas, occ, fq, off):
        idx = np.flatnonzero(feas); return int(idx[np.argmin(BAY_DIST[idx])])

class BestFit:
    name, uses_fc = "Best-fit heuristic", False
    def choose(self, tr, feas, occ, fq, off):
        idx = np.flatnonzero(feas)
        return int(idx[np.argmin((BAY_LEN[idx] - tr["len"]) * 1000 + BAY_DIST[idx])])


class Scored:

    def __init__(self, w, use_fc, tau=0.75, name=""):
        self.w, self.uses_fc, self.tau, self.name = np.asarray(w, float), use_fc, tau, name

    def feats(self, tr, idx, occ, fq, off):
        zone = BAY_ZONE[idx]
        zo = occ.reshape(Z, BPZ).mean(1)                              # current zone occupancy
        spec = ((tr["cargo"] == 0) & SPECIAL[idx]).astype(float)      # general truck in special bay
        zq = np.zeros(Z)
        if self.uses_fc and fq is not None:                           # q90 zone occupancy over dwell window
            h = max(1, int(min(tr["dwell"], H - off)))
            zq = fq[N:N + Z, off:off + h, 2].mean(1)
        f = np.stack([BAY_DIST[idx] / 200, (BAY_LEN[idx] - tr["len"]) / 6, spec,
                      spec * zo[zone], spec * zq[zone], zq[zone] - zo[zone]], 1)
        return f, zq

    def choose(self, tr, feas, occ, fq, off):
        idx = np.flatnonzero(feas)
        f, zq = self.feats(tr, idx, occ, fq, off)
        if self.uses_fc and fq is not None and tr["cargo"] == 0:      # risk-aware safeguard
            risky = SPECIAL[idx] & (zq[BAY_ZONE[idx]] > self.tau)
            if risky.any() and (~risky).any():
                idx, f = idx[~risky], f[~risky]
        return int(idx[np.argmin(f @ self.w)])

    def scores(self, tr, feas, occ, fq, off):
        idx = np.flatnonzero(feas); f, _ = self.feats(tr, idx, occ, fq, off)
        return idx, f @ self.w


class YardEnv:
    def __init__(self, seed, start_slot=0, n_slots=SPD, scale=1.3):
        self.seed, self.s0, self.S, self.scale = seed, start_slot, n_slots, scale

    def run(self, policy, fc=None):
        """Simulate; returns (occupancy [S, N], metrics). fc: object with predict(window, slots)."""
        rng = np.random.default_rng(self.seed)
        release = np.zeros(N); queue = []
        hist = np.zeros((T_IN + self.S, N), np.float32)
        waits, dists, pw = [], [], []
        over = viol = ndec = 0; busy = lat = 0.0
        fq, fq_t = None, -99
        use_fc = fc is not None and getattr(policy, "uses_fc", False)
        for t in range(self.S):
            slot = self.s0 + t
            for _ in range(rng.poisson(rate(slot, self.scale))):
                queue.append(new_truck(rng, t))
            occ = release > t
            if use_fc and t - fq_t >= 3:                              # refresh forecast every 15 min
                t0 = time.perf_counter()
                fq = fc.predict(hist[t:t + T_IN], np.arange(slot - T_IN, slot)); fq_t = t
                lat += time.perf_counter() - t0
            queue.sort(key=lambda x: (x["cargo"] == 0, x["t"]))       # priority cargo first
            rest = []
            for tr in queue:
                feas = feasible(tr, occ)
                if feas.any():
                    t0 = time.perf_counter()
                    i = policy.choose(tr, feas, occ, fq, t - fq_t)
                    lat += time.perf_counter() - t0; ndec += 1
                    viol += int(not feas[i])
                    release[i] = t + tr["dwell"]; occ[i] = True
                    waits.append((t - tr["t"]) * SLOT_MIN); dists.append(BAY_DIST[i])
                    if tr["cargo"]: pw.append(waits[-1])
                elif t - tr["t"] >= MAX_WAIT:
                    over += 1; waits.append(MAX_WAIT * SLOT_MIN)
                    if tr["cargo"]: pw.append(waits[-1])
                else:
                    rest.append(tr)
            queue = rest
            hist[T_IN + t] = occ; busy += occ.mean()
        w = np.array(waits) if waits else np.zeros(1)
        m = dict(mean_wait=float(w.mean()), p95_wait=float(np.percentile(w, 95)),
                 overflow_pct=100 * over / max(1, len(waits)),
                 prio_wait=float(np.mean(pw)) if pw else 0.0,
                 mean_dist=float(np.mean(dists)) if dists else 0.0,
                 util_pct=100 * busy / self.S, violations=viol,
                 latency_ms=1000 * lat / max(1, ndec), trucks=len(waits))
        return hist[T_IN:], m


def eval_policy(pol, fc, seeds):
    ms = [YardEnv(s, start_slot=(k % 7) * SPD).run(pol, fc)[1] for k, s in enumerate(seeds)]
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}


def cem_train(use_fc, fc, iters=5, pop=10, eps=2, seed=0, log=print):

    rng = np.random.default_rng(seed)
    mask = np.ones(6) if use_fc else np.array([1, 1, 1, 1, 0, 0.])
    mu, sd, curve, best = np.array([1, 1, 1, 0, 0, 0.]), np.ones(6) * 0.8, [], (1e9, None)

    def cost(w):
        ms = [YardEnv(s, 72, 144).run(Scored(w * mask, use_fc), fc)[1] for s in range(100, 100 + eps)]
        return float(np.mean([m["mean_wait"] + 0.5 * m["prio_wait"] + 5 * m["mean_dist"] / 100 + 0.3 * m["overflow_pct"] for m in ms]))

    for it in range(iters):
        cand = [mu] + [mu + sd * rng.standard_normal(6) for _ in range(pop - 1)]
        costs = np.array([cost(w) for w in cand])
        order = np.argsort(costs); elite = np.array(cand)[order[:max(2, pop // 4)]]
        mu, sd = 0.5 * mu + 0.5 * elite.mean(0), np.maximum(0.5 * sd + 0.5 * elite.std(0), 0.1)
        if costs[order[0]] < best[0]: best = (costs[order[0]], np.array(cand[order[0]]))
        curve.append(float(best[0])); log(f"  CEM iter {it + 1}/{iters}  best cost {best[0]:.2f}")
    return best[1] * mask, curve
