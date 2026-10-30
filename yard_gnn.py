"""
yard_gnn.py - Spatio-Temporal GNN for forecast-informed truck parking allocation.

Pipeline (paper Sec. 4):  hierarchical yard graph -> ST-GNN encoder -> quantile head
                          -> constraint-aware allocation policy -> operational KPIs.

Run:   python yard_gnn.py                 # full run (~10-20 min on CPU)
       python yard_gnn.py --quick         # smoke test (~2 min)
Output: results/results.json, results/ours.pt, results/policy.json  (read by app.py)
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn

from yard_sim import *   # layout constants, YardEnv, policies, cem_train, eval_policy

F_IN = 8                                  # occupancy + 4 cyclical time + 3 node-type flags
QS = [0.1, 0.5, 0.9]
HZ = [3, 6, 12]                           # evaluated horizons: 15, 30, 60 min
OURS_CFG = dict(kind="st", sups=["phys", "hier"], adaptive=True, dynamic=True, Q=3)


# =============================================================== graph + features
def build_supports():
    """Physical and hierarchical adjacency over V = bays + zones + gate (row-normalised, self-loops)."""
    P, Hh = np.zeros((V, V)), np.zeros((V, V))
    for z in range(Z):
        b = z * BPZ
        for k in range(BPZ - 1):                                  # neighbouring bays in a lane
            P[b + k, b + k + 1] = P[b + k + 1, b + k] = 1
        for k in range(BPZ):                                      # bay <-> zone
            Hh[b + k, N + z] = Hh[N + z, b + k] = 1
        Hh[N + z, N + Z] = Hh[N + Z, N + z] = 1                   # zone <-> gate
        if z < Z - 1:
            P[N + z, N + z + 1] = P[N + z + 1, N + z] = 1         # neighbouring zones
    nm = lambda A: torch.tensor((A + np.eye(V)) / (A + np.eye(V)).sum(1, keepdims=True), dtype=torch.float32)
    return {"phys": nm(P), "hier": nm(Hh)}


def make_feats(occ, slots):
    """occ [S, N] -> node features [S, V, F_IN] and full occupancy [S, V] (bays, zones, gate)."""
    zone = np.stack([occ[:, BAY_ZONE == z].mean(1) for z in range(Z)], 1)
    full = np.concatenate([occ, zone, occ.mean(1, keepdims=True)], 1)
    tod, dow = 2 * np.pi * (slots % SPD) / SPD, 2 * np.pi * ((slots // SPD) % 7) / 7
    tf = np.stack([np.sin(tod), np.cos(tod), np.sin(dow), np.cos(dow)], 1)
    typ = np.zeros((V, 3)); typ[:N, 0] = 1; typ[N:N + Z, 1] = 1; typ[-1, 2] = 1
    X = np.concatenate([full[:, :, None], np.repeat(tf[:, None], V, 1),
                        np.repeat(typ[None], len(slots), 0)], 2)
    return X.astype(np.float32), full.astype(np.float32)


# =============================================================== models
class Block(nn.Module):
    """Gated temporal conv (STGCN style) + multi-support graph conv (Eq. 4) + residual."""
    def __init__(self, C, n_sup, k=3):
        super().__init__()
        self.t = nn.Conv2d(C, 2 * C, (1, k), padding=(0, k - 1))
        self.g = nn.Conv2d(C * n_sup, C, 1)
        self.bn = nn.BatchNorm2d(C)

    def forward(self, x, sups, gates):                            # x [B, C, V, T]
        h = self.t(x)[..., :x.shape[-1]]                          # causal crop
        a, b = h.chunk(2, 1); h = torch.tanh(a) * torch.sigmoid(b)
        outs = []
        for A, g in zip(sups, gates):
            o = torch.einsum("bcvt,wv->bcwt", h, A)
            outs.append(o if g is None else o * g.view(-1, 1, 1, 1))
        return torch.relu(self.bn(self.g(torch.cat(outs, 1)) + x))


class STGNN(nn.Module):
    """Configurable ST-GNN. Flags reproduce the baselines and the ablations:
       sups     subset of {'phys','hier'}        adaptive  Graph-WaveNet adaptive edges (Eq. 3)
       dynamic  state-gated hierarchical edges   Q         1 = point forecast, 3 = quantiles"""
    def __init__(self, sups=("phys", "hier"), adaptive=True, dynamic=True, Q=3, C=32, L=2, emb=10):
        super().__init__()
        self.sups, self.adaptive, self.Q = list(sups), adaptive, Q
        self.dynamic = dynamic and "hier" in self.sups
        S = build_supports()
        for k in self.sups:
            self.register_buffer("A_" + k, S[k])
        if adaptive:
            self.E1, self.E2 = nn.Parameter(torch.randn(V, emb) * .3), nn.Parameter(torch.randn(V, emb) * .3)
        n_sup = len(self.sups) + int(adaptive)
        self.inp = nn.Conv2d(F_IN, C, 1)
        self.blocks = nn.ModuleList([Block(C, n_sup) for _ in range(L)])
        self.gate = nn.Linear(1, 1)
        self.head = nn.Sequential(nn.Linear(C * T_IN, 128), nn.ReLU(), nn.Linear(128, H * Q))

    def forward(self, x):                                         # x [B, T, V, F]
        A = [getattr(self, "A_" + k) for k in self.sups]
        if self.adaptive:
            A.append(torch.softmax(torch.relu(self.E1 @ self.E2.T), 1))   # Eq. 3
        g = 2 * torch.sigmoid(self.gate(x[..., 0].mean((1, 2))[:, None]))[:, 0] if self.dynamic else None
        gates = [g if k == "hier" else None for k in self.sups] + ([None] if self.adaptive else [])
        h = self.inp(x.permute(0, 3, 2, 1))                       # [B, C, V, T]
        for b in self.blocks:
            h = b(h, A, gates)
        z = h.permute(0, 2, 1, 3).reshape(h.shape[0], V, -1)      # [B, V, C*T]
        y = torch.sigmoid(self.head(z)).view(-1, V, H, self.Q)
        return torch.sort(y, -1)[0] if self.Q > 1 else y          # non-crossing quantiles


class LSTMBase(nn.Module):
    """Node-shared LSTM baseline (no graph)."""
    def __init__(self, C=48, Q=1):
        super().__init__(); self.Q = Q
        self.l = nn.LSTM(F_IN, C, batch_first=True); self.o = nn.Linear(C, H * Q)

    def forward(self, x):
        B = x.shape[0]
        o, _ = self.l(x.permute(0, 2, 1, 3).reshape(B * V, T_IN, F_IN))
        return torch.sigmoid(self.o(o[:, -1])).view(B, V, H, self.Q)


def make_model(cfg):
    cfg = dict(cfg); kind = cfg.pop("kind")
    return LSTMBase(**cfg) if kind == "lstm" else STGNN(**cfg)


ZOO = {   # name -> config (all trained with the same pinball loss; Q=1 means q=0.5 only)
    "LSTM":               dict(kind="lstm", Q=1),
    "STGCN-style":        dict(kind="st", sups=["phys"], adaptive=False, dynamic=False, Q=1),
    "GraphWaveNet-style": dict(kind="st", sups=["phys"], adaptive=True, dynamic=False, Q=1),
    "Ours (full)":        OURS_CFG,
}
ABLATIONS = {
    "Ours - adaptive edges":     dict(OURS_CFG, adaptive=False),
    "Ours - hierarchical edges": dict(OURS_CFG, sups=["phys"], dynamic=False),
    "Ours - quantile head":      dict(OURS_CFG, Q=1),
}


class Forecaster:
    """Wrapper used inside the simulator: window [T, N] + slots -> quantiles [V, H, Q]."""
    def __init__(self, model): self.m = model.eval()

    @torch.no_grad()
    def predict(self, win, slots):
        X, _ = make_feats(win.astype(np.float32), slots)
        return self.m(torch.from_numpy(X)[None])[0].numpy()


def load_ours(path="results/ours.pt"):
    m = make_model(OURS_CFG); m.load_state_dict(torch.load(path, map_location="cpu")); return m.eval()


# =============================================================== data
def make_dataset(days=21, seed=0):
    """History generated by the yard simulator under a 'nearest bay' dispatcher."""
    occ, _ = YardEnv(seed, 0, days * SPD).run(Nearest())
    X, full = make_feats(occ, np.arange(len(occ)))
    return dict(X=torch.tensor(X), Y=torch.tensor(full), occ=occ, days=days)


def split_windows(days):
    b = [0, int(.7 * days) * SPD, int(.8 * days) * SPD, days * SPD]      # chronological 70/10/20 by day
    return [np.arange(b[i] + T_IN - 1, b[i + 1] - H) for i in range(3)]


def batch(d, idx):
    ti = torch.as_tensor(idx)
    x = d["X"][ti[:, None] + torch.arange(-T_IN + 1, 1)]                  # [B, T, V, F]
    y = d["Y"][ti[:, None] + torch.arange(1, H + 1)].permute(0, 2, 1)     # [B, V, H]
    return x, y


# =============================================================== training / evaluation
def pinball(pred, y, qt):                                                  # Eq. 5
    u = y.unsqueeze(-1) - pred
    return torch.maximum(qt * u, (qt - 1) * u).mean()


@torch.no_grad()
def _val(model, d, idx, qt):
    model.eval(); tot = 0.0
    for i in range(0, len(idx), 256):
        x, y = batch(d, idx[i:i + 256]); tot += pinball(model(x), y, qt).item() * len(x)
    return tot / len(idx)


def train(model, qs, d, tr, va, epochs, bs=64, lr=3e-3, log=print, name=""):
    qt, opt = torch.tensor(qs), torch.optim.Adam(model.parameters(), lr)
    best, best_state, curve, bad = 1e9, None, [], 0
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(tr); tl = []
        for i in range(0, len(perm), bs):
            x, y = batch(d, perm[i:i + bs])
            loss = pinball(model(x), y, qt); opt.zero_grad(); loss.backward(); opt.step(); tl.append(loss.item())
        vl = _val(model, d, va, qt); curve.append(dict(epoch=ep + 1, train=float(np.mean(tl)), val=vl))
        if vl < best: best, bad, best_state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else: bad += 1
        log(f"  [{name}] epoch {ep + 1}/{epochs}  train {np.mean(tl):.4f}  val {vl:.4f}")
        if bad >= 4: break                                                  # early stopping
    model.load_state_dict(best_state)
    return curve


@torch.no_grad()
def predict_all(fn, d, idx, bs=256):
    P, Y = [], []
    for i in range(0, len(idx), bs):
        x, y = batch(d, idx[i:i + bs]); P.append(fn(x, torch.as_tensor(idx[i:i + bs]))); Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def metrics(P, Y, qs):
    mid, out = qs.index(0.5), {}
    for h in HZ:
        e = P[:, :N, h - 1, mid] - Y[:, :N, h - 1]
        out[f"MAE@{h * 5}"] = float(np.abs(e).mean())
        out[f"RMSE@{h * 5}"] = float(np.sqrt((e ** 2).mean()))
        out[f"ZoneMAE@{h * 5}"] = float(np.abs(P[:, N:N + Z, h - 1, mid] - Y[:, N:N + Z, h - 1]).mean())
    out["MAE_avg"] = float(np.mean([out[f"MAE@{h * 5}"] for h in HZ]))
    out["ZoneMAE_avg"] = float(np.mean([out[f"ZoneMAE@{h * 5}"] for h in HZ]))
    if len(qs) == 3:                                                        # 80% interval on zone occupancy
        zy = Y[:, N:N + Z]
        out["Coverage80"] = float(((zy >= P[:, N:N + Z, :, 0]) & (zy <= P[:, N:N + Z, :, 2])).mean())
        out["Pinball"] = float(np.mean([np.maximum(q * (Y[..., None] - P)[..., i], (q - 1) * (Y[..., None] - P)[..., i]).mean()
                                        for i, q in enumerate(qs)]))
    return out


def latency_ms(model, d, idx, n=30):
    x, _ = batch(d, idx[:1]); model.eval()
    with torch.no_grad():
        model(x); t = time.perf_counter()
        for _ in range(n): model(x)
    return 1000 * (time.perf_counter() - t) / n


# =============================================================== main experiment
def main(a):
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    log = lambda *s: print(*s, flush=True)
    log(f"[1/4] Simulating {a.days} days of yard history ...")
    d = make_dataset(a.days, a.seed); tr, va, te = split_windows(a.days)
    res = dict(meta=dict(days=a.days, epochs=a.epochs, seed=a.seed, n_train=len(tr), n_test=len(te),
                         bays=N, zones=Z, nodes=V, horizons_min=[h * 5 for h in HZ]),
               forecast={}, curves={}, allocation={}, cem={})

    # --- reference baselines (no training)
    occ_full = d["Y"].numpy(); tab = np.stack([occ_full[:tr[-1] + 1][np.arange(tr[-1] + 1) % SPD == s].mean(0)
                                                for s in range(SPD)])
    ha = lambda x, ti: torch.from_numpy((tab[(ti[:, None].numpy() + np.arange(1, H + 1)) % SPD] > .5)
                                        .astype(np.float32)).permute(0, 2, 1).unsqueeze(-1)
    pers = lambda x, ti: x[:, -1, :, 0][:, :, None, None].expand(-1, -1, H, 1)
    log("[2/4] Evaluating baselines and training neural models ...")
    for name, fn in [("Persistence", pers), ("Historical average", ha)]:
        P, Y = predict_all(fn, d, te); res["forecast"][name] = dict(metrics(P, Y, [0.5]), params=0, latency_ms=0.0)

    zoo = dict(ZOO)
    if a.ablate: zoo.update(ABLATIONS)
    models = {}
    for name, cfg in zoo.items():
        m = make_model(cfg); qs = QS if cfg["Q"] == 3 else [0.5]
        t0 = time.time(); curve = train(m, qs, d, tr, va, a.epochs, log=log, name=name)
        m.eval(); P, Y = predict_all(lambda x, ti: m(x), d, te)
        res["forecast"][name] = dict(metrics(P, Y, qs), params=sum(p.numel() for p in m.parameters()),
                                     latency_ms=latency_ms(m, d, te), train_s=time.time() - t0)
        res["curves"][name] = curve; models[name] = m
    torch.save(models["Ours (full)"].state_dict(), f"{a.out}/ours.pt")

    # --- allocation: policy search in the simulator, then held-out evaluation
    log("[3/4] Training allocation policies (cross-entropy method in the yard simulator) ...")
    fc = Forecaster(models["Ours (full)"])
    w_nf, c_nf = cem_train(False, None, a.cem_iters, a.cem_pop, a.cem_eps, log=log)
    w_f, c_f = cem_train(True, fc, a.cem_iters, a.cem_pop, a.cem_eps, log=log)
    res["cem"] = {"RL without forecast": c_nf, "Ours (forecast-informed RL)": c_f}
    json.dump(dict(w=w_f.tolist(), tau=0.75), open(f"{a.out}/policy.json", "w"))

    log("[4/4] Evaluating allocation policies on held-out episodes ...")
    seeds = list(range(1000, 1000 + a.eval_seeds))
    pols = [(FCFS(), None), (Nearest(), None), (BestFit(), None),
            (Scored(w_nf, False, name="RL without forecast"), None),
            (Scored(w_f, True, name="Ours (forecast-informed RL)"), fc)]
    for p, f in pols:
        res["allocation"][p.name] = eval_policy(p, f, seeds); log(f"  {p.name}: {res['allocation'][p.name]}")
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=1)
    log(f"Done. Results in {a.out}/  ->  streamlit run app.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=21); ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--cem-iters", type=int, default=5); ap.add_argument("--cem-pop", type=int, default=10)
    ap.add_argument("--cem-eps", type=int, default=2); ap.add_argument("--eval-seeds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results")
    ap.add_argument("--ablate", action="store_true", help="also train the three ablation variants")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick: a.days, a.epochs, a.cem_iters, a.cem_pop, a.cem_eps, a.eval_seeds = 7, 2, 2, 5, 1, 2
    main(a)
