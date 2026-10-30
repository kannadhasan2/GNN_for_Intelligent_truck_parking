"""
app.py - dashboard:  streamlit run app.py
Tabs: Overview | Forecast comparison | Allocation comparison | Live yard & dispatch | Training
"""
import json, os, subprocess, sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from yard_sim import (N, Z, BPZ, BAY_ZONE, BAY_LEN, BAY_DIST, ZONE_NAMES, CARGO_NAMES, feasible,
                      Scored, T_IN, H, SLOT_MIN)

RES = "results"
st.set_page_config(page_title="Yard-GNN | Truck Parking Allocation", page_icon="🚛", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.4rem;max-width:1250px}
.hero{background:linear-gradient(120deg,#0f2742 0%,#1d4e89 60%,#2a9d8f 100%);color:#fff;padding:26px 32px;
      border-radius:16px;margin-bottom:18px}
.hero h1{margin:0;font-size:1.8rem;color:#fff}.hero p{margin:6px 0 0;opacity:.88}
.card{background:var(--secondary-background-color);border-radius:14px;padding:14px 18px;border-left:5px solid #2a9d8f}
.card .l{font-size:.78rem;opacity:.7;text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:1.65rem;font-weight:700}.card .d{font-size:.85rem;color:#2a9d8f;font-weight:600}
.card.bad{border-left-color:#e76f51}.card.bad .d{color:#e76f51}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🚛 Spatio-Temporal GNN for Truck Parking Allocation</h1>'
            '<p>Hierarchical yard graph · probabilistic multi-horizon forecasts · constraint-aware bay assignment</p></div>',
            unsafe_allow_html=True)


def card(label, value, delta=None, good=True):
    d = f'<div class="d">{delta}</div>' if delta else ""
    return f'<div class="card {"" if good else "bad"}"><div class="l">{label}</div><div class="v">{value}</div>{d}</div>'


@st.cache_data
def load_results():
    p = f"{RES}/results.json"
    return json.load(open(p)) if os.path.exists(p) else None


@st.cache_resource
def load_model():
    import torch
    from yard_gnn import load_ours, Forecaster
    return Forecaster(load_ours(f"{RES}/ours.pt"))


@st.cache_resource
def load_data(days, seed):
    from yard_gnn import make_dataset
    return make_dataset(days, seed)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Experiment")
    quick = st.toggle("Quick run (smoke test)", value=True)
    days = st.slider("Simulated days", 7, 42, 21, disabled=quick)
    epochs = st.slider("Epochs", 2, 40, 12, disabled=quick)
    ablate = st.checkbox("Include ablations", value=False)
    if st.button("▶ Run full pipeline", type="primary", use_container_width=True):
        cmd = [sys.executable, "yard_gnn.py", "--out", RES] + (["--quick"] if quick else
               ["--days", str(days), "--epochs", str(epochs)]) + (["--ablate"] if ablate else [])
        box = st.empty(); log = []
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as pr:
            for line in pr.stdout:
                log.append(line.rstrip()); box.code("\n".join(log[-12:]))
        load_results.clear(); load_model.clear(); st.rerun()
    st.caption("Training runs `yard_gnn.py`; results are cached in `results/`.")

R = load_results()
if R is None:
    st.info("No results yet. Run the pipeline from the sidebar, or `python yard_gnn.py --quick` in a terminal.")
    st.stop()

F = pd.DataFrame(R["forecast"]).T
A = pd.DataFrame(R["allocation"]).T
OURS, OURS_A = "Ours (full)", "Ours (forecast-informed RL)"
t_overview, t_fc, t_al, t_live, t_train = st.tabs(
    ["📊 Overview", "📈 Forecast comparison", "🚚 Allocation comparison", "🗺️ Live yard & dispatch", "🧠 Training"])

# ------------------------------------------------------------------ overview
with t_overview:
    base_f = F.drop(index=[OURS], errors="ignore")["MAE_avg"]
    base_a = A.drop(index=[OURS_A], errors="ignore")
    c = st.columns(4)
    g = 100 * (1 - F.loc[OURS, "MAE_avg"] / base_f.min())
    c[0].markdown(card("Avg bay MAE (15–60 min)", f"{F.loc[OURS, 'MAE_avg']:.3f}",
                       f"{g:+.1f}% vs best baseline ({base_f.idxmin()})", g > 0), unsafe_allow_html=True)
    c[1].markdown(card("80% interval coverage (zones)", f"{100 * F.loc[OURS, 'Coverage80']:.1f}%", "nominal 80%"),
                  unsafe_allow_html=True)
    gw = 100 * (1 - A.loc[OURS_A, "mean_wait"] / base_a["mean_wait"].min())
    c[2].markdown(card("Mean waiting time", f"{A.loc[OURS_A, 'mean_wait']:.1f} min",
                       f"{gw:+.1f}% vs best baseline", gw > 0), unsafe_allow_html=True)
    go_ = 100 * (1 - A.loc[OURS_A, "prio_wait"] / base_a["prio_wait"].min())
    c[3].markdown(card("Priority-cargo wait", f"{A.loc[OURS_A, 'prio_wait']:.1f} min",
                       f"{go_:+.1f}% vs best baseline", go_ > 0), unsafe_allow_html=True)
    st.write("")
    l, r = st.columns([1.1, 1])
    with l:
        st.subheader("Hierarchical yard graph")
        fig = go.Figure()
        gx, gy = 0, 3.4
        zx = np.linspace(-4.5, 4.5, Z)
        cols = ["#4c78a8", "#59a14f", "#2a9d8f", "#e76f51"]
        for z in range(Z):
            fig.add_trace(go.Scatter(x=[gx, zx[z]], y=[gy, 2], mode="lines", line=dict(color="#999", width=2), showlegend=False))
            bx = zx[z] + np.linspace(-1, 1, BPZ)
            for b in range(BPZ):
                fig.add_trace(go.Scatter(x=[zx[z], bx[b]], y=[2, 0.4], mode="lines", line=dict(color="#ccc", width=1), showlegend=False))
            fig.add_trace(go.Scatter(x=bx, y=[0.4] * BPZ, mode="markers", marker=dict(size=13, color=cols[z]),
                                     name=ZONE_NAMES[z], hovertext=[f"Bay {z * BPZ + b} · {BAY_LEN[z * BPZ + b]:.0f} m" for b in range(BPZ)]))
            fig.add_trace(go.Scatter(x=[zx[z]], y=[2], mode="markers+text", text=[f"Z{z}"], textposition="top center",
                                     marker=dict(size=22, color=cols[z], symbol="square"), showlegend=False))
        fig.add_trace(go.Scatter(x=[gx], y=[gy], mode="markers+text", text=["Gate"], textposition="top center",
                                 marker=dict(size=30, color="#264653", symbol="diamond"), showlegend=False))
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False, range=[0, 4.2]),
                          legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig, use_container_width=True)
    with r:
        st.subheader("What this project does")
        st.markdown(f"""
* **Graph:** {R['meta']['nodes']} nodes (gate → {Z} zones → {N} bays) with *physical*, *hierarchical* and *adaptive* edges.
* **Forecast:** gated temporal conv + multi-support graph conv → **q10 / q50 / q90** occupancy for 15/30/60 min.
* **Allocate:** feasibility-masked scoring policy with a forecast-driven **risk threshold τ**.
* **Train:** pinball loss for the encoder; cross-entropy-method policy search in the simulator.
* **Data:** simulator calibrated for TPIMS-like cycles ({R['meta']['days']} days, {R['meta']['n_train']} train windows).
""")
        st.caption("Numbers shown come from *your* run of the pipeline; they are experimental outputs, not paper claims.")

# ------------------------------------------------------------------ forecast comparison
with t_fc:
    st.subheader("Forecast accuracy on the chronological test split")
    hz = R["meta"]["horizons_min"]
    show = F[[f"MAE@{h}" for h in hz] + [f"RMSE@{h}" for h in hz] + ["ZoneMAE_avg", "Coverage80", "params", "latency_ms"]]
    st.dataframe(show.style.highlight_min(subset=[c for c in show.columns if "MAE" in c or "RMSE" in c], color="#b7e4c7")
                 .format("{:.3f}", na_rep="–"), use_container_width=True)
    c1, c2 = st.columns(2)
    long = pd.DataFrame([dict(model=m, horizon=f"{h} min", MAE=F.loc[m, f"MAE@{h}"]) for m in F.index for h in hz])
    c1.plotly_chart(px.bar(long, x="horizon", y="MAE", color="model", barmode="group", title="Bay-level MAE by horizon (lower is better)")
                    .update_layout(height=380, legend_title=None), use_container_width=True)
    lines = pd.DataFrame([dict(model=m, horizon=h, ZoneMAE=F.loc[m, f"ZoneMAE@{h}"]) for m in F.index for h in hz])
    c2.plotly_chart(px.line(lines, x="horizon", y="ZoneMAE", color="model", markers=True, title="Zone-occupancy MAE vs horizon")
                    .update_layout(height=380, legend_title=None), use_container_width=True)
    c3, c4 = st.columns(2)
    nn_ = F[F["params"] > 0]
    c3.plotly_chart(px.scatter(nn_.reset_index(), x="latency_ms", y="MAE_avg", size="params", color="index",
                               title="Accuracy vs inference latency (bubble = parameters)", labels={"index": "model"})
                    .update_layout(height=360), use_container_width=True)
    c4.info("**Reading the table.** Bay occupancy is binary, so bay MAE is the error rate of the median forecast; "
            "zone occupancy is continuous and shows quantile calibration. Coverage80 should be ≈ 0.80 for a calibrated "
            "q10–q90 interval. Persistence is a strong short-horizon baseline, so the gap should widen at 60 min.")

# ------------------------------------------------------------------ allocation comparison
with t_al:
    st.subheader("Operational KPIs on held-out simulated days")
    cols = ["mean_wait", "p95_wait", "prio_wait", "overflow_pct", "mean_dist", "util_pct", "violations", "latency_ms"]
    names = {"mean_wait": "Mean wait (min)", "p95_wait": "P95 wait (min)", "prio_wait": "Priority wait (min)",
             "overflow_pct": "Overflow (%)", "mean_dist": "Search dist (m)", "util_pct": "Utilization (%)",
             "violations": "Constraint violations", "latency_ms": "Latency (ms/decision)"}
    st.dataframe(A[cols].rename(columns=names).style.highlight_min(
        subset=[names[c] for c in cols if c not in ("util_pct", "latency_ms")], color="#b7e4c7").format("{:.2f}"),
        use_container_width=True)
    pick = st.multiselect("KPIs to plot", [names[c] for c in cols], default=[names[c] for c in cols[:4]])
    inv = {v: k for k, v in names.items()}
    grid = st.columns(2)
    for i, p in enumerate(pick):
        d = A[[inv[p]]].reset_index().rename(columns={"index": "policy", inv[p]: p})
        grid[i % 2].plotly_chart(px.bar(d, x="policy", y=p, color="policy", title=p).update_layout(height=330, showlegend=False,
                                 xaxis_title=None), use_container_width=True)
    st.info("Hard constraints (length, cargo class, free bay) are enforced by the feasibility mask, so violations are 0 for every policy. "
            "The forecast-informed policy differs from *RL without forecast* only by the q90 zone-occupancy features and τ.")

# ------------------------------------------------------------------ live yard
with t_live:
    if not os.path.exists(f"{RES}/ours.pt"):
        st.warning("Model checkpoint not found."); st.stop()
    fc = load_model(); d = load_data(R["meta"]["days"], R["meta"]["seed"])
    pol = json.load(open(f"{RES}/policy.json"))
    from yard_gnn import split_windows, make_feats
    te = split_windows(R["meta"]["days"])[2]
    l, r = st.columns([1, 1.5])
    with l:
        st.subheader("Select a moment")
        k = st.slider("Test-set time index", 0, len(te) - 1, len(te) // 3)
        hsel = st.select_slider("Forecast horizon", [3, 6, 12], value=6, format_func=lambda h: f"{h * SLOT_MIN} min")
        t = int(te[k]); win = d["occ"][t - T_IN + 1:t + 1]
        q = fc.predict(win, np.arange(t - T_IN + 1, t + 1))
        st.caption(f"Day {t // 288 + 1}, {(t % 288) * 5 // 60:02d}:{(t % 288) * 5 % 60:02d}")
        st.subheader("Dispatch a truck")
        length = st.radio("Trailer length", [12.0, 16.5], horizontal=True, format_func=lambda x: f"{x} m")
        cargo = st.selectbox("Cargo class", range(3), format_func=lambda i: CARGO_NAMES[i])
        dwell = st.slider("Expected dwell (slots of 5 min)", 3, 24, 12)
    occ_now = d["occ"][t].astype(bool)
    with r:
        def grid(vals, title, scale, zmax=1.0):
            m = np.asarray(vals)[:N].reshape(Z, BPZ)
            f = go.Figure(go.Heatmap(z=m, zmin=0, zmax=zmax, colorscale=scale, showscale=False, xgap=3, ygap=3,
                                     text=[[f"{v:.2f}" for v in row] for row in m], texttemplate="%{text}"))
            f.update_layout(title=title, height=230, margin=dict(l=0, r=0, t=40, b=0),
                            yaxis=dict(tickvals=list(range(Z)), ticktext=[f"Z{z}" for z in range(Z)], autorange="reversed"),
                            xaxis=dict(visible=False))
            return f
        a, b, c3 = st.columns(3)
        a.plotly_chart(grid(occ_now.astype(float), "Occupied now", "Blues"), use_container_width=True)
        b.plotly_chart(grid(q[:N, hsel - 1, 1], f"q50 @ +{hsel * 5} min", "Greens"), use_container_width=True)
        c3.plotly_chart(grid(q[:N, hsel - 1, 2], f"q90 (risk) @ +{hsel * 5} min", "Oranges"), use_container_width=True)
        truck = dict(len=length, cargo=cargo, dwell=dwell, t=0)
        feas = feasible(truck, occ_now)
        if not feas.any():
            st.error("No feasible bay right now: the truck would join the waiting queue.")
        else:
            p = Scored(pol["w"], True, pol["tau"])
            best = p.choose(truck, feas, occ_now, q, 0); idx, sc = p.scores(truck, feas, occ_now, q, 0)
            st.success(f"Assign **bay {best}** · {ZONE_NAMES[BAY_ZONE[best]]} · {BAY_LEN[best]:.0f} m · {BAY_DIST[best]:.0f} m from gate")
            sd = pd.DataFrame(dict(bay=idx, zone=[ZONE_NAMES[BAY_ZONE[i]] for i in idx], cost=sc,
                                   q90_risk=q[idx, :min(dwell, H), 2].mean(1)))
            st.plotly_chart(px.bar(sd.sort_values("cost"), x="bay", y="cost", color="zone", hover_data=["q90_risk"],
                                   title="Policy cost of each feasible bay (lowest wins)").update_layout(height=300),
                            use_container_width=True)

# ------------------------------------------------------------------ training
with t_train:
    c1, c2 = st.columns(2)
    cv = pd.DataFrame([dict(model=m, epoch=e["epoch"], val_loss=e["val"]) for m, cs in R["curves"].items() for e in cs])
    c1.plotly_chart(px.line(cv, x="epoch", y="val_loss", color="model", markers=True, title="Validation pinball loss")
                    .update_layout(height=380), use_container_width=True)
    cm = pd.DataFrame([dict(policy=k, iter=i + 1, cost=v) for k, cs in R["cem"].items() for i, v in enumerate(cs)])
    c2.plotly_chart(px.line(cm, x="iter", y="cost", color="policy", markers=True, title="Policy search (CEM) best cost")
                    .update_layout(height=380), use_container_width=True)
    st.json(R["meta"], expanded=False)
