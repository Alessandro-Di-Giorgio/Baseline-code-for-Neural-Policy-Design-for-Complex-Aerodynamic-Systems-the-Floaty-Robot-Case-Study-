"""
Ensemble-LQR residual (ELQR): a model-based alternative to the NN residual.

Pipeline
  1. Load the same ensemble you train the NN on (Ensemble_29_6_26).
  2. For every design member, simulate the existing LQR + exploration noise
     (warm-up from your fitted U distribution, white process noise) and
     record (x_hist, u_hist) -> x_next from the deterministic member model.
  3. Fit an affine 5-step history model per member (ridge LS), average the
     members -> linearisation of the ensemble mean.
  4. Design the residual with a DARE using your training Q, R on the TOTAL u,
     with the existing LQR inside the loop (elqr_design.design_residual).
  5. Check closed-loop stability on every member's linearisation.
  6. Export elqr_gains.h (ESP32) and elqr_design.json (for
     compare_controllers.py, which does the LQR / LQR+NN / LQR+ELQR
     comparison with your exact post-training protocol).

No training. Runs in a few minutes on CPU.
"""

import os
import json
import joblib

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn

import elqr_design as ed

# ============================================================
# Configuration (paths identical to your training script)
# ============================================================

OUTER_FOLDER = "Floaty_cluster_repo/Cluster_experiments_Floaty"
DYNAMICS_ENSEMBLE_DIR = os.path.join(OUTER_FOLDER, "Dynamical_models", "Ensemble_29_6_26")
STATS_JSON_PATH = os.path.join(OUTER_FOLDER, "Training_flight_data", "floaty_dataset_limits_cluster.json")
OUT_DIR = os.path.join(OUTER_FOLDER, "Trained_policies", "Ensemble_LQR_residual")

USE_BEST_MODEL = True
DESIGN_IDS = list(range(0, 11))     # members used to build the controller (train + val)
TEST_IDS = [11, 12]                 # held out, as in your evaluation
ALL_IDS = DESIGN_IDS + TEST_IDS

SEED = 4
DT = 0.02
SEQ_LEN = 5
N_X, N_U = 12, 4

# ---- data collection for identification
ID_BATCH = 256
ID_STEPS = 400
ID_SIGMA_U = 0.04                  # exploration noise on the residual channel
ID_FIRST_CONSTANT_STEPS = 5
ID_EXTRA_RANDOM_STEPS = 45
ID_RIDGE = 1e-4

PROCESS_NOISE_STD = np.array([
    0.001, 0.001, 0.001,
    0.005, 0.005, 0.005,
    0.002, 0.002, 0.002,
    0.050, 0.050, 0.050,
], dtype=np.float32)

# ---- design (training cost of the NN, so the comparison is fair)
Q_DESIGN = np.diag([50, 80, 10, 20, 50, 10, 0.02, 0.1, 0.7, 0.02, 0.1, 0.1]).astype(np.float64)
Q_DESIGN[[3, 4, 5, 9, 10, 11], [3, 4, 5, 9, 10, 11]] *= 10.0
R_DESIGN = 25.0 * np.diag([50, 10, 5, 5]).astype(np.float64)

USE_FEEDFORWARD = True
ROBUST_REFINE = True      # minimise the average cost over all design members (recommended)
REFINE_ITERS = 600

# ---- anti-oscillation settings
# Every member's closed-loop poles are forced inside |lambda| < STABILITY_RADIUS.
# 0.98 -> slowest mode decays with time constant ~1 s (1.0 = no constraint,
# previous design had 0.998, i.e. ~10 s, lightly damped).
STABILITY_RADIUS = 0.98
# Penalty on the input RATE (u_t - u_{t-1})^T R_DELTA (...) of the total input.
# Smooths the commands and lowers the gains on noisy states. 0 = off.
R_DELTA = 1.0 * R_DESIGN
# Extra penalty on the residual itself (keeps the correction small). 0 = off.
R_RESIDUAL = 0.0 * R_DESIGN
# False: the residual uses only the measured error history, not the ESP32
# replica of the applied u. Try False if the Crazyflie LQR / timing may
# differ from the Python model.
U_HISTORY_FEEDBACK = True
GAIN_SCALE = 1.0          # 0..1, like your residual_alpha. Lower it for first flights.


# ============================================================
# Dynamics loading (same behaviour as your script, condensed)
# ============================================================

class FloatyMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, dropout_rate):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_units:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_member(model_dir, device):
    with open(os.path.join(model_dir, "model_log.json")) as f:
        log = json.load(f)
    p = {k: os.path.join(model_dir, os.path.basename(v)) for k, v in log["paths"].items()
         if k in ("best_model_path", "final_model_path", "sc_in_path", "sc_out_path")}
    path = p["best_model_path"] if (USE_BEST_MODEL and os.path.exists(p["best_model_path"])) \
        else p["final_model_path"]
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["model_config"]
    model = FloatyMLP(cfg["input_dim"], cfg["output_dim"],
                      tuple(cfg["hidden_units"]), float(cfg["dropout_rate"]))
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    return model, joblib.load(p["sc_in_path"]), joblib.load(p["sc_out_path"]), log


class HistoryDynamics:
    """Same equations as NeuralHistoryDynamics12Torch.predict_next_state."""

    def __init__(self, model, sc_in, sc_out, x_bar, u_bar, device):
        t = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), device=device)
        self.model = model
        self.in_mean, self.in_scale = t(sc_in.mean_), t(np.maximum(sc_in.scale_, 1e-6))
        self.out_mean, self.out_scale = t(sc_out.mean_), t(np.maximum(sc_out.scale_, 1e-6))
        self.x_bar = t(x_bar)[None, :]
        self.u_bar = t(u_bar)[None, :]
        self.seq_len = SEQ_LEN

    @torch.no_grad()
    def predict_next_state(self, x_t, x_hist, u_hist):
        B = x_t.shape[0]
        inp = torch.cat([x_hist.reshape(B, -1), u_hist.reshape(B, -1)], dim=1)
        y = self.model((inp - self.in_mean) / self.in_scale) * self.out_scale + self.out_mean
        x_next = x_t + y
        for pi, vi in (([0, 1, 2], [3, 4, 5]), ([6, 7, 8], [9, 10, 11])):
            x_next[:, pi] = x_t[:, pi] + 0.5 * (y[:, pi] + DT * x_next[:, vi])
        return x_next


def load_ensemble(ids, x_bar, u_bar, device):
    with open(os.path.join(DYNAMICS_ENSEMBLE_DIR, "experiments_summary.json")) as f:
        summary = json.load(f)
    by_id = {int(m["ensemble_id"]): m for m in summary["models"] if "ensemble_id" in m}
    out = {}
    for eid in ids:
        mdir = os.path.join(DYNAMICS_ENSEMBLE_DIR, by_id[eid]["model_name"])
        model, sc_in, sc_out, log = load_member(mdir, device)
        out[eid] = HistoryDynamics(model, sc_in, sc_out, x_bar, u_bar, device)
        print(f"Loaded member {eid}: {log.get('model_name')}")
    return out


# ============================================================
# Warm-up U sampling (same as your script)
# ============================================================

DISTS = {"normal": stats.norm, "laplace": stats.laplace, "logistic": stats.logistic,
         "t": stats.t, "beta": stats.beta}


def sample_U_batch(B, u_fit, u_min, u_max, rng):
    cols = []
    for col in u_fit["components"]:
        best = u_fit["fits"][col]["best"]
        cols.append(DISTS[best["distribution"]].rvs(*best["params"], size=B, random_state=rng))
    return np.clip(np.stack(cols, axis=1), u_min, u_max).astype(np.float32)


def sample_U_warmup(B, u_fit, u_min, u_max, rng, first_constant_steps, extra_random_steps):
    # Draw order differs from your per-sample loop, so sequences are
    # statistically identical but not bit-identical to your evaluation.
    first = sample_U_batch(B, u_fit, u_min, u_max, rng)
    seq = [first] * first_constant_steps
    seq += [sample_U_batch(B, u_fit, u_min, u_max, rng) for _ in range(extra_random_steps)]
    return np.stack(seq, axis=1)


# ============================================================
# Existing LQR in torch
# ============================================================

def rot_torch(e):
    e2 = e.clone()
    c, s = torch.cos(e[:, 8]), torch.sin(e[:, 8])
    e2[:, 0] = c * e[:, 0] + s * e[:, 1]
    e2[:, 1] = -s * e[:, 0] + c * e[:, 1]
    return e2


class LqrTorch:
    def __init__(self, lqr, device):
        t = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), device=device)
        self.Kt, self.Ktm1, self.Ku = t(lqr["K_star_t"]), t(lqr["K_star_tm1"]), t(lqr["K_u_diag"])
        self.u_min, self.u_max = t(ed.U_MIN)[None, :], t(ed.U_MAX)[None, :]

    def base(self, u_prev, e_t, e_prev):
        return u_prev * self.Ku[None, :] + rot_torch(e_t) @ self.Kt.T + rot_torch(e_prev) @ self.Ktm1.T

    def clamp(self, u):
        return torch.clamp(u, self.u_min, self.u_max)


# ============================================================
# 1) Identification data
# ============================================================

@torch.no_grad()
def collect_id_data(dyn, lqr_t, u_fit, u_min_ds, u_max_ds, x_abs_thr, seed, device):
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    noise = torch.tensor(PROCESS_NOISE_STD, device=device)[None, :]
    thr = torch.tensor(x_abs_thr, device=device)[None, :]
    B = ID_BATCH

    U_w = torch.tensor(sample_U_warmup(B, u_fit, u_min_ds, u_max_ds, rng,
                                       ID_FIRST_CONSTANT_STEPS, ID_EXTRA_RANDOM_STEPS), device=device)
    x = dyn.x_bar.repeat(B, 1)
    xh = dyn.x_bar[:, None, :].repeat(B, SEQ_LEN, 1)
    uh = dyn.u_bar[:, None, :].repeat(B, SEQ_LEN, 1)
    alive = torch.ones(B, dtype=torch.bool, device=device)
    XH, UH, XN = [], [], []

    def advance(x, xh, uh, u):
        xn_det = dyn.predict_next_state(x, xh, uh)
        keep = alive & torch.isfinite(xn_det).all(dim=1)
        XH.append(xh[keep].cpu().numpy())
        UH.append(uh[keep].cpu().numpy())
        XN.append(xn_det[keep].cpu().numpy())
        xn = xn_det + noise * torch.randn(xn_det.shape, generator=gen, device=device)
        xh = torch.cat([xh[:, 1:], xn[:, None]], dim=1)
        uh = torch.cat([uh[:, 1:], u[:, None]], dim=1)   # u enters history after the step
        return xn, xh, uh

    for k in range(U_w.shape[1]):
        x, xh, uh = advance(x, xh, uh, U_w[:, k])
        alive &= torch.isfinite(x).all(dim=1) & (x.abs() < thr).all(dim=1)

    u = U_w[:, -1]
    e_prev = dyn.x_bar - x
    for t in range(ID_STEPS):
        e_t = dyn.x_bar - x
        u = lqr_t.clamp(lqr_t.base(u, e_t, e_prev)
                        + ID_SIGMA_U * torch.randn((B, N_U), generator=gen, device=device))
        x, xh, uh = advance(x, xh, uh, u)
        alive &= torch.isfinite(x).all(dim=1) & (x.abs() < thr).all(dim=1)
        e_prev = e_t

    print(f"   alive at end: {int(alive.sum())}/{B}")
    return np.concatenate(XH), np.concatenate(UH), np.concatenate(XN)


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(STATS_JSON_PATH) as f:
        limits = json.load(f)
    u_fit = limits["u_sat"]["best_distribution_fit"]
    u_min_ds = np.array(limits["arrays"]["u_sat_min"], dtype=np.float32)
    u_max_ds = np.array(limits["arrays"]["u_sat_max"], dtype=np.float32)
    x_abs_thr = np.array(limits["arrays"]["x_max_abs_value"], dtype=np.float32)

    x_bar = np.zeros(N_X, dtype=np.float32)
    x_bar[2] = 1.12
    u_bar = np.zeros(N_U, dtype=np.float32)

    lqr = ed.build_lqr_gain_matrices()
    lqr_t = LqrTorch(lqr, device)
    dyns = load_ensemble(ALL_IDS, x_bar, u_bar, device)

    # ---- 1) identification per design member
    thetas, cs = {}, {}
    zz_sum, zz_n = np.zeros((ed.NZ, ed.NZ)), 0
    print("\nIdentifying linear history models around hover")
    for eid in ALL_IDS:          # test members fitted only for the stability report
        XH, UH, XN = collect_id_data(dyns[eid], lqr_t, u_fit, u_min_ds, u_max_ds,
                                     x_abs_thr, seed=SEED + 100 * eid, device=device)
        Theta, c, r2 = ed.fit_affine_history_model(XH, UH, XN, x_bar, u_bar, ridge=ID_RIDGE)
        thetas[eid], cs[eid] = Theta, c
        if eid in DESIGN_IDS:
            Z = np.concatenate([(XH - x_bar).reshape(len(XH), -1), (UH - u_bar).reshape(len(UH), -1)], 1)
            zz_sum += Z.T @ Z
            zz_n += len(Z)
        print(f"   member {eid:2d}: N={len(XN)}  R2 min/mean = {r2.min():.3f}/{r2.mean():.3f}"
              f"  |drift c| = {np.linalg.norm(c):.2e}")

    Theta_mean = np.mean([thetas[i] for i in DESIGN_IDS], axis=0)
    c_mean = np.mean([cs[i] for i in DESIGN_IDS], axis=0)

    # ---- 2) design
    design = ed.design_residual(
        Theta_mean, c_mean, Q_DESIGN, R_DESIGN, lqr=lqr,
        use_feedforward=USE_FEEDFORWARD,
        R_delta=R_DELTA, R_res=R_RESIDUAL,
        stability_radius=STABILITY_RADIUS,
        u_history_feedback=U_HISTORY_FEEDBACK,
    )
    if not U_HISTORY_FEEDBACK and not ROBUST_REFINE:
        raise ValueError("U_HISTORY_FEEDBACK = False needs ROBUST_REFINE = True")
    if ROBUST_REFINE:
        print("\nMulti-model refinement over design members")
        W = zz_sum / max(zz_n, 1) + 1e-8 * np.eye(ed.NZ)
        W = W / np.trace(W) * ed.NZ
        design = ed.refine_multimodel(design, [thetas[i] for i in DESIGN_IDS], W=W, iters=REFINE_ITERS)

    print("\nu_eq (hover feed-forward):", np.round(design["u_eq"], 4))
    print("|F| max:", np.abs(design["F"]).max())

    # ---- 3) robustness across members (linearised)
    print("\nClosed loop (linearised, no saturation): spectral radius | min damping ratio")
    stab = []
    for eid in ALL_IDS:
        rl, rn = ed.closed_loop_spectral_radius(thetas[eid], design, GAIN_SCALE)
        zl, zn = ed.closed_loop_damping(thetas[eid], design, GAIN_SCALE)
        stab.append({"ensemble_id": eid, "rho_lqr": rl, "rho_lqr_elqr": rn,
                     "zeta_lqr": zl, "zeta_lqr_elqr": zn})
        flag = "  <-- UNSTABLE" if rn >= 1.0 else ""
        tag = " (held out)" if eid in TEST_IDS else ""
        print(f"   member {eid:2d}{tag:11s}: LQR {rl:.4f} | {zl:.3f}    "
              f"LQR+ELQR {rn:.4f} | {zn:.3f}{flag}")
    pd.DataFrame(stab).to_csv(os.path.join(OUT_DIR, "stability.csv"), index=False)

    # ---- 4) export
    hdr = ed.export_header(
        design, os.path.join(OUT_DIR, "elqr_gains.h"), gain_scale=GAIN_SCALE,
        comment=(f"Ensemble {DYNAMICS_ENSEMBLE_DIR}, design members {DESIGN_IDS}\n"
                 f"feed-forward={USE_FEEDFORWARD}, multi-model refine={ROBUST_REFINE}\n"
                 f"stability radius={STABILITY_RADIUS}, R_delta scale={np.trace(R_DELTA) / np.trace(R_DESIGN):.3g}, "
                 f"R_res scale={np.trace(R_RESIDUAL) / np.trace(R_DESIGN):.3g}, u-history feedback={U_HISTORY_FEEDBACK}"),
    )
    ed.save_design_json(design, os.path.join(OUT_DIR, "elqr_design.json"),
                        extra={"design_ids": DESIGN_IDS, "gain_scale": GAIN_SCALE,
                               "u_history_feedback": U_HISTORY_FEEDBACK,
                               "R_delta": R_DELTA.tolist(), "R_res": R_RESIDUAL.tolist()})
    print("\nExported:", hdr)
    print("Next: run compare_controllers.py for the LQR / LQR+NN / LQR+ELQR comparison.")


if __name__ == "__main__":
    main()
