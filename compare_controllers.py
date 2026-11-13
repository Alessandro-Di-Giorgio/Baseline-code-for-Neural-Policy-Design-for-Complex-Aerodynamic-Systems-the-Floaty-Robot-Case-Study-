"""
Fair comparison: LQR  vs  LQR+NN  vs  LQR+ELQR

This is your post-training rollout evaluation (from
Ensemble_policy_pick_one_white_noise_infoprop_roll_20_20_horizon_750.py)
with a third controller added. Everything that determines the numbers is
copied from your script without changes:

  - dynamics wrapper NeuralHistoryDynamics12Torch and ensemble loading
  - warm-up U sampling (same functions -> same RNG draw order)
  - ROLLOUT_WARMUP_SEED = 12, deterministic warm-up, B = 16, 30 s
  - white process noise, generator re-seeded with ROLLOUT_NOISE_SEED = 10
    for EVERY controller -> identical noise realisation for all three
  - LQR formula, yaw rotation, clamp to the control limits
  - cost_computation with Q_ROLLOUT / Q_T_ROLLOUT / R_ROLLOUT
  - one "Rollout costs.csv" with per-component costs for all 13 members
  - plots + rollout_info.json for held-out models 11 and 12

With SIMULATE_UINT8_LINK = False, the LQR and LQR+NN rows must equal the
ones in your existing "Rollout costs.csv" (good sanity check).

Run elqr_main.py first (it writes elqr_design.json).
"""

import os
import json
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats

import torch
import torch.nn as nn

import elqr_design as ed


# ============================================================
# Configuration
# ============================================================

OUTER_FOLDER = "Floaty_cluster_repo/Cluster_experiments_Floaty"

DYNAMICS_ENSEMBLE_DIR = (
    "Floaty_cluster_repo/Cluster_experiments_Floaty/"
    "Dynamical_models/Ensemble_29_6_26"
)

STATS_JSON_PATH = os.path.join(
    OUTER_FOLDER, "Training_flight_data", "floaty_dataset_limits_cluster.json",
)

# Your trained NN run
NN_RUN_DIR = os.path.join(
    OUTER_FOLDER, "Trained_policies",
    "Floaty_policy_lqr_residual_ensemble_dyn_29_6_26_wh_IP_horizons",
    "horizon_750_IP_wh_h20x20",
)
NN_POLICY_PATH = os.path.join(NN_RUN_DIR, "policy_best.pt")
NN_POLICY_NAME = os.path.basename(NN_RUN_DIR)

# Output of elqr_main.py
ELQR_DIR = os.path.join(OUTER_FOLDER, "Trained_policies", "Ensemble_LQR_residual")
ELQR_DESIGN_JSON = os.path.join(ELQR_DIR, "elqr_design.json")


USE_BEST_MODEL = True
MODEL_SEED = 5

# False: residuals are used as floats (= your original evaluation of the NN).
# True : both residuals (NN and ELQR) pass through the ESP32 uint8 link
#        (saturation to [OUT_MIN, OUT_MAX] + 8-bit quantisation), as on hardware.
SIMULATE_UINT8_LINK = False

# Steps of delay between the residual computed by the ESP32 and the step the
# Crazyflie applies it (0 = as in training). 1 is a realistic hardware check:
# if a controller degrades a lot with 1, it is sensitive to link timing.
# Applied identically to NN and ELQR.
LINK_DELAY_STEPS = 0

# One folder per setting, so runs with/without delay or uint8 link don't overwrite each other
OUT_DIR = os.path.join(
    ELQR_DIR,
    f"Comparison_vs_{NN_POLICY_NAME}"
    + (f"_delay{LINK_DELAY_STEPS}" if LINK_DELAY_STEPS > 0 else "")
    + ("_uint8" if SIMULATE_UINT8_LINK else ""),
)

n_x = 12
n_u = 4
DT = 0.02
SEQ_LEN = 5

CONTROLLERS = ["LQR", "LQR+NN", "LQR+ELQR"]
CONTROLLER_FILE_NAMES = {"LQR": "LQR", "LQR+NN": "LQR_plus_NN", "LQR+ELQR": "LQR_plus_ELQR"}


# ============================================================
# Dynamics (copied from your script)
# ============================================================

def to_torch(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def scale_torch(x, mean, scale):
    return (x - mean) / scale


def inv_scale_torch(x_scaled, mean, scale):
    return x_scaled * scale + mean


class FloatyMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, dropout_rate):
        super().__init__()
        layers_list = []
        prev_dim = input_dim
        for units in hidden_units:
            layers_list.append(nn.Linear(prev_dim, units))
            layers_list.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers_list.append(nn.Dropout(dropout_rate))
            prev_dim = units
        layers_list.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.net(x)


def load_model_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["model_config"]
    model = FloatyMLP(
        input_dim=cfg["input_dim"],
        output_dim=cfg["output_dim"],
        hidden_units=tuple(cfg["hidden_units"]),
        dropout_rate=float(cfg["dropout_rate"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint


def load_selected_pytorch_model(model_dir, use_best_model=True, device=None):
    with open(os.path.join(model_dir, "model_log.json"), "r") as f:
        model_log = json.load(f)

    def local(key):
        return os.path.join(model_dir, os.path.basename(model_log["paths"][key]))

    best_model_path = local("best_model_path")
    final_model_path = local("final_model_path")
    sc_in_path = local("sc_in_path")
    sc_out_path = local("sc_out_path")

    if use_best_model and os.path.exists(best_model_path):
        model_path = best_model_path
    elif os.path.exists(final_model_path):
        model_path = final_model_path
    else:
        raise FileNotFoundError(f"No model in {model_dir}")

    model, _ = load_model_checkpoint(model_path, device=device)
    model.eval()
    return model, joblib.load(sc_in_path), joblib.load(sc_out_path), model_log, model_path


class NeuralHistoryDynamics12Torch(nn.Module):
    def __init__(self, dyn_model, sc_in, sc_out, x_bar, u_bar,
                 seq_len=5, n_x=12, n_u=4, dt=0.02, device=torch.device("cpu")):
        super().__init__()
        self.model = dyn_model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.seq_len = int(seq_len)
        self.n_x = int(n_x)
        self.n_u = int(n_u)
        self.device = device
        self.dt = float(dt)

        self.register_buffer("in_mean", to_torch(sc_in.mean_.astype(np.float32), device))
        self.register_buffer("in_scale", to_torch(np.maximum(sc_in.scale_, 1e-6).astype(np.float32), device))
        self.register_buffer("out_mean", to_torch(sc_out.mean_.astype(np.float32), device))
        self.register_buffer("out_scale", to_torch(np.maximum(sc_out.scale_, 1e-6).astype(np.float32), device))
        self.register_buffer("x_bar", to_torch(np.asarray(x_bar, dtype=np.float32)[None, :], device))
        self.register_buffer("u_bar", to_torch(np.asarray(u_bar, dtype=np.float32)[None, :], device))

    def predict_next_state(self, x_t, u_t, x_hist, u_hist):
        x_t = x_t.float()
        x_hist = x_hist.float()
        u_hist = u_hist.float()
        B = x_t.shape[0]

        seq_x = x_hist.reshape(B, self.seq_len * self.n_x)
        seq_u = u_hist.reshape(B, self.seq_len * self.n_u)
        inp = scale_torch(torch.cat([seq_x, seq_u], dim=1), self.in_mean, self.in_scale)
        y = inv_scale_torch(self.model(inp), self.out_mean, self.out_scale)

        x_next = x_t + y
        pos_idx, lin_vel_idx = [0, 1, 2], [3, 4, 5]
        angle_idx, ang_rate_idx = [6, 7, 8], [9, 10, 11]

        x_next[:, pos_idx] = x_t[:, pos_idx] + 0.5 * (y[:, pos_idx] + self.dt * x_next[:, lin_vel_idx])
        x_next[:, angle_idx] = x_t[:, angle_idx] + 0.5 * (y[:, angle_idx] + self.dt * x_next[:, ang_rate_idx])
        return x_next

    def step(self, x_t, u_t, x_hist, u_hist, model_seed=None):
        return self.predict_next_state(x_t=x_t, u_t=u_t, x_hist=x_hist, u_hist=u_hist)


def load_ensemble_dynamics_members(ensemble_dir, ensemble_ids, use_best_model, device,
                                   x_bar, u_bar, seq_len=5, n_x=12, n_u=4, dt=0.02):
    with open(os.path.join(ensemble_dir, "experiments_summary.json"), "r") as f:
        summary = json.load(f)
    by_id = {int(m["ensemble_id"]): m for m in summary.get("models", []) if "ensemble_id" in m}

    entries = []
    for eid in ensemble_ids:
        info = by_id[int(eid)]
        model_dir = os.path.join(ensemble_dir, info["model_name"])
        dyn_model, sc_in, sc_out, model_log, dyn_model_path = load_selected_pytorch_model(
            model_dir=model_dir, use_best_model=use_best_model, device=device,
        )
        dyn = NeuralHistoryDynamics12Torch(
            dyn_model=dyn_model, sc_in=sc_in, sc_out=sc_out, x_bar=x_bar, u_bar=u_bar,
            seq_len=seq_len, n_x=n_x, n_u=n_u, dt=dt, device=device,
        )
        entries.append({
            "ensemble_id": int(eid),
            "model_name": model_log.get("model_name", info.get("model_name")),
            "dyn": dyn,
            "dyn_model_path": dyn_model_path,
        })
        print(f"Loaded ensemble dynamics {eid}: {entries[-1]['model_name']}")
    return entries


# ============================================================
# Warm-up U sampling (copied: same draw order as your script)
# ============================================================

distribution_map = {
    "normal": stats.norm,
    "laplace": stats.laplace,
    "logistic": stats.logistic,
    "t": stats.t,
    "beta": stats.beta,
}


def sample_U_from_loaded_fit(loaded_u_fit, u_min, u_max, rng=None, clip=True):
    U_sample = []
    for col in loaded_u_fit["components"]:
        best = loaded_u_fit["fits"][col]["best"]
        value = distribution_map[best["distribution"]].rvs(*best["params"], random_state=rng)
        U_sample.append(float(value))
    U_sample = np.array(U_sample, dtype=np.float32)
    if clip:
        U_sample = np.clip(U_sample, u_min, u_max)
    return U_sample


def sample_U_batch_from_loaded_fit(B, loaded_u_fit, u_min, u_max, rng=None, clip=True):
    return np.stack([
        sample_U_from_loaded_fit(loaded_u_fit, u_min, u_max, rng=rng, clip=clip)
        for _ in range(B)
    ], axis=0).astype(np.float32)


def sample_U_warmup_sequence_from_loaded_fit(B, loaded_u_fit, u_min, u_max, rng=None, clip=True,
                                             first_constant_steps=5, extra_random_steps=5, scale=1.0):
    U_first = scale * sample_U_batch_from_loaded_fit(B, loaded_u_fit, u_min, u_max, rng, clip)
    if clip:
        U_first = np.clip(U_first, u_min, u_max)
    U_first_seq = np.tile(U_first[:, None, :], (1, first_constant_steps, 1))

    U_extra_list = []
    for _ in range(extra_random_steps):
        U_k = scale * sample_U_batch_from_loaded_fit(B, loaded_u_fit, u_min, u_max, rng, clip)
        if clip:
            U_k = np.clip(U_k, u_min, u_max)
        U_extra_list.append(U_k)

    if extra_random_steps > 0:
        U_seq = np.concatenate([U_first_seq, np.stack(U_extra_list, axis=1)], axis=1)
    else:
        U_seq = U_first_seq
    return U_seq.astype(np.float32)


# ============================================================
# LQR (copied)
# ============================================================

def build_lqr_gain_matrices(n_x=12, n_u=4):
    state_map = np.array([
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    ], dtype=np.float32)
    input_map = np.eye(n_u, dtype=np.float32)
    K_bar_t = np.array([0.035, 0.752, -0.291, 0.052, 0.791, -0.597,
                        -0.514, 0.075, 0.308, -0.151, 0.007, 0.144], dtype=np.float32)
    K_bar_tm1 = np.array([0.008, 0.362, -0.135, -0.002, -0.084, 0.229,
                          -0.059, 0.019, -0.167, 0.090, -0.005, -0.088], dtype=np.float32)
    K_u_bar = np.array([-0.122, -0.136, 0.179, -0.014], dtype=np.float32)
    return {
        "K_star_t": state_map * K_bar_t[None, :],
        "K_star_tm1": state_map * K_bar_tm1[None, :],
        "K_u_star": input_map * K_u_bar[None, :],
    }


def rotate_xy_error_by_yaw_torch(e):
    e_rot = e.clone()
    yaw = e[:, 8]
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    ex = e[:, 0]
    ey = e[:, 1]
    e_rot[:, 0] = c * ex + s * ey
    e_rot[:, 1] = -s * ex + c * ey
    return e_rot


# ============================================================
# NN policy (copied PolicyDU24to4 + loader)
# ============================================================

def make_activation(activation):
    activation = str(activation).lower()
    if activation == "tanh":
        return nn.Tanh()
    if activation == "relu":
        return nn.ReLU()
    if activation == "silu":
        return nn.SiLU()
    raise ValueError(activation)


class ResidualPolicyBlock(nn.Module):
    def __init__(self, dim, activation="tanh"):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.act = make_activation(activation)

    def forward(self, x):
        return x + self.act(self.fc(x))


class PolicyDU24to4(nn.Module):
    def __init__(self, hidden=(32, 32, 32), seed=None, activation="tanh", use_skip=False):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        layers, in_dim = [], 24
        for idx, h in enumerate(hidden):
            if use_skip and idx > 0 and h == in_dim:
                layers.append(ResidualPolicyBlock(h, activation=activation))
            else:
                layers.append(nn.Linear(in_dim, h))
                layers.append(make_activation(activation))
                in_dim = h
        layers.append(nn.Linear(in_dim, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_policy_du32to4_torch(policy_path, device):
    checkpoint = torch.load(policy_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("policy_config", {})
    mu = PolicyDU24to4(
        hidden=tuple(cfg.get("hidden_layers", (32, 32, 32))),
        seed=MODEL_SEED,
        activation=cfg.get("activation", "tanh"),
        use_skip=bool(cfg.get("use_skip", False)),
    ).to(device)
    mu.load_state_dict(checkpoint["model_state_dict"])
    mu.eval()
    print(f"Loaded NN policy: {policy_path} (best epoch {checkpoint.get('best_epoch')})")
    return mu, checkpoint


# ============================================================
# ELQR policy: same call signature as the NN (input [e_next, e_t])
# ============================================================

class ElqrTorchPolicy(nn.Module):
    """Stateful. reset() before every rollout (as the ESP32 starts from scratch)."""

    def __init__(self, design, gain_scale, quantize):
        super().__init__()
        self.ctrl = ed.ElqrResidualController(design, gain_scale=gain_scale, quantize=quantize)

    def reset(self):
        self.ctrl.reset()

    def forward(self, mu_in):
        e_t = mu_in[:, :n_x].detach().cpu().numpy().astype(np.float64)
        e_prev = mu_in[:, n_x:].detach().cpu().numpy().astype(np.float64)
        r = self.ctrl.step(e_t, e_prev)
        return torch.tensor(r, dtype=mu_in.dtype, device=mu_in.device)


def uint8_link_torch(r):
    """ESP32 link applied to a float residual: saturate, quantise, decode."""
    r_np = r.detach().cpu().numpy().astype(np.float64)
    r_q = ed.dequantize_residual(ed.quantize_residual(r_np))
    return torch.tensor(r_q, dtype=r.dtype, device=r.device)


# ============================================================
# Cost (copied)
# ============================================================

def quadratic_form_np(x, Q):
    return np.einsum("...i,ij,...j->...", x, Q, x)


def cost_computation(x_closed_loop_traj, u_closed_loop_traj, x_ref, Q, Q_T, R, T_rollout):
    e_full = x_ref[None, :, :] - x_closed_loop_traj

    state_stage_cost = quadratic_form_np(e_full, Q)
    input_stage_cost = quadratic_form_np(u_closed_loop_traj, R)
    stage_cost = state_stage_cost + input_stage_cost

    terminal_error = e_full[:, -1, :]
    terminal_cost = quadratic_form_np(terminal_error, Q_T)

    total_state_stage_cost = np.sum(state_stage_cost, axis=1)
    total_input_stage_cost = np.sum(input_stage_cost, axis=1)
    total_stage_cost = np.sum(stage_cost, axis=1)
    total_rollout_cost = total_stage_cost + terminal_cost
    normalized_rollout_cost = total_rollout_cost / float(T_rollout + 1)

    state_component_cost_t = e_full ** 2 * np.diag(Q)[None, None, :]
    input_component_cost_t = u_closed_loop_traj ** 2 * np.diag(R)[None, None, :]
    terminal_component_cost = terminal_error ** 2 * np.diag(Q_T)[None, :]

    state_component_stage_cost = np.sum(state_component_cost_t, axis=1)
    input_component_stage_cost = np.sum(input_component_cost_t, axis=1)
    state_component_total_cost = state_component_stage_cost + terminal_component_cost

    reconstructed = np.sum(state_component_total_cost, axis=1) + np.sum(input_component_stage_cost, axis=1)
    if np.all(np.isfinite(total_rollout_cost)) and not np.allclose(
            reconstructed, total_rollout_cost, rtol=1e-5, atol=1e-7):
        raise RuntimeError("Component-wise cost does not reconstruct total rollout cost.")

    return {
        "total_rollout_cost": total_rollout_cost,
        "normalized_rollout_cost": normalized_rollout_cost,
        "total_state_stage_cost": total_state_stage_cost,
        "total_input_stage_cost": total_input_stage_cost,
        "terminal_cost": terminal_cost,
        "state_component_stage_cost": state_component_stage_cost,
        "terminal_component_cost": terminal_component_cost,
        "state_component_total_cost": state_component_total_cost,
        "state_component_normalized_cost": state_component_total_cost / float(T_rollout + 1),
        "input_component_stage_cost": input_component_stage_cost,
        "input_component_normalized_cost": input_component_stage_cost / float(T_rollout + 1),
    }


# ============================================================
# Rollout settings (copied)
# ============================================================

ROLLOUT_TIME_S = 30.0
PLOT_TIME_S = 30.0
T_ROLLOUT = int(round(ROLLOUT_TIME_S / DT))
T_PLOT = min(int(round(PLOT_TIME_S / DT)), T_ROLLOUT)
B_ROLLOUT = 16
ROLLOUT_WARMUP_SEED = 12
ROLLOUT_NOISE_SEED = 10
ROLLOUT_FIRST_CONSTANT_STEPS = 5
ROLLOUT_EXTRA_RANDOM_STEPS = 45

ROLLOUT_NOISE_STD = np.array([
    0.001, 0.001, 0.001,
    0.005, 0.005, 0.005,
    0.002, 0.002, 0.002,
    0.050, 0.050, 0.050,
], dtype=np.float32)

Q_ROLLOUT = np.diag([50, 80, 10, 20, 50, 10, 0.02, 0.1, 0.7, 0.02, 0.1, 0.1]).astype(np.float32)
Q_T_ROLLOUT = Q_ROLLOUT.copy()
R_ROLLOUT = np.diag([50, 10, 5, 5]).astype(np.float32)

STATE_NAMES = ["x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw", "p", "q", "r"]
INPUT_NAMES = ["u1", "u2", "u3", "u4"]

TEST_PLOT_MODEL_IDS = {11, 12}

OUT_MIN_02, OUT_MAX_02 = -0.692699, 0.407301
OUT_MIN_13, OUT_MAX_13 = -0.407301, 0.692699
u_min_control = np.array([OUT_MIN_02, OUT_MIN_13, OUT_MIN_02, OUT_MIN_13], dtype=np.float32)
u_max_control = np.array([OUT_MAX_02, OUT_MAX_13, OUT_MAX_02, OUT_MAX_13], dtype=np.float32)


# ============================================================
# Plot helper
# ============================================================

def plot_batch(path, t, Y, title, ylabel, label, ref=None):
    plt.figure(figsize=(8, 4))
    for b in range(Y.shape[0]):
        plt.plot(t, Y[b], alpha=0.75, label=label if b == 0 else None)
    if ref is not None:
        plt.plot(t, ref, "k--", linewidth=2, label="reference")
    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    with open(STATS_JSON_PATH, "r") as f:
        loaded_limits = json.load(f)
    loaded_u_fit = loaded_limits["u_sat"]["best_distribution_fit"]
    # Your evaluation used the DATASET limits for the warm-up (u_min_loaded)
    u_min_loaded = np.array(loaded_limits["arrays"]["u_sat_min"], dtype=np.float32)
    u_max_loaded = np.array(loaded_limits["arrays"]["u_sat_max"], dtype=np.float32)

    x_bar = np.zeros(n_x, dtype=np.float32)
    x_bar[2] = 1.12
    u_bar = np.zeros(n_u, dtype=np.float32)

    x_ref_rollout = np.tile(x_bar[None, :], (T_ROLLOUT + 1, 1)).astype(np.float32)
    x_ref_rollout_torch = to_torch(x_ref_rollout, device)

    lqr = build_lqr_gain_matrices(n_x, n_u)
    K_star_t = to_torch(lqr["K_star_t"], device)
    K_star_tm1 = to_torch(lqr["K_star_tm1"], device)
    K_u_star = to_torch(lqr["K_u_star"], device)
    u_min_t = to_torch(u_min_control[None, :], device)
    u_max_t = to_torch(u_max_control[None, :], device)
    noise_std_t = to_torch(ROLLOUT_NOISE_STD[None, :], device)

    # ---- controllers
    mu_nn, nn_ckpt = load_policy_du32to4_torch(NN_POLICY_PATH, device)

    design = ed.load_design_json(ELQR_DESIGN_JSON)
    gain_scale = float(design.get("gain_scale", 1.0))
    elqr_policy = ElqrTorchPolicy(design, gain_scale=gain_scale, quantize=SIMULATE_UINT8_LINK)
    print(f"Loaded ELQR design: {ELQR_DESIGN_JSON} (gain_scale {gain_scale})")

    # ---- dynamics: all members 0..12
    rollout_dyn_entries = load_ensemble_dynamics_members(
        DYNAMICS_ENSEMBLE_DIR, list(range(13)), USE_BEST_MODEL, device,
        x_bar, u_bar, SEQ_LEN, n_x, n_u, DT,
    )

    # ---- common warm-up input sequence (same seed and sampler as yours)
    U_warmup_rollout_np = sample_U_warmup_sequence_from_loaded_fit(
        B=B_ROLLOUT, loaded_u_fit=loaded_u_fit, u_min=u_min_loaded, u_max=u_max_loaded,
        rng=np.random.default_rng(ROLLOUT_WARMUP_SEED), clip=True,
        first_constant_steps=ROLLOUT_FIRST_CONSTANT_STEPS,
        extra_random_steps=ROLLOUT_EXTRA_RANDOM_STEPS, scale=1.0,
    )
    U_warmup_rollout_torch = to_torch(U_warmup_rollout_np, device)
    total_warmup_steps = ROLLOUT_FIRST_CONSTANT_STEPS + ROLLOUT_EXTRA_RANDOM_STEPS

    time_full = np.arange(T_ROLLOUT + 1) * DT
    plot_slice = slice(0, T_PLOT + 1)
    time_plot = time_full[plot_slice]

    rollout_cost_records = []

    for dyn_entry in rollout_dyn_entries:
        ensemble_id = int(dyn_entry["ensemble_id"])
        model_name = dyn_entry["model_name"]
        dyn = dyn_entry["dyn"]
        dyn.model.eval()

        print("\n" + "=" * 90)
        print(f"ROLLOUT - MODEL {ensemble_id} ({model_name})")
        print("=" * 90)

        # ---- deterministic warm-up (no process noise), shared by all controllers
        x_t = dyn.x_bar.repeat(B_ROLLOUT, 1)
        x_hist = dyn.x_bar[:, None, :].repeat(B_ROLLOUT, dyn.seq_len, 1)
        u_hist = dyn.u_bar[:, None, :].repeat(B_ROLLOUT, dyn.seq_len, 1)

        with torch.no_grad():
            for k in range(total_warmup_steps):
                u_k = U_warmup_rollout_torch[:, k, :]
                x_next = dyn.step(x_t=x_t, u_t=u_k, x_hist=x_hist, u_hist=u_hist)
                x_hist = torch.cat([x_hist[:, 1:, :], x_next[:, None, :]], dim=1)
                u_hist = torch.cat([u_hist[:, 1:, :], u_k[:, None, :]], dim=1)
                x_t = x_next

        x0 = x_t.clone()
        u0 = U_warmup_rollout_torch[:, -1, :].clone()
        x_hist0 = x_hist.clone()
        u_hist0 = u_hist.clone()

        for controller_name in CONTROLLERS:
            print(f"Model {ensemble_id} | {controller_name}")

            if controller_name == "LQR+ELQR":
                elqr_policy.reset()

            x_cl, u_cl = x0.clone(), u0.clone()
            x_hist_cl, u_hist_cl = x_hist0.clone(), u_hist0.clone()

            # Same noise realisation for every controller
            noise_gen = torch.Generator(device=device)
            noise_gen.manual_seed(int(ROLLOUT_NOISE_SEED))

            e_t = x_ref_rollout_torch[0:1, :].repeat(B_ROLLOUT, 1) - x_cl

            res_queue = [torch.zeros_like(u_cl) for _ in range(LINK_DELAY_STEPS)]

            x_traj = [x_cl.cpu().numpy()]
            u_traj = [u_cl.cpu().numpy()]
            u_base_traj = [np.zeros((B_ROLLOUT, n_u), dtype=np.float32)]
            u_res_traj = [np.zeros((B_ROLLOUT, n_u), dtype=np.float32)]

            with torch.no_grad():
                for t in range(T_ROLLOUT):
                    x_next = dyn.step(x_t=x_cl, u_t=u_cl, x_hist=x_hist_cl, u_hist=u_hist_cl)

                    if np.any(ROLLOUT_NOISE_STD > 0.0):
                        eps = torch.randn(x_next.shape, generator=noise_gen,
                                          device=device, dtype=x_next.dtype)
                        x_next = x_next + noise_std_t * eps

                    x_hist_cl = torch.cat([x_hist_cl[:, 1:, :], x_next[:, None, :]], dim=1)
                    u_hist_cl = torch.cat([u_hist_cl[:, 1:, :], u_cl[:, None, :]], dim=1)

                    e_next = x_ref_rollout_torch[t + 1:t + 2, :].repeat(B_ROLLOUT, 1) - x_next

                    u_base_next = (
                        torch.matmul(u_cl, K_u_star.T)
                        + torch.matmul(rotate_xy_error_by_yaw_torch(e_next), K_star_t.T)
                        + torch.matmul(rotate_xy_error_by_yaw_torch(e_t), K_star_tm1.T)
                    )

                    mu_in_next = torch.cat([e_next, e_t], dim=1)

                    if controller_name == "LQR":
                        u_res_next = torch.zeros_like(u_base_next)
                    elif controller_name == "LQR+NN":
                        u_res_next = mu_nn(mu_in_next)
                        if SIMULATE_UINT8_LINK:
                            u_res_next = uint8_link_torch(u_res_next)
                    else:  # LQR+ELQR (link handled inside, the u replica needs it)
                        u_res_next = elqr_policy(mu_in_next)

                    if LINK_DELAY_STEPS > 0:
                        res_queue.append(u_res_next)
                        u_res_next = res_queue.pop(0)

                    u_next = torch.clamp(u_base_next + u_res_next, min=u_min_t, max=u_max_t)

                    x_traj.append(x_next.cpu().numpy())
                    u_traj.append(u_next.cpu().numpy())
                    u_base_traj.append(u_base_next.cpu().numpy())
                    u_res_traj.append(u_res_next.cpu().numpy())

                    x_cl, u_cl = x_next, u_next
                    e_t = e_next

            x_traj = np.stack(x_traj, axis=1)
            u_traj = np.stack(u_traj, axis=1)
            u_base_traj = np.stack(u_base_traj, axis=1)
            u_res_traj = np.stack(u_res_traj, axis=1)

            cr = cost_computation(x_traj, u_traj, x_ref_rollout,
                                  Q_ROLLOUT, Q_T_ROLLOUT, R_ROLLOUT, T_ROLLOUT)

            rec = {
                "ensemble_id": ensemble_id,
                "split": ("train" if ensemble_id <= 7 else "val" if ensemble_id <= 10 else "test"),
                "model_name": model_name,
                "controller": controller_name,
                "mean_total_rollout_cost": float(np.mean(cr["total_rollout_cost"])),
                "mean_normalized_rollout_cost": float(np.mean(cr["normalized_rollout_cost"])),
                "mean_state_stage_cost": float(np.mean(cr["total_state_stage_cost"])),
                "mean_input_stage_cost": float(np.mean(cr["total_input_stage_cost"])),
                "mean_terminal_cost": float(np.mean(cr["terminal_cost"])),
                "n_non_finite_rollouts": int(np.sum(~np.isfinite(cr["total_rollout_cost"]))),
                # oscillation / smoothness metrics (lower = smoother)
                "mean_abs_du": float(np.mean(np.abs(np.diff(u_traj, axis=1)))),
                "rms_attitude_rad": float(np.sqrt(np.mean(x_traj[:, :, 6:9] ** 2))),
                "rms_body_rates": float(np.sqrt(np.mean(x_traj[:, :, 9:12] ** 2))),
                "rms_pos_err_last_10s": float(np.sqrt(np.mean(
                    (x_traj[:, -int(10.0 / DT):, :3] - x_bar[None, None, :3]) ** 2))),
            }
            for j, name in enumerate(STATE_NAMES):
                rec[f"state_{name}_stage"] = float(np.mean(cr["state_component_stage_cost"][:, j]))
                rec[f"state_{name}_terminal"] = float(np.mean(cr["terminal_component_cost"][:, j]))
                rec[f"state_{name}_total"] = float(np.mean(cr["state_component_total_cost"][:, j]))
                rec[f"state_{name}_normalized"] = float(np.mean(cr["state_component_normalized_cost"][:, j]))
            for j, name in enumerate(INPUT_NAMES):
                rec[f"input_{name}_total"] = float(np.mean(cr["input_component_stage_cost"][:, j]))
                rec[f"input_{name}_normalized"] = float(np.mean(cr["input_component_normalized_cost"][:, j]))
            rollout_cost_records.append(rec)
            print(f"   normalized cost = {rec['mean_normalized_rollout_cost']:.6e}")

            # ---- plots for held-out models only
            if ensemble_id in TEST_PLOT_MODEL_IDS:
                plot_dir = os.path.join(OUT_DIR, f"Rollout_plots_model_{ensemble_id}")
                os.makedirs(plot_dir, exist_ok=True)
                fname = CONTROLLER_FILE_NAMES[controller_name]

                for j, name in enumerate(STATE_NAMES):
                    plot_batch(os.path.join(plot_dir, f"state_{name}_{fname}.png"),
                               time_plot, x_traj[:, plot_slice, j],
                               f"Model {ensemble_id} | {controller_name} | state {name}",
                               name, "trajectory", ref=x_ref_rollout[plot_slice, j])

                for j, name in enumerate(INPUT_NAMES):
                    plot_batch(os.path.join(plot_dir, f"input_{name}_{fname}.png"),
                               time_plot, u_traj[:, plot_slice, j],
                               f"Model {ensemble_id} | {controller_name} | total applied {name}",
                               name, "u total")

                if controller_name != "LQR":
                    for j, name in enumerate(INPUT_NAMES):
                        plot_batch(os.path.join(plot_dir, f"input_{name}_{fname}_base.png"),
                                   time_plot, u_base_traj[:, plot_slice, j],
                                   f"Model {ensemble_id} | {controller_name} | baseline {name}",
                                   name, "u_base")
                        plot_batch(os.path.join(plot_dir, f"input_{name}_{fname}_residual.png"),
                                   time_plot, u_res_traj[:, plot_slice, j],
                                   f"Model {ensemble_id} | {controller_name} | residual {name}",
                                   name, "u_residual")

        if ensemble_id in TEST_PLOT_MODEL_IDS:
            info = {
                "ensemble_id": ensemble_id,
                "dynamics_model_name": model_name,
                "dynamics_model_path": dyn_entry["dyn_model_path"],
                "nn_policy_path": NN_POLICY_PATH,
                "nn_best_epoch": nn_ckpt.get("best_epoch"),
                "elqr_design_json": ELQR_DESIGN_JSON,
                "elqr_gain_scale": gain_scale,
                "simulate_uint8_link": SIMULATE_UINT8_LINK,
                "link_delay_steps": LINK_DELAY_STEPS,
                "rollout_time_s": ROLLOUT_TIME_S,
                "dt": DT,
                "T_rollout": T_ROLLOUT,
                "batch_size": B_ROLLOUT,
                "warmup_seed": ROLLOUT_WARMUP_SEED,
                "noise_seed": ROLLOUT_NOISE_SEED,
                "first_constant_steps": ROLLOUT_FIRST_CONSTANT_STEPS,
                "extra_random_steps": ROLLOUT_EXTRA_RANDOM_STEPS,
                "cost_matrices": {"Q": Q_ROLLOUT.tolist(), "Q_T": Q_T_ROLLOUT.tolist(),
                                  "R": R_ROLLOUT.tolist()},
                "white_noise_std": ROLLOUT_NOISE_STD.tolist(),
                "controllers_evaluated": CONTROLLERS,
            }
            with open(os.path.join(OUT_DIR, f"Rollout_plots_model_{ensemble_id}",
                                   "rollout_info.json"), "w") as f:
                json.dump(info, f, indent=2)

    # ---- one CSV with everything
    df = (pd.DataFrame(rollout_cost_records)
          .sort_values(by=["ensemble_id", "controller"])
          .reset_index(drop=True))
    csv_path = os.path.join(OUT_DIR, "Rollout costs.csv")
    df.to_csv(csv_path, index=False)

    # ---- summary tables
    wide = df.pivot(index="ensemble_id", columns="controller",
                    values="mean_normalized_rollout_cost")[CONTROLLERS]
    wide["NN_vs_LQR_%"] = 100.0 * (wide["LQR+NN"] / wide["LQR"] - 1.0)
    wide["ELQR_vs_LQR_%"] = 100.0 * (wide["LQR+ELQR"] / wide["LQR"] - 1.0)
    wide["ELQR_vs_NN_%"] = 100.0 * (wide["LQR+ELQR"] / wide["LQR+NN"] - 1.0)
    wide.to_csv(os.path.join(OUT_DIR, "Rollout costs summary.csv"))

    osc_cols = ["mean_abs_du", "rms_attitude_rad", "rms_body_rates", "rms_pos_err_last_10s"]
    osc = df.groupby("controller")[osc_cols].mean().loc[CONTROLLERS]

    split = df.groupby(["split", "controller"])["mean_normalized_rollout_cost"].mean().unstack()[CONTROLLERS]

    pd.set_option("display.width", 160)
    print("\n" + "=" * 90)
    print("Normalized rollout cost per model (negative % = better)")
    print("=" * 90)
    print(wide.round(4))
    print("\nMean per split:")
    print(split)
    print("\nOscillation metrics (mean over all models, lower = smoother):")
    print(osc)
    print("\nSaved:", csv_path)


if __name__ == "__main__":
    main()
