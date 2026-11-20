"""Diagnostic evaluation for the supplied Sample Factory enjoy implementation."""
import argparse
import csv
import inspect
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class Metrics:
    def __init__(self, folder, sigma, probes, every, dt_override):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=False)
        self.writer = SummaryWriter(str(folder / "tensorboard"))
        self.file = (folder / "metrics.csv").open("w", newline="", encoding="utf-8")
        self.fields = ["step", "episode", "agent", "dt", "sensitivity_xy",
                       "command_change_rate", "command_second_difference",
                       "mean_change_rate_raw", "saturation_fraction"]
        self.csv = csv.DictWriter(self.file, fieldnames=self.fields)
        self.csv.writeheader()
        self.sigma, self.probes, self.every = sigma, probes, every
        self.dt_override = dt_override
        self.rng = np.random.default_rng(2026)
        self.step = 0
        self.prev = self.prev2 = self.prev_raw = None
        self.episodes = None
        self.ready = False
        # Logging controls only. CSV formulas and sampling are unchanged.
        self.tensorboard_every = max(1, int(os.environ.get('METRICS_TB_EVERY', '1')))
        self.tensorboard_agents = max(0, int(os.environ.get('METRICS_TB_AGENTS', '1000000')))

    def setup(self, cfg, env):
        if str(cfg.device) != "cpu" or cfg.use_rnn:
            raise RuntimeError("This version supports CPU and use_rnn=False only.")
        if cfg.quads_obs_repr != "xyz_vxyz_R_omega_floor":
            raise RuntimeError("Unsupported observation representation.")
        if cfg.normalize_input or cfg.obs_scale != 1.0 or cfg.obs_subtract_mean != 0.0:
            raise RuntimeError("Sensitivity requires unnormalized observations.")
        if cfg.env_frameskip != 1 or cfg.eval_env_frameskip != 1:
            raise RuntimeError("Use env_frameskip=1 and eval_env_frameskip=1.")
        if not cfg.eval_deterministic:
            raise RuntimeError("Use --eval_deterministic=True.")
        self.low = np.asarray(env.action_space.low, dtype=float)
        self.high = np.asarray(env.action_space.high, dtype=float)
        if self.low.shape != (4,) or not np.allclose(self.low, -1) or not np.allclose(self.high, 1):
            raise RuntimeError("Expected four RawControl commands in [-1, 1].")
        dt = self.dt_override
        if dt is None:
            pending, seen = [env], set()
            while pending:
                item = pending.pop()
                if id(item) in seen:
                    continue
                seen.add(id(item))
                freq = getattr(item, "control_freq", None)
                if freq is not None:
                    dt = 1.0 / float(freq)
                    break
                for key in ("unwrapped", "env"):
                    child = getattr(item, key, None)
                    if child is not None:
                        pending.append(child)
                children = getattr(item, "envs", None)
                if children is not None and len(children):
                    pending.append(children[0])
        if dt is None or dt <= 0:
            raise RuntimeError("Cannot find control dt. Supply --metrics-dt after checking control_freq.")
        self.dt = dt
        meta = dict(sigma_xy_m=self.sigma, probes=self.probes, sample_every=self.every,
                    control_dt=dt, sensitivity="RMS change of clipped, [0,1]-scaled mean command under iid Gaussian perturbations of relative-goal x,y only",
                    confidence_probability=False, argv=sys.argv, config=vars(cfg))
        (self.folder / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        self.ready = True
        print("METRICS DIRECTORY:", self.folder.resolve(), "CONTROL DT:", dt)

    def observe(self, cfg, env, model, obs, states):
        if not self.ready:
            self.setup(cfg, env)
        dist = model.action_distribution()
        if not hasattr(dist, "means"):
            raise RuntimeError("Expected distribution.means.")
        raw = dist.means.detach().cpu().numpy().copy()
        command = (np.clip(raw, self.low, self.high) - self.low) / (self.high - self.low)
        n = len(command)
        if self.episodes is None:
            self.episodes = np.zeros(n, dtype=int)
        change = np.full(n, np.nan)
        second = np.full(n, np.nan)
        raw_change = np.full(n, np.nan)
        sensitivity = np.full(n, np.nan)
        rms = lambda x: np.sqrt(np.mean(x * x, axis=-1))
        if self.prev is not None:
            change = rms(command - self.prev) / self.dt
            raw_change = rms(raw - self.prev_raw) / self.dt
        if self.prev2 is not None:
            second = rms(command - 2 * self.prev + self.prev2) / self.dt**2
        if self.step % self.every == 0:
            if "obs" not in obs or obs["obs"].shape[-1] < 19:
                raise RuntimeError("Expected obs['obs'] with at least 19 features.")
            values = []
            try:
                with torch.random.fork_rng(devices=[]):
                    for _ in range(self.probes):
                        perturbed = {k: v.clone() for k, v in obs.items()}
                        delta = self.rng.normal(0, self.sigma, size=(n, 2))
                        perturbed["obs"][:, :2] += torch.as_tensor(delta, dtype=obs["obs"].dtype, device=obs["obs"].device)
                        model(perturbed, states.clone())
                        probe = model.action_distribution().means.detach().cpu().numpy()
                        probe = (np.clip(probe, self.low, self.high) - self.low) / (self.high - self.low)
                        values.append(np.mean((probe - command)**2, axis=-1))
                sensitivity = np.sqrt(np.mean(values, axis=0))
            finally:
                model.last_action_distribution = dist
        saturation = np.mean((raw <= self.low) | (raw >= self.high), axis=-1)
        arrays = dict(sensitivity_xy=sensitivity, command_change_rate=change,
                      command_second_difference=second, mean_change_rate_raw=raw_change,
                      saturation_fraction=saturation)
        self.last_arrays = arrays
        for agent in range(n):
            row = dict(step=self.step, episode=int(self.episodes[agent]), agent=agent, dt=self.dt)
            row.update({k: float(v[agent]) for k, v in arrays.items()})
            self.csv.writerow(row)
            if agent < self.tensorboard_agents and self.step % self.tensorboard_every == 0:
                for key, array in arrays.items():
                    if np.isfinite(array[agent]):
                        self.writer.add_scalar(f"drone_{agent}/{key}", array[agent], self.step)
        for key, array in arrays.items():
            valid = array[np.isfinite(array)]
            if len(valid) and self.step % self.tensorboard_every == 0:
                self.writer.add_scalar(f"swarm_mean/{key}", float(valid.mean()), self.step)
                self.writer.add_scalar(f"swarm_max/{key}", float(valid.max()), self.step)
        self.prev2 = None if self.prev is None else self.prev.copy()
        self.prev, self.prev_raw = command.copy(), raw.copy()
        self.step += 1
        if self.step % 100 == 0:
            self.file.flush()
            self.writer.flush()

    def done(self, dones):
        mask = np.asarray(dones, dtype=bool).reshape(-1)
        if not mask.any() or self.episodes is None:
            return
        self.episodes[mask] += 1
        for name in ("prev", "prev2", "prev_raw"):
            array = getattr(self, name)
            if array is not None:
                array[mask] = np.nan

    def close(self):
        self.writer.close()
        self.file.close()
        print("METRICS SAVED:", self.folder.resolve())


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--metrics-dir", default=None)
    parser.add_argument("--metrics-sigma", type=float, default=0.02)
    parser.add_argument("--metrics-probes", type=int, default=8)
    parser.add_argument("--metrics-every", type=int, default=10)
    parser.add_argument("--metrics-dt", type=float, default=None)
    args, remaining = parser.parse_known_args()
    if args.metrics_sigma <= 0 or args.metrics_probes < 2 or args.metrics_every < 1:
        raise SystemExit("Invalid metrics settings.")
    sys.argv = [sys.argv[0]] + remaining
    from swarm_rl.train import parse_swarm_cfg, register_swarm_components
    import importlib
    enjoy_module = importlib.import_module("sample_factory.enjoy")
    register_swarm_components()
    cfg = parse_swarm_cfg(evaluation=True)
    folder = Path(args.metrics_dir) if args.metrics_dir else Path.home() / "quad-metrics" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    source = inspect.getsource(enjoy_module.enjoy)
    target = "            policy_outputs = actor_critic(normalized_obs, rnn_states)"
    replacement = target + "\n            _metrics.observe(cfg, env, actor_critic, normalized_obs, rnn_states)"
    done_target = "                dones = dones.cpu().numpy()"
    done_replacement = done_target + "\n                _metrics.done(dones)"
    if source.count(target) != 1 or source.count(done_target) != 1:
        raise SystemExit("Installed enjoy.py differs from supplied code. No files were modified.")
    source = source.replace(target, replacement, 1).replace(done_target, done_replacement, 1)
    compiled = compile(source, "<diagnostic_enjoy>", "exec")
    metrics = Metrics(folder, args.metrics_sigma, args.metrics_probes, args.metrics_every, args.metrics_dt)
    namespace = dict(enjoy_module.__dict__)
    namespace["_metrics"] = metrics
    try:
        exec(compiled, namespace)
        namespace["enjoy"](cfg)
    finally:
        metrics.close()


if __name__ == "__main__":
    main()
