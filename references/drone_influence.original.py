"""Local block-scaling sensitivity. Does not change actions sent to the environment."""
import csv
import json
import os
import numpy as np
import torch
import drone_metrics as base

Original = base.Metrics


class InfluenceMetrics(Original):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = float(os.environ.get("INFLUENCE_ALPHA", "0.01"))
        self.threshold = float(os.environ.get("INFLUENCE_MIN_EFFECT", "0.000001"))
        if not 0 < self.alpha < 1 or self.threshold <= 0:
            raise ValueError("Invalid influence settings")
        self.influence_file = (self.folder / "influence.csv").open("w", newline="", encoding="utf-8")
        fields = ["step", "episode", "agent", "alpha", "neighbors_deviation", "obstacles_deviation", "total_deviation", "obstacle_share", "valid"]
        self.influence_csv = csv.DictWriter(self.influence_file, fieldnames=fields)
        self.influence_csv.writeheader()
        self.layout = None

    def get_layout(self, model, obs):
        candidates = set()
        for module in model.modules():
            if hasattr(module, "self_obs_dim") and hasattr(module, "all_neighbor_obs_size"):
                start = int(module.self_obs_dim)
                end = start + int(module.all_neighbor_obs_size)
                candidates.add((start, end))
        if len(candidates) != 1:
            raise RuntimeError(f"Cannot uniquely determine observation slices: {candidates}")
        start, end = next(iter(candidates))
        width = int(obs["obs"].shape[-1])
        if not 0 < start < end < width:
            raise RuntimeError(f"Invalid group layout: {(start, end, width)}")
        self.layout = (start, end, width)
        settings = dict(method="central block scaling in normalized input coordinates", alpha=self.alpha,
                        neighbors_slice=[start, end], obstacles_slice=[end, width],
                        min_total_effect=self.threshold,
                        interpretation="relative local sensitivity, not attention weights or causal attribution",
                        perturbation="each complete group independently multiplied by 1-alpha and 1+alpha; self block unchanged")
        (self.folder / "influence_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        print("INFLUENCE GROUPS:", settings)

    def observe(self, cfg, env, model, obs, states):
        super().observe(cfg, env, model, obs, states)
        step = self.step - 1
        if step % self.every:
            return
        if self.layout is None:
            self.get_layout(model, obs)
        start, end, width = self.layout
        original_distribution = model.action_distribution()
        effects = {}
        try:
            with torch.no_grad(), torch.random.fork_rng(devices=[]):
                for name, a, b in (("neighbors", start, end), ("obstacles", end, width)):
                    commands = []
                    for sign in (-1.0, 1.0):
                        changed = {k: v.clone() for k, v in obs.items()}
                        changed["obs"][:, a:b] *= 1.0 + sign * self.alpha
                        model(changed, states.clone())
                        raw = model.action_distribution().means.detach().cpu().numpy().copy()
                        commands.append((np.clip(raw, self.low, self.high) - self.low) / (self.high - self.low))
                    effects[name] = 0.5 * np.sqrt(np.mean((commands[1] - commands[0])**2, axis=-1))
        finally:
            model.last_action_distribution = original_distribution
        total = effects["neighbors"] + effects["obstacles"]
        valid = total > self.threshold
        share = np.full(len(total), np.nan)
        np.divide(effects["obstacles"], total, out=share, where=valid)
        for agent in range(len(total)):
            row = dict(step=step, episode=int(self.episodes[agent]), agent=agent, alpha=self.alpha,
                       neighbors_deviation=float(effects["neighbors"][agent]),
                       obstacles_deviation=float(effects["obstacles"][agent]),
                       total_deviation=float(total[agent]), obstacle_share=float(share[agent]),
                       valid=int(valid[agent]))
            self.influence_csv.writerow(row)
            prefix = f"drone_{agent}/influence"
            for key in ("neighbors_deviation", "obstacles_deviation", "total_deviation", "valid"):
                self.writer.add_scalar(f"{prefix}/{key}", row[key], step)
            self.writer.add_scalar(f"{prefix}/obstacle_share", float(share[agent]), step)
        self.influence_file.flush()
        self.writer.flush()

    def close(self):
        try:
            self.influence_file.close()
        finally:
            super().close()


if __name__ == "__main__":
    base.Metrics = InfluenceMetrics
    base.main()
