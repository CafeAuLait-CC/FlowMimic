from dataclasses import dataclass


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _lerp(start, end, fraction):
    fraction = _clamp01(fraction)
    return float(start) + (float(end) - float(start)) * fraction


def _interpolate_probs(start, end, fraction):
    values = tuple(_lerp(a, b, fraction) for a, b in zip(start, end))
    total = sum(values)
    if total <= 0.0:
        raise ValueError("Condition probabilities must have a positive sum")
    return tuple(value / total for value in values)


@dataclass(frozen=True)
class UnifiedRound0State:
    completed_updates: int
    phase: str
    learning_rate: float
    condition_weight_scale: float
    condition_choices: tuple
    condition_probs: tuple
    condition_pattern_choices: tuple
    condition_pattern_probs: tuple
    solver_steps: tuple
    condition_joint_quotas: tuple = ()


class UnifiedRound0Curriculum:
    """Update-based schedule for the AIST unified round-0 flow run."""

    def __init__(self, config, max_updates=None):
        self.config = dict(config)
        self.reference_updates_per_epoch = int(
            self.config["reference_updates_per_epoch"]
        )
        self.warmup_end = int(self.config["warmup_end_update"])
        self.dense_end = int(self.config["dense_end_update"])
        self.mid_end = int(self.config["mid_end_update"])
        self.sparse_end = int(self.config["sparse_end_update"])
        self.cond_start = int(self.config["condition_start_update"])
        self.cond_full = int(self.config["condition_full_update"])
        self.max_updates = int(
            max_updates
            if max_updates is not None
            else self.config.get("max_updates", self.sparse_end)
        )
        self.optional_max_updates = int(
            self.config.get("optional_max_updates", self.max_updates)
        )
        self.eval_every_updates = int(self.config["eval_every_updates"])
        self.lr_peak = float(self.config["lr_peak"])
        self.lr_mid = float(self.config["lr_mid"])
        self.lr_final = float(self.config["lr_final"])
        self.condition_choices = tuple(
            int(value) for value in self.config["condition_choices"]
        )
        self.dense_probs = self._probabilities("dense_probs")
        self.mid_probs = self._probabilities("mid_probs")
        self.final_probs = self._probabilities("final_probs")
        self.condition_pattern_choices = tuple(
            str(value)
            for value in self.config.get("condition_pattern_choices", ["even"])
        )
        self.pattern_start_probs = self._probabilities(
            "pattern_start_probs",
            default=[1.0],
        )
        self.pattern_final_probs = self._probabilities(
            "pattern_final_probs",
            default=[1.0],
        )
        self.pattern_ramp_start = int(
            self.config.get("pattern_ramp_start_update", self.dense_end)
        )
        self.pattern_ramp_end = int(
            self.config.get("pattern_ramp_end_update", self.mid_end)
        )
        self.condition_joint_quotas = tuple(
            (
                str(item["pattern"]),
                int(item["count"]),
                float(item["fraction"]),
            )
            for item in self.config.get("condition_joint_quotas", [])
        )
        self.joint_quota_start = int(
            self.config.get("joint_quota_start_update", self.mid_end)
        )
        self.joint_quota_full = int(
            self.config.get("joint_quota_full_update", self.sparse_end)
        )
        self.solver_steps_dense = tuple(
            int(value) for value in self.config.get("solver_steps_dense", [16])
        )
        self.solver_steps_sparse = tuple(
            int(value)
            for value in self.config.get("solver_steps_sparse", [8, 16])
        )
        self._validate()

    def _probabilities(self, key, default=None):
        values = tuple(float(value) for value in self.config.get(key, default))
        total = sum(values)
        if total <= 0.0:
            raise ValueError(f"{key} must have a positive sum")
        return tuple(value / total for value in values)

    def _validate(self):
        milestones = (
            self.warmup_end,
            self.dense_end,
            self.cond_start,
            self.cond_full,
            self.mid_end,
            self.sparse_end,
        )
        if self.reference_updates_per_epoch <= 0:
            raise ValueError("reference_updates_per_epoch must be positive")
        if not (
            0 < self.warmup_end <= self.dense_end
            <= self.cond_start <= self.cond_full
            <= self.mid_end <= self.sparse_end
        ):
            raise ValueError(f"Invalid curriculum milestones: {milestones}")
        if self.max_updates <= 0 or self.eval_every_updates <= 0:
            raise ValueError("max_updates and eval_every_updates must be positive")
        expected = len(self.condition_choices)
        for name, values in (
            ("dense_probs", self.dense_probs),
            ("mid_probs", self.mid_probs),
            ("final_probs", self.final_probs),
        ):
            if len(values) != expected:
                raise ValueError(
                    f"{name} must have {expected} entries, got {len(values)}"
                )
            if any(value < 0.0 for value in values):
                raise ValueError(f"{name} cannot contain negative probabilities")
        if not self.solver_steps_dense or not self.solver_steps_sparse:
            raise ValueError("Solver-step schedules cannot be empty")
        expected_patterns = len(self.condition_pattern_choices)
        if len(self.pattern_start_probs) != expected_patterns:
            raise ValueError("Pattern start probabilities must match choices")
        if len(self.pattern_final_probs) != expected_patterns:
            raise ValueError("Pattern final probabilities must match choices")
        if not (
            0 <= self.pattern_ramp_start < self.pattern_ramp_end <= self.max_updates
        ):
            raise ValueError("Invalid unified pattern-ramp milestones")
        if not (
            0 <= self.joint_quota_start < self.joint_quota_full <= self.max_updates
        ):
            raise ValueError("Invalid unified joint-quota milestones")
        quota_total = 0.0
        for pattern, count, fraction in self.condition_joint_quotas:
            if pattern not in self.condition_pattern_choices:
                raise ValueError(f"Unknown joint-quota pattern: {pattern}")
            if count not in self.condition_choices:
                raise ValueError(f"Unknown joint-quota condition count: {count}")
            if fraction < 0.0:
                raise ValueError("Joint-quota fractions cannot be negative")
            quota_total += fraction
        if quota_total > 1.0 + 1e-8:
            raise ValueError("Joint-quota fractions cannot sum above one")

    def state(self, completed_updates):
        updates = max(0, int(completed_updates))
        if updates < self.warmup_end:
            phase = "warmup_dense"
            lr = self.lr_peak * (updates + 1) / self.warmup_end
            probs = self.dense_probs
        elif updates < self.dense_end:
            phase = "dense_velocity"
            lr = self.lr_peak
            probs = self.dense_probs
        elif updates < self.mid_end:
            phase = "mid_density_ramp"
            fraction = (updates - self.dense_end) / max(
                self.mid_end - self.dense_end, 1
            )
            lr = _lerp(self.lr_peak, self.lr_mid, fraction)
            probs = _interpolate_probs(self.dense_probs, self.mid_probs, fraction)
        elif updates < self.sparse_end:
            phase = "sparse_density_ramp"
            fraction = (updates - self.mid_end) / max(
                self.sparse_end - self.mid_end, 1
            )
            lr = _lerp(self.lr_mid, self.lr_final, fraction)
            probs = _interpolate_probs(self.mid_probs, self.final_probs, fraction)
        else:
            phase = "sparse_hold"
            lr = self.lr_final
            probs = self.final_probs

        if updates <= self.cond_start:
            condition_weight_scale = 0.0
        elif updates >= self.cond_full:
            condition_weight_scale = 1.0
        else:
            condition_weight_scale = (updates - self.cond_start) / max(
                self.cond_full - self.cond_start, 1
            )
        solver_steps = (
            self.solver_steps_dense
            if updates < self.mid_end
            else self.solver_steps_sparse
        )
        if updates <= self.pattern_ramp_start:
            pattern_probs = self.pattern_start_probs
        elif updates >= self.pattern_ramp_end:
            pattern_probs = self.pattern_final_probs
        else:
            pattern_fraction = (updates - self.pattern_ramp_start) / max(
                self.pattern_ramp_end - self.pattern_ramp_start,
                1,
            )
            pattern_probs = _interpolate_probs(
                self.pattern_start_probs,
                self.pattern_final_probs,
                pattern_fraction,
            )
        if updates <= self.joint_quota_start:
            quota_scale = 0.0
        elif updates >= self.joint_quota_full:
            quota_scale = 1.0
        else:
            quota_scale = (updates - self.joint_quota_start) / max(
                self.joint_quota_full - self.joint_quota_start,
                1,
            )
        joint_quotas = tuple(
            (pattern, count, fraction * quota_scale)
            for pattern, count, fraction in self.condition_joint_quotas
            if fraction * quota_scale > 0.0
        )
        return UnifiedRound0State(
            completed_updates=updates,
            phase=phase,
            learning_rate=float(lr),
            condition_weight_scale=float(condition_weight_scale),
            condition_choices=self.condition_choices,
            condition_probs=tuple(float(value) for value in probs),
            condition_pattern_choices=self.condition_pattern_choices,
            condition_pattern_probs=tuple(float(value) for value in pattern_probs),
            solver_steps=solver_steps,
            condition_joint_quotas=joint_quotas,
        )

    def metadata(self):
        return {
            "name": self.config.get("name", "unified_round0"),
            "max_updates": self.max_updates,
            "optional_max_updates": self.optional_max_updates,
            "reference_updates_per_epoch": self.reference_updates_per_epoch,
            "eval_every_updates": self.eval_every_updates,
            "config": dict(self.config),
        }


class SparsePatternPhase1Curriculum:
    """Update-based sparse-pattern fine-tuning after unified Round 0."""

    def __init__(self, config, max_updates=None):
        self.config = dict(config)
        self.source_updates = int(self.config["source_optimizer_updates"])
        self.reference_updates_per_epoch = int(
            self.config["reference_updates_per_epoch"]
        )
        self.pattern_ramp_updates = int(self.config["pattern_ramp_updates"])
        self.relative_max_updates = int(self.config["relative_max_updates"])
        self.relative_optional_max_updates = int(
            self.config.get("relative_optional_max_updates", self.relative_max_updates)
        )
        self.max_updates = int(
            max_updates
            if max_updates is not None
            else self.source_updates + self.relative_max_updates
        )
        self.optional_max_updates = self.source_updates + self.relative_optional_max_updates
        self.eval_every_updates = int(self.config["eval_every_updates"])
        self.lr_peak = float(self.config["learning_rate"])
        self.condition_weight_scale = float(
            self.config.get("condition_weight_scale", 1.0)
        )
        self.condition_choices = tuple(
            int(value) for value in self.config["condition_choices"]
        )
        self.condition_probs = self._probabilities("condition_probs")
        self.condition_pattern_choices = tuple(
            str(value) for value in self.config["condition_pattern_choices"]
        )
        self.pattern_start_probs = self._probabilities("pattern_start_probs")
        self.pattern_final_probs = self._probabilities("pattern_final_probs")
        self.condition_joint_quotas = tuple(
            (
                str(item["pattern"]),
                int(item["count"]),
                float(item["fraction"]),
            )
            for item in self.config.get("condition_joint_quotas", [])
        )
        self.solver_steps = tuple(
            int(value) for value in self.config.get("solver_steps", [8, 16])
        )
        self._validate()

    def _probabilities(self, key):
        values = tuple(float(value) for value in self.config[key])
        total = sum(values)
        if total <= 0.0:
            raise ValueError(f"{key} must have a positive sum")
        return tuple(value / total for value in values)

    def _validate(self):
        if self.source_updates < 0 or self.reference_updates_per_epoch <= 0:
            raise ValueError("Invalid Phase 1 source/reference updates")
        if not (0 < self.pattern_ramp_updates <= self.relative_max_updates):
            raise ValueError("Invalid Phase 1 pattern-ramp duration")
        if self.max_updates <= self.source_updates:
            raise ValueError("Phase 1 max_updates must exceed source updates")
        if self.eval_every_updates <= 0 or not self.solver_steps:
            raise ValueError("Invalid Phase 1 eval or solver schedule")
        if len(self.condition_probs) != len(self.condition_choices):
            raise ValueError("Condition choices/probabilities must match")
        expected_patterns = len(self.condition_pattern_choices)
        if len(self.pattern_start_probs) != expected_patterns:
            raise ValueError("Pattern start probabilities must match choices")
        if len(self.pattern_final_probs) != expected_patterns:
            raise ValueError("Pattern final probabilities must match choices")
        quota_total = 0.0
        for pattern, count, fraction in self.condition_joint_quotas:
            if pattern not in self.condition_pattern_choices:
                raise ValueError(f"Unknown joint-quota pattern: {pattern}")
            if count not in self.condition_choices:
                raise ValueError(f"Unknown joint-quota condition count: {count}")
            if fraction < 0.0:
                raise ValueError("Joint-quota fractions cannot be negative")
            quota_total += fraction
        if quota_total > 1.0 + 1e-8:
            raise ValueError("Joint-quota fractions cannot sum above one")

    def state(self, completed_updates):
        updates = max(0, int(completed_updates))
        relative_updates = max(0, updates - self.source_updates)
        if relative_updates < self.pattern_ramp_updates:
            phase = "pattern_ramp"
            fraction = relative_updates / max(self.pattern_ramp_updates, 1)
            pattern_probs = _interpolate_probs(
                self.pattern_start_probs,
                self.pattern_final_probs,
                fraction,
            )
        else:
            phase = "pattern_hold"
            pattern_probs = self.pattern_final_probs
        return UnifiedRound0State(
            completed_updates=updates,
            phase=phase,
            learning_rate=self.lr_peak,
            condition_weight_scale=self.condition_weight_scale,
            condition_choices=self.condition_choices,
            condition_probs=self.condition_probs,
            condition_pattern_choices=self.condition_pattern_choices,
            condition_pattern_probs=tuple(float(value) for value in pattern_probs),
            solver_steps=self.solver_steps,
            condition_joint_quotas=self.condition_joint_quotas,
        )

    def metadata(self):
        return {
            "name": self.config.get("name", "sparse_pattern_phase1"),
            "source_optimizer_updates": self.source_updates,
            "max_updates": self.max_updates,
            "optional_max_updates": self.optional_max_updates,
            "reference_updates_per_epoch": self.reference_updates_per_epoch,
            "eval_every_updates": self.eval_every_updates,
            "config": dict(self.config),
        }
