"""Bound host compensation while retaining motor-side MIT PD.

Torque-squared exposure is an I^2t proxy, not a calibrated thermal model.
All arrays cover the seven arm axes. The POS_FORCE gripper is independent.
Limits apply only to the MIT torque field (gravity + FF), NOT total motor
torque. Native PD can exceed these limits, including after FF is removed.
"""

from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class TorqueProtectionSettings:
    continuous_nm: np.ndarray
    absolute_nm: np.ndarray
    budget_s: float = 10.0
    reference_ratio: float = 1.5
    recovery_s: float = 30.0
    recovery_ratio: float = 0.8
    release_fraction: float = 0.25
    ff_fade_s: float = 2.0
    limit_fade_s: float = 3.0
    restore_s: float = 10.0
    rotor_derate_c: float = 75.0
    rotor_full_derate_c: float = 90.0
    mos_derate_c: float = 85.0
    mos_full_derate_c: float = 105.0

    @classmethod
    def from_mapping(cls, value):
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("Unknown torque_protection setting")
        arrays = {}
        for name in ("continuous_nm", "absolute_nm"):
            arr = np.asarray(value.get(name), dtype=float)
            if arr.shape != (7,) or not np.all(np.isfinite(arr)) or np.any(arr <= 0):
                raise ValueError(f"torque_protection.{name} requires seven positive finite values")
            arrays[name] = arr.copy()
        settings = cls(**arrays, **{k: float(v) for k, v in value.items() if k not in arrays})
        if np.any(settings.continuous_nm >= settings.absolute_nm):
            raise ValueError("continuous_nm must be below absolute_nm")
        scalars = [getattr(settings, k) for k in cls.__dataclass_fields__ if k not in arrays]
        if not all(np.isfinite(v) and v > 0 for v in scalars):
            raise ValueError("torque_protection values must be finite and positive")
        if (settings.reference_ratio <= 1 or not 0 < settings.recovery_ratio < 1
                or not 0 < settings.release_fraction < 1
                or settings.rotor_derate_c >= settings.rotor_full_derate_c
                or settings.mos_derate_c >= settings.mos_full_derate_c):
            raise ValueError("Invalid torque_protection ratios or temperature range")
        return settings


class TorqueProtection:
    def __init__(self, settings, hz, state_timeout_s):
        self.settings = settings
        self.hz = hz
        self.state_timeout_s = state_timeout_s
        self.load_s = np.zeros(7)
        self.derating = np.zeros(7, dtype=bool)
        self.ff_reduction = np.zeros(7)
        self.limit_reduction = np.zeros(7)
        self.last_time = None
        self.last_torque = np.zeros(7)
        self.last_gravity = np.zeros(7)
        self.fault = None

    @property
    def limit_nm(self):
        s = self.settings
        return s.absolute_nm - self.limit_reduction * (s.absolute_nm - s.continuous_nm)

    def apply(self, gravity, ff, state, feedback, now=None):
        if self.fault is not None:
            raise RuntimeError(f"Torque protection fault: {self.fault}") from self.fault
        try:
            return self._apply(gravity, ff, state, feedback,
                               time.monotonic() if now is None else now)
        except Exception as exc:
            self.fault = exc
            raise

    def _apply(self, gravity, ff, state, feedback, now):
        s = self.settings
        values = {}
        for key in ("qpos", "qvel", "qtorque", "temperature_rotor", "temperature_mos"):
            arr = np.asarray(state[key], dtype=float)
            if arr.ndim != 1 or len(arr) < 7 or not np.all(np.isfinite(arr[:7])):
                raise ValueError(f"Invalid torque protection feedback: {key}")
            values[key] = arr[:7]
        ages = np.asarray(feedback["age_s"], dtype=float)
        counts = np.asarray(feedback["responses"])
        if (ages.shape != (7,) or counts.shape != (7,) or not np.all(np.isfinite(ages))
                or np.any(ages < 0) or np.any(ages > self.state_timeout_s)
                or not np.all(np.isfinite(counts)) or np.any(counts <= 0)):
            raise RuntimeError("Stale/missing feedback for compensation protection")
        vectors = [np.asarray(v, dtype=float)[:7] for v in (gravity, ff)]
        if any(v.shape != (7,) or not np.all(np.isfinite(v)) for v in vectors):
            raise ValueError("Compensation requires seven finite command values")
        gravity, ff = vectors
        if not np.isfinite(now) or (self.last_time is not None and now < self.last_time):
            raise ValueError("Torque protection clock must be monotonic")
        dt = 0.0 if self.last_time is None else now - self.last_time

        # Motor feedback includes native PD. The host compensation command is
        # not total torque (PD can reinforce or oppose it), so integrate measured
        # torque only. New actions and resets do not erase accumulated load.
        ratio = np.abs(values["qtorque"]) / s.continuous_nm
        with np.errstate(over="raise", invalid="raise"):
            charge = np.maximum(ratio ** 2 - 1.0, 0.0) / (s.reference_ratio ** 2 - 1.0)
            cool = np.where(ratio < s.recovery_ratio, s.budget_s / s.recovery_s, 0.0)
            self.load_s = np.clip(self.load_s + (charge - cool) * dt, 0.0, 2 * s.budget_s)
        self.derating |= self.load_s >= s.budget_s
        self.derating &= self.load_s > s.budget_s * s.release_fraction

        thermal = np.clip(np.maximum(
            (values["temperature_rotor"] - s.rotor_derate_c) / (s.rotor_full_derate_c - s.rotor_derate_c),
            (values["temperature_mos"] - s.mos_derate_c) / (s.mos_full_derate_c - s.mos_derate_c),
        ), 0.0, 1.0)
        desired = np.maximum(self.derating.astype(float), thermal)
        for name, fade_s in (("ff_reduction", s.ff_fade_s), ("limit_reduction", s.limit_fade_s)):
            current = getattr(self, name)
            current += np.clip(desired - current, -dt / s.restore_s, dt / fade_s)
            # A hot startup must not get a new burst allowance.
            current[:] = np.maximum(current, thermal)

        scaled_ff = ff * (1.0 - self.ff_reduction)
        requested = gravity + scaled_ff
        if not np.all(np.isfinite(requested)):
            raise ValueError("Nonfinite compensation torque")
        torque = np.clip(requested, -self.limit_nm, self.limit_nm)
        self.last_time = now
        self.last_torque = torque.copy()
        self.last_gravity = gravity.copy()
        return {"torque": torque, "velocity_ff_torque": scaled_ff,
                "requested_compensation_torque": requested,
                "compensation_limit_nm": self.limit_nm.copy(),
                "overload_s": self.load_s.copy(), "ff_scale": 1.0 - self.ff_reduction,
                "temperature_rotor_c": values["temperature_rotor"].copy(),
                "temperature_mos_c": values["temperature_mos"].copy(),
                "compensation_saturated": (np.abs(requested) > self.limit_nm).astype(float)}

    def gravity_hold(self):
        """Fault fallback: remove motion PD/FF without abruptly removing gravity.

        No stale-state PD computation. This is one bounded command, not a
        replacement for a motor-side communication watchdog.
        """
        limit = np.minimum(self.limit_nm, self.settings.continuous_nm)
        self.last_torque = np.clip(self.last_gravity, -limit, limit)
        return self.last_torque.copy()
