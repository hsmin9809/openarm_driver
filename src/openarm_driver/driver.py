# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Driver for OpenArm."""

import time
from collections.abc import Callable, Iterator

import numpy as np
from numpy.typing import ArrayLike
import openarm_can as oa

from .config import Config, get_default_config
from .base_safety import Checker, CompositeChecker
from .safety import (
    JointPosChecker,
    JointDeltaPosChecker,
)


def _create_default_checker(arm_side: str, config: Config, control_hz: float) -> CompositeChecker:
    """Create basic checker with joint limits."""
    raw_limits = config.get_joint_limits(arm_side)
    offsets = config.get_joint_offsets(arm_side)
    # raw_limits are in raw motor coordinates; send_position receives logical
    # (logical = raw - offset), so convert limits to logical space.
    logical_limits = raw_limits - offsets[:, np.newaxis]
    # yaml values are in rad/s; convert to rad/step for per-call delta check.
    vel_limits = config.get_joint_delta_position_limits()
    delta_limits = np.asarray(vel_limits, dtype=float) / control_hz
    return CompositeChecker(
        [
            JointPosChecker(logical_limits),
            JointDeltaPosChecker(delta_limits),
        ]
    )


class SingleArmDriver:
    """Driver for single arm."""

    def __init__(
        self,
        arm_side: str,
        config: Config | None = None,
        kps: ArrayLike | None = None,
        kds: ArrayLike | None = None,
        safety_checker: Checker | None = None,
        control_hz: float = 20.0,
    ):
        """Initialize single arm driver.

        Args:
            arm_side: "left_arm" or "right_arm".
            config: Driver configuration. Uses default if None.
            kps: Proportional gains. Uses config defaults if None.
            kds: Derivative gains. Uses config defaults if None.
            safety_checker: Safety checker to use. If None, creates a basic
                           checker with joint limits.

        """
        self.config = config if config is not None else get_default_config()
        self.arm_side = arm_side
        self.can_interface = self.config.get_can_interface(self.arm_side)
        self.openarm = oa.OpenArm(self.can_interface, self.config.get_can_fd())
        self.latest_state = None
        self.started = False

        # Load joint offsets from config
        self.joint_offsets = self.config.get_joint_offsets(self.arm_side)

        # Load motor configuration from config
        motor_type_strs = self.config.get_motor_types()
        motor_types = [getattr(oa.MotorType, mt) for mt in motor_type_strs]
        send_ids = self.config.get_send_ids()
        recv_ids = self.config.get_recv_ids()
        self.gripper_posforce = self.config.get_gripper_posforce()
        self.gripper_posforce_limits = self.config.get_gripper_posforce_limits()

        # Initialize motors
        if self.gripper_posforce:
            self.num_mit_motors = 7
            self.openarm.init_arm_motors(motor_types[:-1], send_ids[:-1], recv_ids[:-1])
            self.openarm.init_gripper_motor(
                motor_types[-1], send_ids[-1], recv_ids[-1], oa.ControlMode.POS_FORCE
            )
        else:
            self.num_mit_motors = 8
            self.openarm.init_arm_motors(motor_types, send_ids, recv_ids)

        self.openarm.set_callback_mode_all(oa.CallbackMode.STATE)

        # Use provided gains or defaults from config
        self.kps = self.config.get_default_kps() if kps is None else np.array(kps)
        self.kds = self.config.get_default_kds() if kds is None else np.array(kds)

        # If no checker provided, create basic joint limit checker only
        self.safety_checker = (
            _create_default_checker(arm_side, self.config, control_hz)
            if safety_checker is None
            else safety_checker
        )

        # iterate until commutation is stable
        for _ in range(20):
            time.sleep(0.01)
            self.last_command = self.fetch_position(refresh=True)

    def start(self):
        """Start the arm."""
        self.openarm.set_callback_mode_all(oa.CallbackMode.STATE)
        self.openarm.enable_all()
        self.set_latest_state(timeout_us=500)
        self.openarm.refresh_all()
        self.set_latest_state(timeout_us=500)
        self._on_start()
        self.started = True

    def stop(self):
        """Stop the arm."""
        self._on_stop()
        self.openarm.disable_all()
        self.set_latest_state(timeout_us=1000)
        time.sleep(1)
        self.started = False

    def set_latest_state(self, timeout_us=300):
        """Update the state."""
        self.openarm.recv_all(timeout_us)
        motor_values = (
            (m.get_position(), m.get_velocity(), m.get_torque())
            for m in self._iter_motors()
        )
        qpos, qvel, qtau = zip(*motor_values)
        raw = np.array(qpos, dtype=float)
        self.latest_state = {
            "qpos": raw - self.joint_offsets,
            "qvel": np.array(qvel, dtype=float),
            "qtorque": np.array(qtau, dtype=float),
        }

    def fetch_state(self, refresh=True) -> dict[str, np.ndarray]:
        """Fetch the state."""
        if refresh:
            self.openarm.refresh_all()
        # TODO: maybe ?
        self.set_latest_state(timeout_us=300)
        return self.latest_state

    def fetch_position(self, refresh=True) -> np.ndarray:
        """Fetch the position."""
        return self.fetch_state(refresh=refresh)["qpos"]

    def fetch_velocity(self, refresh=True) -> np.ndarray:
        """Fetch the velocity."""
        return self.fetch_state(refresh=refresh)["qvel"]

    def fetch_torque(self, refresh=True) -> np.ndarray:
        """Fetch the torque."""
        return self.fetch_state(refresh=refresh)["qtorque"]

    def send_position(
        self,
        position: ArrayLike,
        velocity: ArrayLike | None = None,
        gravity_torque: ArrayLike | None = None,
        *,
        velocity_reference: ArrayLike | None = None,
        velocity_dt: float | None = None,
        velocity_feedforward: Callable[[np.ndarray], ArrayLike] | None = None,
    ):
        """Send a bounded target and optionally derive velocity AFTER position limiting.

        velocity_reference is the measured logical joint position (not a velocity).
        velocity_feedforward maps the resulting MIT target velocities to extra
        torque; its caller owns coefficient validation and torque caps.
        """
        position = np.asarray(position, dtype=float)
        if position.shape != self.joint_offsets.shape or not np.all(np.isfinite(position)):
            raise ValueError("Position must contain one finite value per configured motor")
        checked_result = self.safety_checker.check(position, driver=self)
        if not checked_result.is_safe:
            if checked_result.force_stop:
                raise RuntimeError(checked_result.message)
            if checked_result.fixed_joint_positions is not None:
                position = checked_result.fixed_joint_positions

        target_pos = np.asarray(position, dtype=float)
        vel = np.zeros(self.num_mit_motors) if velocity is None else np.asarray(velocity[:self.num_mit_motors], dtype=float)
        tau = np.zeros(self.num_mit_motors) if gravity_torque is None else np.asarray(gravity_torque[:self.num_mit_motors], dtype=float)
        if velocity_reference is not None:
            reference = np.asarray(velocity_reference, dtype=float)
            if (reference.shape not in (target_pos.shape, (7,)) or not np.all(np.isfinite(reference))
                    or not np.all(np.isfinite(target_pos))
                    or velocity_dt is None or not np.isfinite(velocity_dt) or velocity_dt <= 0):
                raise ValueError("velocity_reference must match the finite target; velocity_dt must be positive")
            count = min(len(reference), self.num_mit_motors)
            vel = vel.copy()
            vel[:count] = (target_pos[:count] - reference[:count]) / velocity_dt
        elif velocity_dt is not None or velocity_feedforward is not None:
            raise ValueError("Derived velocity/feedforward requires velocity_reference")
        ff = np.zeros(self.num_mit_motors)
        if velocity_feedforward is not None:
            ff = np.asarray(velocity_feedforward(vel.copy()), dtype=float)
        if (vel.shape != (self.num_mit_motors,) or tau.shape != vel.shape
                or ff.shape != vel.shape or not np.all(np.isfinite([vel, tau, ff]))):
            raise ValueError("MIT velocity and torque must be finite per-motor vectors")
        command_torque = tau + ff
        if not np.all(np.isfinite(command_torque)):
            raise ValueError("Combined torque must be finite")
        self.last_command = target_pos.copy()

        self.openarm.get_arm().mit_control_all(
            [
                oa.MITParam(
                    self.kps[i],
                    self.kds[i],
                    target_pos[i] + self.joint_offsets[i],
                    vel[i],
                    # 0,
                    command_torque[i],
                )
                for i in range(self.num_mit_motors)
            ]
        )
        if self.gripper_posforce:
            # TODO: Now We should multiply 10 to convert Nm to pu?
            self.openarm.get_gripper().set_position(
                target_pos[-1] + self.joint_offsets[-1],
                speed_rad_s=self.gripper_posforce_limits[0],
                torque_pu=self.gripper_posforce_limits[1] / 4.5,
            )

        self.set_latest_state(timeout_us=300)
        return {
            "position": target_pos.copy(),
            "velocity": vel.copy(),
            "base_torque": tau.copy(),
            "velocity_ff_torque": ff.copy(),
            "torque": command_torque.copy(),
        }

    def smooth_move(
        self,
        position: ArrayLike,
        hz: float,
        duration: float,
        gravity_fn: Callable[[], ArrayLike] | None = None,
    ):
        """Move the arm smoothly by interpolating the trajectory to the final position."""
        num_steps = int(hz * duration)
        if num_steps <= 0:
            raise ValueError(
                f"smooth_move step calc error: hz {hz}, duration: {duration}"
            )
        for smoothed_position in self._interpolate(
            np.array([np.array(self.last_command), np.array(position)]),
            num_steps=num_steps,
        ):
            tau = gravity_fn() if gravity_fn is not None else None
            self.send_position(smoothed_position, gravity_torque=tau)
            time.sleep(1.0 / hz)

    def move_to_start_position(self):
        """Move to start position."""
        start_config = self.config.get_start_config()
        if start_config["moves"]:
            for move in start_config["moves"]:
                self.smooth_move(
                    move["position"][self.arm_side],
                    hz=move["hz"],
                    duration=move["duration"],
                )

    def move_to_stop_position(self):
        """Move to end position."""
        end_config = self.config.get_stop_config()
        if end_config["moves"]:
            for move in end_config["moves"]:
                self.smooth_move(
                    move["position"][self.arm_side],
                    hz=move["hz"],
                    duration=move["duration"],
                )

    def _interpolate(
        self, positions: np.ndarray, num_steps: int
    ) -> Iterator[np.ndarray]:
        for a0, a1 in zip(positions[:-1], positions[1:]):
            yield from np.linspace(a0, a1, num_steps)

    def _iter_motors(self) -> Iterator:
        yield from self.openarm.get_arm().get_motors()
        if self.gripper_posforce:
            yield self.openarm.get_gripper().get_motors()[0]

    def _on_start(self):
        self.move_to_start_position()

    def _on_stop(self):
        self.move_to_stop_position()
