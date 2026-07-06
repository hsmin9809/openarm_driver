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

"""Safety checkers for OpenArm robot control."""

import numpy as np
from numpy.typing import ArrayLike

from .base_safety import Checker, CheckResult


class JointPosChecker(Checker):
    """Check that joint positions are within limits."""

    def __init__(self, joint_limits: ArrayLike):
        """Initialize joint position checker.

        Args:
            joint_limits: (8, 2) array of [min, max] limits.

        """
        self.joint_limits = np.asarray(joint_limits, dtype=float)

    def check(self, joint_positions: ArrayLike, **kwargs) -> CheckResult:
        """Run check."""
        positions = np.asarray(joint_positions, dtype=float)
        low = self.joint_limits[:, 0]
        high = self.joint_limits[:, 1]

        position_limited = np.clip(positions, low, high)
        violations = (positions < low) | (positions > high)

        if np.any(violations):
            violated_joints = np.where(violations)[0].tolist()
            return CheckResult(
                is_safe=False,
                force_stop=False,
                fixed_joint_positions=position_limited,
                message=f"Joint positions over limits at joints: {violated_joints}",
                check_type="joint_limits",
                details={"violated_joints": violated_joints},
            )

        return CheckResult(
            is_safe=True,
            message="All joint positions within limits.",
            check_type="joint_limits",
        )


class JointDeltaPosChecker(Checker):
    """Check that joint position changes don't exceed velocity limits."""

    def __init__(self, delta_limits: ArrayLike):
        """Initialize delta position checker.

        Args:
            delta_limits: Maximum allowed change per step for each joint.

        """
        self.delta_limits = np.asarray(delta_limits, dtype=float)

    def check(self, joint_positions: ArrayLike, **kwargs) -> CheckResult:
        """Run check."""
        driver = kwargs.get("driver")
        if driver is None or not hasattr(driver, "last_command"):
            return CheckResult(
                is_safe=True,
                message="No previous command to compare against.",
                check_type="joint_delta",
            )

        positions = np.asarray(joint_positions, dtype=float)
        delta = positions - driver.last_command
        clamped = driver.last_command + np.clip(delta, -self.delta_limits, self.delta_limits)

        if not np.allclose(clamped, positions):
            return CheckResult(
                is_safe=False,
                force_stop=False,
                fixed_joint_positions=clamped,
                message="Joint delta clamped to limits.",
                check_type="joint_delta",
            )

        return CheckResult(
            is_safe=True,
            message="All joint deltas within limits.",
            check_type="joint_delta",
        )
