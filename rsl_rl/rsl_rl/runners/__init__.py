# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .on_policy_runner import OnPolicyRunner
from .dreamwaq_runner import DreamWaQRunner
from .on_policy_runner_cts import OnPolicyRunnerCTS


def _missing_pybullet_runner(class_name):
    """Create a placeholder that reports the optional AMP dependency clearly."""

    class MissingPyBulletRunner:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                f"{class_name} requires the optional 'pybullet' package. "
                "Install it with: python -m pip install pybullet"
            )

    MissingPyBulletRunner.__name__ = class_name
    return MissingPyBulletRunner


try:
    from .dreamwaq_runner_amp import DreamWaQRunner_AMP
    from .on_policy_runner_cts_amp import OnPolicyRunnerCTSAMP
    from .on_policy_runner_amp_ts import OnPolicyRunnerAMP_TS
    from .distill_policy_runner import DistillPolicyRunner
except ModuleNotFoundError as exc:
    # Standard PPO/DreamWaQ/CTS tasks do not use AMP motion data and should not
    # require pybullet merely because all runner names are registered together.
    if exc.name != "pybullet_utils":
        raise
    DreamWaQRunner_AMP = _missing_pybullet_runner("DreamWaQRunner_AMP")
    OnPolicyRunnerCTSAMP = _missing_pybullet_runner("OnPolicyRunnerCTSAMP")
    OnPolicyRunnerAMP_TS = _missing_pybullet_runner("OnPolicyRunnerAMP_TS")
    DistillPolicyRunner = _missing_pybullet_runner("DistillPolicyRunner")
