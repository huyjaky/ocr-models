"""
2025.7.4
2025.7.3
4.52.4
0.19.1
__UNSLOTH_VERSIONING__
"""

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import importlib.util
if importlib.util.find_spec("unsloth_studio") is None:
    UNSLOTH_STUDIO_ENABLED = False
else:
    UNSLOTH_STUDIO_ENABLED = os.environ.get("UNSLOTH_STUDIO_DISABLED", "0") == "0"
pass
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
import math

UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"


torch_compile_options = {'epilogue_fusion': True, 'max_autotune': False, 'shape_padding': True, 'trace.enabled': False, 'triton.cudagraphs': False, 'debug': False, 'dce': True, 'memory_planning': True, 'coordinate_descent_tuning': False, 'trace.graph_diagram': False, 'combo_kernels': False, 'group_fusion': True, 'disable_progress': True, 'verbose_progress': False, 'triton.multi_kernel': False, 'triton.use_block_ptr': False, 'triton.enable_persistent_tma_matmul': True, 'triton.autotune_at_compile_time': True}
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from transformers.models.gemma3.modeling_gemma3 import (List, Optional, Tuple, nn)

def forward(self, input: Tensor, output_size: Optional[list[int]] = None) -> Tensor:
    if self.padding_mode != "zeros":
        raise ValueError(
            "Only `zeros` padding mode is supported for ConvTranspose1d"
        )

    assert isinstance(self.padding, tuple)
    # One cannot replace List by Tuple or Sequence in "_output_padding" because
    # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
    num_spatial_dims = 1
    output_padding = self._output_padding(
        input,
        output_size,
        self.stride,  # type: ignore[arg-type]
        self.padding,  # type: ignore[arg-type]
        self.kernel_size,  # type: ignore[arg-type]
        num_spatial_dims,
        self.dilation,  # type: ignore[arg-type]
    )
    return F.conv_transpose1d(
        input,
        self.weight,
        self.bias,
        self.stride,
        self.padding,
        output_padding,
        self.groups,
        self.dilation,
    ).to(input.dtype).to(input.dtype)
