# Copyright 2024 The AI Edge Torch Authors.
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
# ==============================================================================

"""Example of converting a Phi-2 model to multi-signature tflite model."""

import os
import pathlib

from ai_edge_torch.generative.examples.phi import phi2
from ai_edge_torch.generative.utilities import converter


def convert_phi2_to_tflite(
    checkpoint_path: str,
    tflite_path_prefix: str = '/tmp/phi2',
    prefill_seq_len: int = 512,
    kv_cache_max_len: int = 1024,
    quantize: bool = True,
):
  """Converts a Phi-2 model to multi-signature tflite model.

  Args:
      checkpoint_path (str): The filepath to the model checkpoint, or directory
        holding the checkpoint.
      tflite_path_prefix (str): The prefix of the tflite file path to export.
      prefill_seq_len (int, optional): The maximum size of prefill input tensor.
        Defaults to 512.
      kv_cache_max_len (int, optional): The maximum size of KV cache buffer,
        including both prefill and decode. Defaults to 1024.
      quantize (bool, optional): Whether the model should be quanized. Defaults
        to True.
  """
  pytorch_model = phi2.build_model(
      checkpoint_path, kv_cache_max_len=kv_cache_max_len
  )
  converter.convert_to_tflite(
      pytorch_model,
      tflite_path_prefix=tflite_path_prefix,
      prefill_seq_len=prefill_seq_len,
      kv_cache_max_len=kv_cache_max_len,
      quantize=quantize,
  )


if __name__ == '__main__':
  path = os.path.join(pathlib.Path.home(), 'Downloads/llm_data/phi2')
  convert_phi2_to_tflite(path)
