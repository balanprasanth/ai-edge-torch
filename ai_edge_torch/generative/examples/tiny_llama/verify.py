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

"""Verifies the reauthored TinyLlama-1.1B model."""

from absl import app
from ai_edge_torch.generative.examples.tiny_llama import tiny_llama
from ai_edge_torch.generative.utilities import verifier


def main(_):
  verifier.verify_reauthored_transformers(
      checkpoint="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
      reauthored_model_builer=tiny_llama.build_model,
      prompts="Show me the program to add 2 and 3.",
      atol=1e-04,
  )


if __name__ == "__main__":
  app.run(main)
