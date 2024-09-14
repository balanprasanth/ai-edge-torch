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

"""Verifies the reauthored OpenELM-3B model."""

from absl import app
from ai_edge_torch.generative.examples.openelm import openelm
from ai_edge_torch.generative.utilities import verifier


def main(_):
  verifier.verify_reauthored_transformers(
      checkpoint="apple/OpenELM-3B",
      tokenizer_checkpoint="meta-llama/Llama-2-7b-hf",
      reauthored_model_builer=openelm.build_model,
      prompts="What is the meaning of life?",
      trust_remote_code=True,
  )


if __name__ == "__main__":
  app.run(main)
