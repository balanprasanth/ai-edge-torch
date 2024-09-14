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

"""Verifies the reauthored Phi-2 model."""

from absl import app
from ai_edge_torch.generative.examples.phi import phi2
from ai_edge_torch.generative.utilities import verifier
import kagglehub


def main(_):
  path = kagglehub.model_download("Microsoft/phi/transformers/2")
  verifier.verify_reauthored_transformers(
      checkpoint=path,
      reauthored_model_builer=phi2.build_model,
      prompts="What is the meaning of life?",
      atol=1e-03,
  )


if __name__ == "__main__":
  app.run(main)
