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

"""Common utility functions for model conversion."""

import datetime
import os
import pathlib
from typing import Any, Callable

from ai_edge_torch.generative.layers import kv_cache as kv_utils
import numpy as np
import torch
import transformers

# Default prompts to verify the reauthored models.
DEFAULT_PROMPTS_FOR_VERIFICATION = "What is the meaning of life?"


def _log_msg(*args):
  print("[%s]" % datetime.datetime.now(), *args)


def forward(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    kv_cache: kv_utils.KVCache,
) -> tuple[torch.Tensor, kv_utils.KVCache]:
  """Forwards the model reauthored with ai_edge_torch Generative API.

  Args:
    model (torch.nn.Module): The model to forward. It should be a model built
      with ai_edge_torch Generative API.
    tokens (torch.Tensor): The input tokens to forward.
    kv_cache (KVCache): The KV cache to forward.

  Returns:
    The output logits and the updated KV cache.
  """
  input_pos = torch.arange(0, tokens.shape[1], dtype=torch.int)
  output = model.forward(tokens, input_pos, kv_cache)
  return output["logits"], output["kv_cache"]


def generate(
    model: torch.nn.Module, prompts: torch.Tensor, response_len: int
) -> torch.Tensor:
  """Generates the response to the prompts.

  It appends tokens output by the model to the prompts and feeds them back to
  the model up to decode_len.

  Args:
    model (torch.nn.Module): The model to generate. It should be a model built
      with ai_edge_torch Generative API.
    prompts (torch.Tensor): The prompts to generate.
    response_len (int): The number of tokens to generate.

  Returns:
    The generated tokens.
  """
  input_ids = prompts[0].tolist()
  kv_cache = kv_utils.KVCache.from_model_config(model.config)
  for _ in range(response_len - len(input_ids)):
    logits, kv_cache = forward(model, torch.tensor([input_ids]), kv_cache)
    generated_token = logits[0][-1].argmax().item()
    input_ids.append(generated_token)
  return torch.tensor([input_ids])


def verify_with_input_ids(
    original_model: torch.nn.Module,
    reauthored_model: torch.nn.Module,
    input_ids: torch.Tensor = torch.from_numpy(np.array([[1, 2, 3, 4]])),
    kv_cache_max_len: int = 1024,
    atol: float = 1e-05,
) -> bool:
  """Verifies if the model reauthored generates the same output of the oringal.

  It compares only one outputs from the original and the reauthored model.

  Args:
    original_model (torch.nn.Module): The original model.
    reauthored_model (torch.nn.Module): The model reauthored with ai_edge_torch
      Generative API.
    input_ids (torch.Tensor): The input token IDs to forward.
    kv_cache_max_len (int): The maximum sequence length of the KV cache.
    atol (float): The absolute tolerance for the comparison.

  Returns:
    True if the model reauthored generates the same output of the original.
  """
  tokens = torch.full((1, kv_cache_max_len), 0, dtype=torch.long, device="cpu")
  input_ids_len = input_ids.shape[1]
  tokens[0, :input_ids_len] = input_ids

  _log_msg("Forwarding the original model...")
  outputs_original = original_model.forward(tokens)
  logits_original = outputs_original.logits[0, input_ids_len - 1, :]
  _log_msg("logits_original: ", logits_original)

  _log_msg("Forwarding the reauthored model...")
  kv_cache = kv_utils.KVCache.from_model_config(reauthored_model.config)
  outputs_reauthored = forward(reauthored_model, tokens, kv_cache)
  logits_reauthored = outputs_reauthored[0][0, input_ids_len - 1, :]
  _log_msg("logits_reauthored:", logits_reauthored)

  return torch.allclose(logits_original, logits_reauthored, atol=atol)


def verify_model_with_prompts(
    original_model: torch.nn.Module,
    reauthored_model: torch.nn.Module,
    tokenizer: torch.nn.Module,
    prompts: str,
) -> bool:
  """Verifies if the model reauthored generates the same answer of the oringal.

  It compares an answer, i.e. multiple continuous outputs generated by the
  original and the reauthored model.

  Args:
    original_model (torch.nn.Module): The original model.
    reauthored_model (torch.nn.Module): The model reauthored with ai_edge_torch
      Generative API.
    tokenizer (torch.nn.Module): The tokenizer.
    prompts (str): The input prompts to generate answers.

  Returns:
    True if the model reauthored generates the same answer of the original.
  """
  prompt_tokens = tokenizer.encode(prompts, return_tensors="pt")

  _log_msg("Generating answer with the original model...")
  outputs_original = original_model.generate(prompt_tokens)
  response_original = tokenizer.decode(outputs_original[0])
  _log_msg("outputs_from_original_model: [[", response_original, "]]")

  _log_msg("Generating answer with the reauthored model...")
  generate_len = len(outputs_original[0])
  outputs_reauthored = generate(reauthored_model, prompt_tokens, generate_len)
  response_reauthored = tokenizer.decode(outputs_reauthored[0])
  _log_msg("outputs from reauthored model: [[", response_reauthored, "]]")

  return response_original == response_reauthored


def verify_reauthored_transformers(
    checkpoint: str,
    reauthored_model_builer: Callable[[str, ..., Any], torch.nn.Module],
    prompts: str = DEFAULT_PROMPTS_FOR_VERIFICATION,
    atol: float = 1e-05,
    tokenizer_checkpoint: str = None,
    **kwargs_for_original_model,
):
  """Verifies the reauthored model from one built with transformers package.

  It verifies the reauthored model with two methods:
  1. It compares the output of the original and the reauthored model with an
     arbitrary input.
  2. It compares the answer generated by the original and the reauthored model
     with a prompt.

  It prints out "PASS" or "FAILED" to the console.

  Args:
    checkpoint (str): The checkpoint of the original model and the reauthored
      model.
    reauthored_model_builer (Callable): The builder of the reauthored model.
    prompts (str): The input prompts to generate answers.
    atol (float): The absolute tolerance for the comparison.
    tokenizer_checkpoint (str): The checkpoint of the tokenizer. If None, it is
      set to checkpoint.
    **kwargs_for_original_model (dict): The keyword arguments used to load the
      original model and tokenizer.
  """
  _log_msg("Loading checkpoint from", checkpoint)
  original_model = transformers.AutoModelForCausalLM.from_pretrained(
      checkpoint, **kwargs_for_original_model
  )

  # Locate the cached dir.
  if os.path.exists(checkpoint):
    reauthored_checkpoint = checkpoint
  else:
    cached_config_file = transformers.utils.cached_file(
        checkpoint, transformers.utils.CONFIG_NAME
    )
    _log_msg("cached_config_file: ", cached_config_file)
    reauthored_checkpoint = pathlib.Path(cached_config_file).parent

  _log_msg("Instantiating the reauthored model from", reauthored_checkpoint)
  kv_cache_max_len = 1024
  reauthored_model = reauthored_model_builer(
      reauthored_checkpoint, kv_cache_max_len=kv_cache_max_len
  )

  _log_msg("Verifying the reauthored model with an arbitrary input...")
  if verify_with_input_ids(original_model, reauthored_model, atol=atol):
    _log_msg("PASS")
  else:
    _log_msg("FAILED")

  if tokenizer_checkpoint is None:
    tokenizer_checkpoint = checkpoint
  _log_msg("Loading tokenizer from", tokenizer_checkpoint)
  tokenizer = transformers.AutoTokenizer.from_pretrained(
      tokenizer_checkpoint,
      **kwargs_for_original_model,
  )
  _log_msg("Verifying the reauthored model with prompts:", prompts)
  if verify_model_with_prompts(
      original_model, reauthored_model, tokenizer, prompts
  ):
    _log_msg("PASS")
  else:
    _log_msg("FAILED")
