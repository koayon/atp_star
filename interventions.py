from dataclasses import dataclass
from typing import Optional

import torch as t
from einops import rearrange
from jaxtyping import Float, Int
from loguru import logger
from nnsight import LanguageModel
from nnsight.intervention import InterventionProxy
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from helpers import get_num_layers_heads, ioi_metric, mean_logit_diff


@dataclass
class AttentionLayerCache:
    clean_attn_probs: t.Tensor
    query_corrupted_attn_probs: t.Tensor
    key_corrupted_attn_probs: t.Tensor
    fully_corrupted_attn_probs: t.Tensor
    clean_grad_attn_probabilities: t.Tensor
    grad_drop_attn_probs: Optional[t.Tensor] = None


def get_atp_caches(
    model_name: str,
    clean_tokens: Int[t.Tensor, "examples"],
    corrupted_tokens: Int[t.Tensor, "examples"],
    off_distribution_tokens: Int[t.Tensor, "examples"],
    answer_token_indices: Int[t.Tensor, "examples 2"],
) -> list[AttentionLayerCache]:
    """Run the model forward passes on the clean and corrupted tokens and cache the activations.
    We also run a backward pass to cache the gradients on the clean tokens.

    Parameters
    ----------
    model : LanguageModel
    clean_tokens : Int[t.Tensor, "examples"]
    corrupted_tokens : Int[t.Tensor, "examples"]
    answer_token_indices : Int[t.Tensor, "examples, 2"]

    Returns
    -------
    clean_cache : list[t.Tensor]
    corrupted_cache : list[t.Tensor]
    clean_grad_cache : list[t.Tensor]

    """
    model = LanguageModel(model_name, device_map="cpu", dispatch=True)

    num_layers, num_heads, _ = get_num_layers_heads(model)
    attn: GPT2Attention = model.transformer.h[0].attn  # type: ignore
    head_dim = int(attn.head_dim)

    with model.trace() as tracer:
        with tracer.invoke(clean_tokens) as clean_invoker:

            # Calculate L(M(x_clean))
            clean_logits: Float[t.Tensor, "examples seq_len vocab"] = (
                model.lm_head.output.save()
            )  # type: ignore # M(x_clean)
            clean_logit_diff: Float[t.Tensor, "1"] = mean_logit_diff(
                clean_logits, answer_token_indices
            ).save()  # type: ignore

            # Cache the clean activations and gradients for all the nodes
            clean_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].save()
                for i in range(num_layers)  # type: ignore
                # batch head_index seq_len seq_len
            ]

            clean_grad_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].grad.save()
                for i in range(num_layers)  # type: ignore
            ]

        with tracer.invoke(corrupted_tokens) as corrupted_invoker:

            # Calculate L(M(x_corrupted))
            corrupted_logits: Float[t.Tensor, "examples seq_len vocab"] = model.lm_head.output  # type: ignore
            corrupted_logit_diff: Float[t.Tensor, "1"] = mean_logit_diff(
                corrupted_logits, answer_token_indices
            ).save()  # type: ignore

            # Cache the corrupted activations and gradients for all the nodes
            corrupted_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].save()
                for i in range(len(model.transformer.h))  # type: ignore
            ]

            flat_corrupted_queries = [
                model.transformer.h[i].attn.c_attn.output.split(attn.split_size, dim=2)[0]
                for i in range(num_layers)
            ]  # type: ignore
            flat_corrupted_keys = [
                model.transformer.h[i].attn.c_attn.output.split(attn.split_size, dim=2)[1]
                for i in range(num_layers)
            ]  # type: ignore

        q_corrupted_cache: list = []
        for i in range(num_layers):
            with tracer.invoke(clean_tokens) as q_patch_invoker:
                # Patch the queries with the corrupted queries
                qkv = model.transformer.h[i].attn.c_attn.output
                _, k, v = qkv.split(attn.split_size, dim=2)
                new_qkv = t.cat([flat_corrupted_queries[i], k, v], dim=2)  #  type: ignore
                model.transformer.h[i].attn.c_attn.output = new_qkv

                # Compute and cache the attention probabilities with the patched queries
                q_corrupted_cache.append(
                    model.transformer.h[i].attn.attn_dropout.input[0][0].save()  # type: ignore
                )

        k_corrupted_cache: list = []
        # TODO: This is currently the O(n^3) implementation, following Algorithm 4 we can reduce this to O(n^2)

        for i in range(num_layers):
            # TODO: Note here (and in the q version) we're currently computing the whole forward pass
            # but we actually only need to compute the forward until the attention probabilities
            # for the increased speed
            with tracer.invoke(clean_tokens) as k_patch_invoker:
                # Patch the keys with the corrupted keys
                qkv = model.transformer.h[i].attn.c_attn.output
                q, _, v = qkv.split(attn.split_size, dim=2)
                new_qkv = t.cat([q, flat_corrupted_keys[i], v], dim=2)  #  type: ignore
                model.transformer.h[i].attn.c_attn.output = new_qkv

                # Compute and cache the attention probabilities with the patched keys
                k_corrupted_cache.append(
                    model.transformer.h[i].attn.attn_dropout.input[0][0].save()  # type: ignore
                )

                # TODO: Patch back in the clean attention probs

        with tracer.invoke(off_distribution_tokens) as off_distribution_invoker:

            # Calculate L(M(x_corrupted))
            off_distribution_logits: Float[t.Tensor, "examples seq_len vocab"] = model.lm_head.output.save()  # type: ignore
            off_distribution_logit_diff: Float[t.Tensor, "1"] = mean_logit_diff(
                off_distribution_logits, answer_token_indices
            ).save()  # type: ignore

            ioi_score = ioi_metric(
                clean_logit_diff, corrupted_logit_diff, off_distribution_logit_diff
            )  # scalar

            ioi_score.backward(retain_graph=True)

        # with model.trace() as tracer:
        # GradDrop
        grad_drop_collection: list[list[InterventionProxy]] = []
        for l in range(num_layers):
            with tracer.invoke(clean_tokens) as grad_drop_invoker:
                # Zero out the gradients on each layer from the residual connection
                model.transformer.h[l].attn.resid_dropout.input[0][0].grad.zero_()

                layer_dropped_logits: Float[t.Tensor, "examples seq_len vocab"] = (
                    model.lm_head.output
                )  # type: ignore # M(x_clean)
                layer_dropped_logit_diff: Float[t.Tensor, "1"] = mean_logit_diff(
                    layer_dropped_logits, answer_token_indices
                ).save()  # type: ignore

                # Collect the gradients on the attention probabilities
                layer_dropped_grad_cache = [
                    model.transformer.h[i].attn.attn_dropout.input[0][0].grad.save()
                    for i in range(len(model.transformer.h))  # type: ignore
                ]  # layer list[examples head seq_len seq_len]

                layer_dropped_ioi_score = ioi_metric(
                    layer_dropped_logit_diff, corrupted_logit_diff, off_distribution_logit_diff
                )  # scalar
                layer_dropped_ioi_score.backward(retain_graph=True)

            grad_drop_collection.append(
                layer_dropped_grad_cache
            )  # dropped_layer list[layer list[examples head seq_len seq_len]]]

    logger.debug(type(grad_drop_collection))
    logger.debug(type(grad_drop_collection[0]))
    logger.debug(type(grad_drop_collection[0][0]))

    grad_drop_collection = [
        [inner_value.value for inner_value in value] for value in grad_drop_collection
    ]

    logger.debug(type(grad_drop_collection))
    logger.debug(type(grad_drop_collection[0]))
    logger.debug(type(grad_drop_collection[0][0]))

    # Convert list of list of tensors into single tensor
    _tensor_grad_drop_collection = [
        t.stack(layer, dim=0) for layer in grad_drop_collection  # type: ignore
    ]  # dropped_layer list[layer examples head seq_len seq_len]
    tensor_grad_drop_cache = t.stack(
        _tensor_grad_drop_collection, dim=0
    )  # dropped_layer layer examples head seq_len seq_len

    tensor_grad_drop_cache = rearrange(
        tensor_grad_drop_cache,
        "dropped_layer layer examples head seq_len_dest seq_len_source -> layer dropped_layer examples head seq_len_dest seq_len_source",
    )

    grad_drop_cache = [
        tensor_grad_drop_cache[i] for i in range(num_layers)  # type: ignore
    ]  # layer list[dropped_layer examples head seq_len seq_len]

    ###

    logger.debug(clean_logit_diff)
    logger.debug(corrupted_logit_diff)
    logger.debug(off_distribution_logit_diff)

    clean_cache = [value.value for value in clean_cache]
    clean_grad_cache = [value.value for value in clean_grad_cache]
    corrupted_cache = [value.value for value in corrupted_cache]

    q_corrupted_cache = [value.value for value in q_corrupted_cache]
    k_corrupted_cache = [value.value for value in k_corrupted_cache]

    # grad_drop_cache = [value.value for value in grad_drop_cache]

    attn_cache = [
        AttentionLayerCache(
            clean_attn_probs=clean_cache[i],
            query_corrupted_attn_probs=q_corrupted_cache[i],
            key_corrupted_attn_probs=k_corrupted_cache[i],
            fully_corrupted_attn_probs=corrupted_cache[i],
            clean_grad_attn_probabilities=clean_grad_cache[i],
            grad_drop_attn_probs=grad_drop_cache[i] if grad_drop_cache else None,
        )
        for i in range(len(clean_cache))
    ]

    return attn_cache
