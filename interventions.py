from dataclasses import dataclass
from typing import Any, Optional, Union

import torch as t
from einops import rearrange
from jaxtyping import Float, Int
from loguru import logger
from nnsight import LanguageModel
from nnsight.contexts import Runner
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


@dataclass
class MLPLayerCache:
    clean_mlp_output: t.Tensor
    corrupted_mlp_output: t.Tensor
    clean_grad_mlp_output: t.Tensor
    grad_drop_mlp_output: Optional[t.Tensor] = None


def get_atp_caches(
    model_name: str,
    clean_tokens: Int[t.Tensor, "examples"],
    corrupted_tokens: Int[t.Tensor, "examples"],
    off_distribution_tokens: Int[t.Tensor, "examples"],
    answer_token_indices: Int[t.Tensor, "examples 2"],
) -> tuple[list[AttentionLayerCache], list[MLPLayerCache]]:
    """Run the model forward passes on the clean and corrupted tokens and cache the activations.
    We also run a backward pass to cache the gradients on the clean tokens.

    Parameters
    ----------
    model : LanguageModel
    clean_tokens : Int[t.Tensor, "examples"]
    corrupted_tokens : Int[t.Tensor, "examples"]
    off_distribution_tokens : Int[t.Tensor, "examples"]
    answer_token_indices : Int[t.Tensor, "examples, 2"]

    Returns
    -------
    attn_cache : list[AttentionLayerCache]
    mlp_cache : list[MLPLayerCache]

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
            clean_attn_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].save()
                for i in range(num_layers)  # type: ignore
                # batch head_index seq_len seq_len
            ]

            clean_attn_grad_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].grad.save()
                for i in range(num_layers)  # type: ignore
            ]

            clean_mlp_cache = [
                model.transformer.h[i].mlp.c_proj.output.save()
                for i in range(num_layers)  # type: ignore
            ]

            clean_mlp_grad_cache = [
                model.transformer.h[i].mlp.c_proj.output.grad.save()
                for i in range(num_layers)  # type: ignore
            ]

        with tracer.invoke(corrupted_tokens) as corrupted_invoker:

            # Calculate L(M(x_corrupted))
            corrupted_logits: Float[t.Tensor, "examples seq_len vocab"] = model.lm_head.output  # type: ignore
            corrupted_logit_diff: Float[t.Tensor, "1"] = mean_logit_diff(
                corrupted_logits, answer_token_indices
            ).save()  # type: ignore

            # Cache the corrupted activations and gradients for all the nodes
            corrupted_attn_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].save()
                for i in range(num_layers)  # type: ignore
            ]

            corrupted_mlp_cache = [
                model.transformer.h[i].mlp.c_proj.output.save()
                for i in range(num_layers)  # type: ignore
            ]

            flat_corrupted_queries = [
                model.transformer.h[i].attn.c_attn.output.split(attn.split_size, dim=2)[0]
                for i in range(num_layers)
            ]  # type: ignore
            flat_corrupted_keys = [
                model.transformer.h[i].attn.c_attn.output.split(attn.split_size, dim=2)[1]
                for i in range(num_layers)
            ]  # type: ignore

        q_corrupted_cache = get_q_corrupted_cache(
            clean_tokens, model, num_layers, attn, tracer, flat_corrupted_queries
        )

        k_corrupted_cache = get_k_corrupted_cache(
            clean_tokens, model, num_layers, attn, tracer, flat_corrupted_keys
        )

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

        # GradDrop Attention
        grad_drop_attn_collection = cache_grad_drop_attention(
            clean_tokens,
            answer_token_indices,
            model,
            num_layers,
            tracer,
            corrupted_logit_diff,
            off_distribution_logit_diff,
        )

        # GradDrop MLP
        grad_drop_mlp_collection = cache_grad_drop_mlp(
            clean_tokens,
            answer_token_indices,
            model,
            num_layers,
            tracer,
            corrupted_logit_diff,
            off_distribution_logit_diff,
        )

    grad_drop_attn_collection = [
        [inner_value.value for inner_value in value] for value in grad_drop_attn_collection
    ]

    # Convert list of list of tensors into single tensor
    _tensor_grad_drop_attn_collection = [
        t.stack(layer, dim=0) for layer in grad_drop_attn_collection  # type: ignore
    ]  # dropped_layer list[layer examples head seq_len seq_len]
    tensor_grad_drop_attn_cache = t.stack(
        _tensor_grad_drop_attn_collection, dim=0
    )  # dropped_layer layer examples head seq_len seq_len

    tensor_grad_drop_attn_cache = rearrange(
        tensor_grad_drop_attn_cache,
        "dropped_layer layer examples head seq_len_dest seq_len_source -> layer dropped_layer examples head seq_len_dest seq_len_source",
    )

    grad_drop_attn_cache = [
        tensor_grad_drop_attn_cache[i] for i in range(num_layers)  # type: ignore
    ]  # layer list[dropped_layer examples head seq_len seq_len]

    ###

    grad_drop_mlp_collection = [
        [inner_value.value for inner_value in value] for value in grad_drop_mlp_collection
    ]

    # Convert list of list of tensors into single tensor
    _tensor_grad_drop_mlp_collection = [
        t.stack(layer, dim=0) for layer in grad_drop_mlp_collection  # type: ignore
    ]  # dropped_layer list[layer examples seq_len]
    tensor_grad_drop_mlp_cache = t.stack(
        _tensor_grad_drop_mlp_collection, dim=0
    )  # dropped_layer layer examples seq_len

    tensor_grad_drop_mlp_cache = rearrange(
        tensor_grad_drop_mlp_cache,
        "dropped_layer layer examples seq_pos -> layer dropped_layer examples seq_pos",
    )

    grad_drop_mlp_cache = [
        tensor_grad_drop_mlp_cache[i] for i in range(num_layers)  # type: ignore
    ]  # layer list[dropped_layer examples seq_pos]

    ###

    logger.debug(clean_logit_diff)
    logger.debug(corrupted_logit_diff)
    logger.debug(off_distribution_logit_diff)

    def hydrate_intervention_proxy_list(
        proxy_list: Union[list[t.Tensor], list[InterventionProxy]]
    ) -> list[t.Tensor]:
        if type(proxy_list[0]) == t.Tensor:
            return proxy_list  # type: ignore
        return [value.value for value in proxy_list]  # type: ignore

    clean_attn_cache = hydrate_intervention_proxy_list(clean_attn_cache)
    clean_attn_grad_cache = hydrate_intervention_proxy_list(clean_attn_grad_cache)
    corrupted_attn_cache = hydrate_intervention_proxy_list(corrupted_attn_cache)

    q_corrupted_cache = hydrate_intervention_proxy_list(q_corrupted_cache)
    k_corrupted_cache = hydrate_intervention_proxy_list(k_corrupted_cache)

    clean_mlp_cache = hydrate_intervention_proxy_list(clean_mlp_cache)
    clean_mlp_grad_cache = hydrate_intervention_proxy_list(clean_mlp_grad_cache)
    corrupted_mlp_cache = hydrate_intervention_proxy_list(corrupted_mlp_cache)

    # grad_drop_cache = [value.value for value in grad_drop_cache]

    attn_cache = [
        AttentionLayerCache(
            clean_attn_probs=clean_attn_cache[i],
            query_corrupted_attn_probs=q_corrupted_cache[i],
            key_corrupted_attn_probs=k_corrupted_cache[i],
            fully_corrupted_attn_probs=corrupted_attn_cache[i],
            clean_grad_attn_probabilities=clean_attn_grad_cache[i],
            grad_drop_attn_probs=grad_drop_attn_cache[i] if grad_drop_attn_cache else None,
        )
        for i in range(len(clean_attn_cache))
    ]

    mlp_cache = [
        MLPLayerCache(
            clean_mlp_output=clean_mlp_cache[i],
            corrupted_mlp_output=corrupted_mlp_cache[i],
            clean_grad_mlp_output=clean_mlp_grad_cache[i],
            grad_drop_mlp_output=grad_drop_mlp_cache[i] if grad_drop_mlp_cache else None,
        )
        for i in range(len(clean_mlp_cache))
    ]

    return attn_cache, mlp_cache


def get_q_corrupted_cache(
    clean_tokens: Int[t.Tensor, "batch seq"],
    model: LanguageModel,
    num_layers: int,
    attn: GPT2Attention,
    tracer: Union[Runner, Any],
    flat_corrupted_queries: list[Union[t.Tensor, InterventionProxy]],
) -> list[InterventionProxy]:
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

    return q_corrupted_cache


def get_k_corrupted_cache(
    clean_tokens: Int[t.Tensor, "batch seq"],
    model: LanguageModel,
    num_layers: int,
    attn: GPT2Attention,
    tracer: Union[Runner, Any],
    flat_corrupted_keys: list[Union[t.Tensor, InterventionProxy]],
) -> list[InterventionProxy]:
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

    return k_corrupted_cache


def cache_grad_drop_attention(
    clean_tokens: Int[t.Tensor, "batch seq"],
    answer_token_indices: Int[t.Tensor, "batch 2"],
    model: LanguageModel,
    num_layers: int,
    tracer: Union[Runner, Any],
    corrupted_logit_diff: Float[t.Tensor, "1"],
    off_distribution_logit_diff: Float[t.Tensor, "1"],
) -> list[list[InterventionProxy]]:
    grad_drop_attn_collection: list[list[InterventionProxy]] = []
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
            layer_dropped_attn_grad_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].grad.save()
                for i in range(num_layers)  # type: ignore
            ]  # layer list[examples head seq_len seq_len]

            layer_dropped_ioi_score = ioi_metric(
                layer_dropped_logit_diff, corrupted_logit_diff, off_distribution_logit_diff
            )  # scalar
            layer_dropped_ioi_score.backward(retain_graph=True)

        grad_drop_attn_collection.append(
            layer_dropped_attn_grad_cache
        )  # dropped_layer list[layer list[examples head seq_len seq_len]]]

    return grad_drop_attn_collection


def cache_grad_drop_mlp(
    clean_tokens: Int[t.Tensor, "batch seq"],
    answer_token_indices: Int[t.Tensor, "batch 2"],
    model: LanguageModel,
    num_layers: int,
    tracer: Union[Runner, Any],
    corrupted_logit_diff: Float[t.Tensor, "1"],
    off_distribution_logit_diff: Float[t.Tensor, "1"],
) -> list[list[InterventionProxy]]:
    grad_drop_mlp_collection: list[list[InterventionProxy]] = []
    for l in range(num_layers):
        with tracer.invoke(clean_tokens) as grad_drop_invoker:
            # Zero out the gradients on each layer from the residual connection
            model.transformer.h[l].mlp.c_proj.output.grad.zero_()

            layer_dropped_logits: Float[t.Tensor, "examples seq_len vocab"] = (
                model.lm_head.output
            )  # type: ignore # M(x_clean)
            layer_dropped_logit_diff: Float[t.Tensor, "1"] = mean_logit_diff(
                layer_dropped_logits, answer_token_indices
            ).save()  # type: ignore

            # Collect the gradients on the attention probabilities
            layer_dropped_mlp_grad_cache = [
                model.transformer.h[i].mlp.c_proj.output.grad.save()
                for i in range(num_layers)  # type: ignore
            ]  # layer list[examples seq_len]

            layer_dropped_ioi_score = ioi_metric(
                layer_dropped_logit_diff, corrupted_logit_diff, off_distribution_logit_diff
            )  # scalar
            layer_dropped_ioi_score.backward(retain_graph=True)

        grad_drop_mlp_collection.append(
            layer_dropped_mlp_grad_cache
        )  # dropped_layer list[layer list[examples seq_len]]]

    return grad_drop_mlp_collection
