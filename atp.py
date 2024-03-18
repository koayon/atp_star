import sys
from dataclasses import dataclass
from typing import Optional

import torch as t
from einops import einsum, rearrange
from jaxtyping import Float, Int
from loguru import logger
from nnsight import LanguageModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from helpers import get_num_layers_heads, ioi_metric, mean_logit_diff
from plot import plot_attention_attributions, plot_single_attention_pattern
from prompt_store import build_prompt_store


@dataclass
class AttentionLayerCache:
    clean_attn_probs: t.Tensor
    query_corrupted_attn_probs: t.Tensor
    key_corrupted_attn_probs: t.Tensor
    fully_corrupted_attn_probs: t.Tensor
    clean_grad_attn_probabilities: t.Tensor
    grad_drop_attn_probs: Optional[t.Tensor] = None


def atp_component_contribution(
    attn_layer_cache: AttentionLayerCache,
    use_grad_drop: bool = False,
) -> Float[t.Tensor, "head seq_len"]:
    """Calculates the intervention effect of a node in the transformer (here attention)."""
    # Get the activations and gradients for the node
    node_clean_activation: Float[t.Tensor, "examples head seq_len seq_len"] = (
        attn_layer_cache.clean_attn_probs
    )
    node_corrupted_activation: Float[t.Tensor, "examples head seq_len seq_len"] = (
        attn_layer_cache.fully_corrupted_attn_probs
    )
    grad_wrt_node: Float[t.Tensor, "examples head seq_len seq_len"] = (
        attn_layer_cache.clean_grad_attn_probabilities
    )

    # Calculate the intervention effect I_{AtP}(n; x_clean, x_corrupted)
    activation_diff = node_corrupted_activation - node_clean_activation

    logger.debug(activation_diff.max())

    if use_grad_drop and attn_layer_cache.grad_drop_attn_probs is not None:
        grad_drop_wrt_node: Float[t.Tensor, "dropped_layer examples head seq_len seq_len"] = (
            attn_layer_cache.grad_drop_attn_probs
        )

        # Equation 11
        # Calculate the contribution
        # c_{AtP+GD}(n) = ExpectedValue(1/(L-1) * |I_{AtP+GD}_l(n; x_clean, x_corrupted)|)
        intervention_effects = einsum(
            activation_diff,
            grad_drop_wrt_node,
            "batch head seq_len1 seq_len2, dropped_layer batch head seq_len1 seq_len2 -> dropped_layer batch head seq_len1",
        )

        individual_contributions = t.sum(intervention_effects.abs(), dim=0) / (
            len(grad_drop_wrt_node) - 1
        )  # batch head seq_len
        contribution = t.mean(individual_contributions, dim=0)  # head seq_len

    elif use_grad_drop:
        raise ValueError("grad_drop_attn_probs not provided")

    else:
        # Equation 4
        intervention_effect = einsum(
            activation_diff,
            grad_wrt_node,
            "batch head seq_len1 seq_len2, batch head seq_len1 seq_len2 -> batch head seq_len1",
        )

        # Equation 5
        # Calculate the contribution c_{AtP}(n) = ExpectedValue(|I_{AtP}(n; x_clean, x_corrupted)|)
        contribution = t.mean(intervention_effect.abs(), dim=0)  # head seq_len
    return contribution


def atp_q_contribution(
    attn_layer_cache: AttentionLayerCache,
) -> Float[t.Tensor, "head seq_len"]:
    # Get the activations and gradients for the node
    node_clean_activation: Float[t.Tensor, "examples head seq_len seq_len"] = (
        attn_layer_cache.clean_attn_probs
    )
    node_q_corrupted_activation: Float[t.Tensor, "examples head seq_len seq_len"] = (
        attn_layer_cache.query_corrupted_attn_probs
    )
    grad_wrt_node: Float[t.Tensor, "examples head seq_len seq_len"] = (
        attn_layer_cache.clean_grad_attn_probabilities
    )

    # Calculate the intervention effect I_{AtP}(n; x_clean, x_corrupted)
    activation_diff = node_q_corrupted_activation - node_clean_activation

    logger.debug(activation_diff.max())

    # Equation 7
    intervention_effect = einsum(
        activation_diff,
        grad_wrt_node,
        "batch head seq_len1 seq_len2, batch head seq_len1 seq_len2 -> batch head seq_len1",
    )

    # Calculate the contribution c_{AtP}(n) = ExpectedValue(|I_{AtP}(n; x_clean, x_corrupted)|)
    contribution = t.mean(intervention_effect.abs(), dim=0)  # head seq_len seq_len
    return contribution


def atp_k_contribution(
    attn_layer_cache: AttentionLayerCache,
) -> Float[t.Tensor, "head seq_len"]:
    # Get the activations and gradients for the node
    node_clean_activation: Float[t.Tensor, "examples head seq_len seq_len"] = (
        attn_layer_cache.clean_attn_probs
    )
    node_k_corrupted_activation: Float[t.Tensor, "examples head seq_len seq_len"] = (
        attn_layer_cache.query_corrupted_attn_probs
    )
    grad_wrt_node: Float[t.Tensor, "examples head seq_len seq_len"] = (
        attn_layer_cache.clean_grad_attn_probabilities
    )

    # Equation 9
    # Calculate the intervention effect I_{AtP}(n; x_clean, x_corrupted)
    activation_diff = node_k_corrupted_activation - node_clean_activation

    logger.debug(activation_diff.max())

    position_intervention_effect = einsum(
        activation_diff,
        grad_wrt_node,
        "batch head seq_len1 seq_len2, batch head seq_len1 seq_len2 -> batch head seq_len1",
    )

    # Sum in Equation 10 is over n^q in queries(n_t^k).
    # The final query can see all the keys, the first query can only see the first key etc.
    _, _, seq_len = position_intervention_effect.shape

    lower_tri_mask = t.tril(t.ones(seq_len, seq_len))

    # Equation 10
    intervention_effect = einsum(
        position_intervention_effect,
        lower_tri_mask,
        "batch head seq_len_q, seq_len_q seq_len_k -> batch head seq_len_k",
    )

    # Calculate the contribution c_{AtP}(n) = ExpectedValue(|I_{AtP}(n; x_clean, x_corrupted)|)
    contribution = t.mean(intervention_effect.abs(), dim=0)  # head seq_len
    return contribution


def run_atp(
    model: LanguageModel,
    clean_tokens: Int[t.Tensor, "examples"],
    corrupted_tokens: Int[t.Tensor, "examples"],
    off_distribution_tokens: Int[t.Tensor, "examples"],
    answer_token_indices: Int[t.Tensor, "examples 2"],
) -> list[t.Tensor]:
    """Run the ATP algorithm (Nanda 2022).
    Optionally specify improvements to the algorithm (known as AtP*)
    which come from [Kramar et al 2024](https://arxiv.org/pdf/2403.00745.pdf).

    Attribution Patching (AtP) is introduced as a quick approximation to the more
    precise _Activation Patching_ (AcP) which details the contribution of each component
    to some metric (e.g. NLL loss, IOI score, etc.). It works by taking the first order
    Taylor approximation of the contribution c(n).

    Activation Patching is defined as the absolute value of the expected impact on the
    metric of resampling the node n with the corrupted (or noise) distribution.

    Parameters
    ----------
    model : LanguageModel
    clean_tokens : Int[t.Tensor, "examples"]
    corrupted_tokens : Int[t.Tensor, "examples"]
    answer_token_indices : Int[t.Tensor, "examples, 2"]

    Returns
    -------
    atp_component_contributions : list[Tensor]
        The approximate contribution of each component to the metric, as given by the AtP algorithm.
    """

    attn_cache = get_atp_caches(
        model, clean_tokens, corrupted_tokens, off_distribution_tokens, answer_token_indices
    )

    atp_component_contributions: list[t.Tensor] = [
        atp_component_contribution(attn_cache[i]) for i in range(len(attn_cache))
    ]  # layer list[head]
    return atp_component_contributions


def get_atp_caches(
    model: LanguageModel,
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
    num_layers, num_heads, _ = get_num_layers_heads(model)
    attn: GPT2Attention = model.transformer.h[0].attn  # type: ignore
    head_dim = int(attn.head_dim)

    with model.trace() as tracer:
        with tracer.invoke(clean_tokens) as clean_invoker:

            # Calculate L(M(x_clean))
            clean_logits: Float[t.Tensor, "examples seq_len vocab"] = (
                model.lm_head.output
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

            # Get keys and queries
            flat_clean_queries = [
                model.transformer.h[i].attn.c_attn.output.split(attn.split_size, dim=2)[0]
                for i in range(num_layers)
            ]  # type: ignore
            flat_clean_keys = [
                model.transformer.h[i].attn.c_attn.output.split(attn.split_size, dim=2)[1]
                for i in range(num_layers)
            ]  # type: ignore

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

        with tracer.invoke(off_distribution_tokens) as off_distribution_invoker:

            # Calculate L(M(x_corrupted))
            off_distribution_logits: Float[t.Tensor, "examples seq_len vocab"] = model.lm_head.output  # type: ignore
            off_distribution_logit_diff: Float[t.Tensor, "1"] = mean_logit_diff(
                off_distribution_logits, answer_token_indices
            ).save()  # type: ignore

            ioi_score = ioi_metric(
                clean_logit_diff, corrupted_logit_diff, off_distribution_logit_diff
            )  # scalar

            ioi_score.backward()

        q_corrupted_cache: list = []
        for i in range(num_layers):
            with tracer.invoke(clean_tokens) as q_patch_invoker:
                # Patch the queries with the corrupted queries
                model.transformer.h[i].attn.c_attn.output.split(attn.split_size, dim=2)[0] = (
                    flat_corrupted_queries[i]
                )
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
                model.transformer.h[i].attn.c_attn.output.split(attn.split_size, dim=2)[1] = (
                    flat_corrupted_keys[i]
                )
                # Compute and cache the attention probabilities with the patched keys
                k_corrupted_cache.append(
                    model.transformer.h[i].attn.attn_dropout.input[0][0].save()  # type: ignore
                )

                # TODO: Patch back in the clean attention probs

        # GradDrop
        grad_drop_collection: list = []
        for l in range(num_layers):
            with tracer.invoke(clean_tokens) as grad_drop_invoker:
                # Zero out the gradients on each layer from the residual connection
                model.transformer.h[l].attn.resid_dropout.input[0][0].grad.zero_()

                # Collect the gradients on the attention probabilities
                grad_cache_layer_dropped = [
                    model.transformer.h[i].attn.attn_dropout.input[0][0].grad.save()
                    for i in range(num_layers)  # type: ignore
                ]  # layer list[examples head seq_len seq_len]
                grad_drop_collection.append(
                    t.stack(grad_cache_layer_dropped, dim=0)  # type: ignore
                )  # dropped_layer list[layer examples head seq_len seq_len]]

        tensor_grad_drop_cache = t.stack(
            grad_drop_collection, dim=0
        )  # dropped_layer layer examples head seq_len seq_len

        tensor_grad_drop_cache = rearrange(
            tensor_grad_drop_cache,
            "dropped_layer layer examples head seq_len seq_len -> layer dropped_layer examples head seq_len seq_len",
        )

        grad_drop_cache = [
            tensor_grad_drop_cache[i] for i in range(len(tensor_grad_drop_cache))
        ]  # layer list[dropped_layer examples head seq_len seq_len]

    logger.debug(clean_logit_diff)
    logger.debug(corrupted_logit_diff)
    logger.debug(off_distribution_logit_diff)

    clean_cache = [value.value for value in clean_cache]
    clean_grad_cache = [value.value for value in clean_grad_cache]
    corrupted_cache = [value.value for value in corrupted_cache]

    q_corrupted_cache = [value.value for value in q_corrupted_cache]
    k_corrupted_cache = [value.value for value in k_corrupted_cache]

    grad_drop_cache = [value for value in grad_drop_cache]

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


def main():
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    prompt_store = build_prompt_store(tokeniser)
    clean_tokens, corrupted_tokens, off_distribution_tokens, answer_token_indices = (
        prompt_store.prepare_tokens_and_indices()
    )

    atp_component_contributions = run_atp(
        model, clean_tokens, corrupted_tokens, off_distribution_tokens, answer_token_indices
    )

    attn_contributions_tensor = t.stack(
        atp_component_contributions, dim=0
    )  # layer head seq_len seq_len

    logger.info(attn_contributions_tensor.shape)

    clean_token_strs = [tokeniser.decode(token) for token in clean_tokens]

    _num_layers, _num_heads, head_names_signed = get_num_layers_heads(model)

    # plot_attention_attributions(
    #     attention_attr=attn_contributions_tensor,
    #     token_strs=clean_token_strs,
    #     head_names_signed=head_names_signed,
    # )

    plot_single_attention_pattern(attn_contributions_tensor[0, 0, :, :])

    # print(html_plot)


if __name__ == "__main__":

    model = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
    # model = LanguageModel("delphi-suite/v0-llama2-100k", device_map="mps", dispatch=True)
    # model = LanguageModel("roneneldan/TinyStories-1M", device_map="cpu", dispatch=True)
    tokeniser = model.tokenizer

    main()
