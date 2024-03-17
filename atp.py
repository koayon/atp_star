import sys
from dataclasses import dataclass
from typing import Optional

import torch as t
from bleach import clean
from einops import einsum
from jaxtyping import Float, Int
from loguru import logger
from nnsight import LanguageModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from helpers import get_num_layers_heads, ioi_metric, mean_logit_diff
from plot import plot_attention_attributions, plot_single_attention_pattern
from prompt_store import build_prompt_store


@dataclass
class AttentionLayerCache:
    attn_probabilities: t.Tensor
    queries: t.Tensor
    keys: t.Tensor
    grad_attn_probabilities: Optional[t.Tensor]


def atp_component_contribution(
    clean_attn_layer_cache: AttentionLayerCache,
    corrupted_attn_layer_cache: AttentionLayerCache,
) -> Float[t.Tensor, "head seq_len seq_len"]:
    """Calculates the intervention effect of a node in the transformer (here attention)."""
    if clean_attn_layer_cache.grad_attn_probabilities is None:
        raise ValueError("No gradients found in the clean attention layer cache.")

    # Get the activations and gradients for the node
    node_clean_activation: Float[t.Tensor, "examples head seq_len seq_len"] = (
        clean_attn_layer_cache.attn_probabilities
    )
    node_corrupted_activation: Float[t.Tensor, "examples head seq_len seq_len"] = (
        corrupted_attn_layer_cache.attn_probabilities
    )
    grad_wrt_node: Float[t.Tensor, "examples head seq_len seq_len"] = (
        clean_attn_layer_cache.grad_attn_probabilities
    )

    # Calculate the intervention effect I_{AtP}(n; x_clean, x_corrupted)
    activation_diff = node_corrupted_activation - node_clean_activation

    logger.debug(activation_diff.max())

    intervention_effect = einsum(
        activation_diff,
        grad_wrt_node,
        "batch head seq_len1 seq_len2, batch head seq_len1 seq_len2 -> batch head seq_len1 seq_len2",
    )

    # Calculate the contribution c_{AtP}(n) = ExpectedValue(|I_{AtP}(n; x_clean, x_corrupted)|)
    contribution = t.mean(intervention_effect.abs(), dim=0)  # head seq_len seq_len
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

    (
        clean_attn_cache,
        corrupted_attn_cache,
    ) = get_atp_caches(
        model, clean_tokens, corrupted_tokens, off_distribution_tokens, answer_token_indices
    )

    atp_component_contributions: list[t.Tensor] = [
        atp_component_contribution(clean_attn_cache[i], corrupted_attn_cache[i])
        for i in range(len(clean_attn_cache))
    ]  # layer list[head]
    return atp_component_contributions


def get_atp_caches(
    model: LanguageModel,
    clean_tokens: Int[t.Tensor, "examples"],
    corrupted_tokens: Int[t.Tensor, "examples"],
    off_distribution_tokens: Int[t.Tensor, "examples"],
    answer_token_indices: Int[t.Tensor, "examples 2"],
) -> tuple[list[AttentionLayerCache], list[AttentionLayerCache]]:
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

            # Get keys and values
            flat_queries = [
                model.transformer.h[i].attn.c_attn.output.split(attn.split_size, dim=2)[0]
                for i in range(num_layers)
            ]  # type: ignore
            flat_keys = [
                model.transformer.h[i].attn.c_attn.output.split(attn.split_size, dim=2)[1]
                for i in range(num_layers)
            ]  # type: ignore

            clean_queries = [
                attn._split_heads(flat_query, num_heads, head_dim).save()
                for flat_query in flat_queries
            ]
            clean_keys = [
                attn._split_heads(flat_key, num_heads, head_dim).save() for flat_key in flat_keys
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

            flat_queries = [
                model.transformer.h[i].attn.c_attn.output.split(attn.split_size, dim=2)[0]
                for i in range(num_layers)
            ]  # type: ignore
            flat_keys = [
                model.transformer.h[i].attn.c_attn.output.split(attn.split_size, dim=2)[1]
                for i in range(num_layers)
            ]  # type: ignore

            corrupted_queries = [
                attn._split_heads(flat_query, num_heads, head_dim).save()
                for flat_query in flat_queries
            ]
            corrupted_keys = [
                attn._split_heads(flat_key, num_heads, head_dim).save() for flat_key in flat_keys
            ]

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

    logger.debug(clean_logit_diff)
    logger.debug(corrupted_logit_diff)
    logger.debug(off_distribution_logit_diff)

    clean_cache = [value.value for value in clean_cache]
    clean_grad_cache = [value.value for value in clean_grad_cache]
    corrupted_cache = [value.value for value in corrupted_cache]

    clean_queries_cache = [value.value for value in clean_queries]
    clean_keys_cache = [value.value for value in clean_keys]
    corrupted_queries_cache = [value.value for value in corrupted_queries]
    corrupted_keys_cache = [value.value for value in corrupted_keys]

    clean_attn_cache = [
        AttentionLayerCache(
            clean_cache[i], clean_queries_cache[i], clean_keys_cache[i], clean_grad_cache[i]
        )
        for i in range(len(clean_cache))
    ]

    corrupted_attn_cache = [
        AttentionLayerCache(
            corrupted_cache[i], corrupted_queries_cache[i], corrupted_keys_cache[i], None
        )
        for i in range(len(corrupted_cache))
    ]

    return (
        clean_attn_cache,
        corrupted_attn_cache,
    )


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

    # print(model)
    main()

    # print(model.transformer.h[0].attn.c_attn.weight.shape)
