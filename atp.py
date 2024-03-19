import sys

import torch as t
from einops import einsum, rearrange
from jaxtyping import Float, Int
from loguru import logger
from transformers import AutoTokenizer

from interventions import AttentionLayerCache, MLPLayerCache, get_atp_caches
from plot import plot_tensor_2d, plot_tensor_3d
from prompt_store import build_prompt_store


def atp_attn_component_contribution(
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


def atp_mlp_contribution(
    mlp_layer_cache: MLPLayerCache, use_grad_drop: bool = False
) -> Float[t.Tensor, "seq_len"]:
    """Calculates the intervention effect of a node in the transformer (here the MLP)."""
    # Get the activations and gradients for the node
    node_clean_activation: Float[t.Tensor, "examples seq_len"] = mlp_layer_cache.clean_mlp_output
    node_corrupted_activation: Float[t.Tensor, "examples seq_len"] = (
        mlp_layer_cache.corrupted_mlp_output
    )
    grad_wrt_node: Float[t.Tensor, "examples seq_len"] = mlp_layer_cache.clean_grad_mlp_output

    # Calculate the intervention effect I_{AtP}(n; x_clean, x_corrupted)
    activation_diff = node_corrupted_activation - node_clean_activation

    logger.debug(activation_diff.max())

    if use_grad_drop and mlp_layer_cache.grad_drop_mlp_output is not None:
        grad_drop_wrt_node: Float[t.Tensor, "dropped_layer examples seq_len"] = (
            mlp_layer_cache.grad_drop_mlp_output
        )

        # Equation 11
        # Calculate the contribution
        # c_{AtP+GD}(n) = ExpectedValue(1/(L-1) * |I_{AtP+GD}_l(n; x_clean, x_corrupted)|)
        intervention_effects = einsum(
            activation_diff,
            grad_drop_wrt_node,
            "batch seq_len hidden_dim, dropped_layer batch seq_len hidden_dim -> dropped_layer batch seq_len",
        )

        individual_contributions = t.sum(intervention_effects.abs(), dim=0) / (
            len(grad_drop_wrt_node) - 1
        )  # batch seq_len
        contribution = t.mean(individual_contributions, dim=0)  # seq_len

    elif use_grad_drop:
        raise ValueError("grad_drop_attn_probs not provided")

    else:
        # Equation 4
        intervention_effect = einsum(
            activation_diff,
            grad_wrt_node,
            "batch head seq_len hidden_dim, batch head seq_len hidden_dim -> batch head seq_len",
        )

        # Equation 5
        # Calculate the contribution c_{AtP}(n) = ExpectedValue(|I_{AtP}(n; x_clean, x_corrupted)|)
        contribution = t.mean(intervention_effect.abs(), dim=0)  # seq_len
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
    model_name: str,
    clean_tokens: Int[t.Tensor, "examples"],
    corrupted_tokens: Int[t.Tensor, "examples"],
    off_distribution_tokens: Int[t.Tensor, "examples"],
    answer_token_indices: Int[t.Tensor, "examples 2"],
    use_grad_drop: bool = False,
    testing: bool = False,
) -> tuple[list[t.Tensor], list[t.Tensor]]:
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
    atp_attn_component_contributions : list[Tensor]
        The approximate contribution of each component to the metric, as given by the AtP algorithm.
    atp_mlp_component_contributions : list[Tensor]
        The approximate contribution of each component to the metric, as given by the AtP algorithm.
    """
    # model = LanguageModel(model_name, device_map="cpu", dispatch=True)

    attn_cache, mlp_cache = get_atp_caches(
        model_name,
        clean_tokens,
        corrupted_tokens,
        off_distribution_tokens,
        answer_token_indices,
        testing=testing,
    )

    atp_attn_component_contributions: list[t.Tensor] = [
        atp_attn_component_contribution(attn_cache[i], use_grad_drop=use_grad_drop)
        for i in range(len(attn_cache))
    ]  # layer list[head]
    atp_mlp_component_contributions: list[t.Tensor] = [
        atp_mlp_contribution(mlp_cache[i], use_grad_drop=use_grad_drop)
        for i in range(len(mlp_cache))
    ]  # layer list[head]
    return atp_attn_component_contributions, atp_mlp_component_contributions


def main():
    prompt_store = build_prompt_store(tokeniser)
    clean_tokens, corrupted_tokens, off_distribution_tokens, answer_token_indices = (
        prompt_store.prepare_tokens_and_indices()
    )

    attn_atp_component_contributions, mlp_atp_component_contributions = run_atp(
        model_name=MODEL_NAME,
        clean_tokens=clean_tokens,
        corrupted_tokens=corrupted_tokens,
        off_distribution_tokens=off_distribution_tokens,
        answer_token_indices=answer_token_indices,
        use_grad_drop=True,
        # testing=True,
    )

    attn_contributions_tensor = t.stack(
        attn_atp_component_contributions, dim=0
    )  # layer head seq_len

    mlp_contributions_tensor = t.stack(mlp_atp_component_contributions, dim=0)  # layer seq_len

    logger.info(attn_contributions_tensor.shape)
    logger.info(mlp_contributions_tensor.shape)

    clean_token_strs = [tokeniser.decode(token) for token in clean_tokens[0]]

    # plot_tensor_2d(attn_contributions_tensor[0, :, :])

    plot_tensor_3d(
        attn_contributions_tensor,
        title="AtP* Attention Component Contributions",
        sequence_positions=clean_token_strs,
    )

    plot_tensor_2d(
        mlp_contributions_tensor,
        title="AtP* MLP Component Contributions",
        sequence_positions=clean_token_strs,
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    MODEL_NAME = "openai-community/gpt2"
    tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)

    # model = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
    # model = LanguageModel("delphi-suite/v0-llama2-100k", device_map="mps", dispatch=True)
    # model = LanguageModel("roneneldan/TinyStories-1M", device_map="cpu", dispatch=True)
    # tokeniser = model.tokenizer

    main()
