import torch as t
from einops import einsum, rearrange
from jaxtyping import Float, Int
from nnsight import LanguageModel

from helpers import get_logit_diff, ioi_metric
from plot import plot_attention_attributions
from prompt_store import build_prompt_store

model = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
# model = LanguageModel("delphi-suite/v0-llama2-100k", device_map="mps", dispatch=True)
# model = LanguageModel("roneneldan/TinyStories-1M", device_map="cpu", dispatch=True)
tokeniser = model.tokenizer


def atp_component_contribution(
    node_clean_activation: Float[t.Tensor, "examples head seq_len seq_len"],
    node_corrupted_activation: Float[t.Tensor, "examples head seq_len seq_len"],
    grad_wrt_node: Float[t.Tensor, "examples head seq_len seq_len"],
) -> Float[t.Tensor, "head seq_len seq_len"]:
    """Calculates the intervention effect of a node in the transformer (here attention)."""

    # Calculate the intervention effect I_{AtP}(n; x_clean, x_corrupted)
    activation_diff = node_corrupted_activation - node_clean_activation
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
    atp_component_contributions : list[float]
        The approximate contribution of each component to the metric, as given by the AtP algorithm.
    """

    (
        clean_cache,
        corrupted_cache,
        clean_grad_cache,
    ) = get_atp_caches(model, clean_tokens, corrupted_tokens, answer_token_indices)

    atp_component_contributions: list[t.Tensor] = [
        atp_component_contribution(clean_cache[i], corrupted_cache[i], clean_grad_cache[i])
        for i in range(len(clean_cache))
    ]  # layer list[head]
    return atp_component_contributions


def get_atp_caches(
    model: LanguageModel,
    clean_tokens: Int[t.Tensor, "examples"],
    corrupted_tokens: Int[t.Tensor, "examples"],
    answer_token_indices: Int[t.Tensor, "examples 2"],
) -> tuple[list[t.Tensor], list[t.Tensor], list[t.Tensor]]:
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
    with model.trace() as tracer:
        with tracer.invoke((clean_tokens,)) as invoker:

            # Calculate L(M(x_clean))
            clean_logits: Float[t.Tensor, "examples seq_len vocab"] = (
                model.lm_head.output
            )  # type: ignore # M(x_clean)
            clean_logit_diff: Float[t.Tensor, "examples"] = get_logit_diff(clean_logits, answer_token_indices).item().save()  # type: ignore

            # print(clean_logit_diff)

            # Cache the clean activations and gradients for all the nodes
            clean_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].save()
                for i in range(len(model.transformer.h))  # type: ignore
                # batch head_index seq_len seq_len
            ]

            clean_grad_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].grad.save()
                for i in range(len(model.transformer.h))  # type: ignore
            ]

        with tracer.invoke((corrupted_tokens,)) as invoker:

            # Calculate L(M(x_corrupted))
            corrupted_logits: Float[t.Tensor, "examples seq_len vocab"] = model.lm_head.output  # type: ignore
            corrupted_logit_diff: Float[t.Tensor, "examples"] = (
                get_logit_diff(corrupted_logits, answer_token_indices).item().save()  # type: ignore
            )

            # print(corrupted_logit_diff)

            # Cache the corrupted activations and gradients for all the nodes
            corrupted_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].save()
                for i in range(len(model.transformer.h))  # type: ignore
            ]

            clean_ioi_score = ioi_metric(
                clean_logits,
                clean_logit_diff,
                corrupted_logit_diff,
                answer_token_indices,
            )

            clean_ioi_score.backward()

    clean_cache = [value.value for value in clean_cache]
    clean_grad_cache = [value.value for value in clean_grad_cache]
    corrupted_cache = [value.value for value in corrupted_cache]

    return (
        clean_cache,
        corrupted_cache,
        clean_grad_cache,
    )


def main():
    prompt_store = build_prompt_store(tokeniser)
    clean_tokens, corrupted_tokens, answer_token_indices = prompt_store.prepare_tokens_and_indices()

    atp_component_contributions = run_atp(
        model, clean_tokens, corrupted_tokens, answer_token_indices
    )

    # plot_attention_attributions(
    #     attention_attr,
    #     clean_tokens,
    #     index=0,
    #     title="Attention Attribution for first sequence",
    # )


if __name__ == "__main__":
    # print(model)
    main()
