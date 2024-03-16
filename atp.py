import torch as t
from einops import rearrange
from jaxtyping import Float, Int
from nnsight import LanguageModel

from helpers import get_logit_diff, ioi_metric
from plot import plot_attention_attributions
from prompt_store import build_prompt_store

model = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
# model = LanguageModel("delphi-suite/v0-llama2-100k", device_map="mps", dispatch=True)
# model = LanguageModel("roneneldan/TinyStories-1M", device_map="cpu", dispatch=True)
tokeniser = model.tokenizer


def run_atp(
    model: LanguageModel,
    clean_tokens: Int[t.Tensor, "examples"],
    corrupted_tokens: Int[t.Tensor, "examples"],
    answer_token_indices: Int[t.Tensor, "examples, 2"],
):
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
    _type_
        _description_
    """
    with model.trace() as tracer:
        with tracer.invoke((clean_tokens,)) as invoker:

            # Calculate L(M(x_clean))
            clean_logits: Float[t.Tensor, "examples vocab"] = (
                model.lm_head.output
            )  # type: ignore # M(x_clean)
            clean_logit_diff: Float[t.Tensor, "examples"] = get_logit_diff(clean_logits, answer_token_indices).item().save()  # type: ignore

            # print(clean_logit_diff)

            # Cache the clean activations and gradients for all the nodes
            clean_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].save()
                for i in range(len(model.transformer.h))  # type: ignore
            ]

            clean_grad_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].grad.save()
                for i in range(len(model.transformer.h))  # type: ignore
            ]

        with tracer.invoke((corrupted_tokens,)) as invoker:

            # Calculate L(M(x_corrupted))
            corrupted_logits: Float[t.Tensor, "examples vocab"] = model.lm_head.output  # type: ignore
            corrupted_logit_diff: Float[t.Tensor, "examples"] = (
                get_logit_diff(corrupted_logits, answer_token_indices).item().save()  # type: ignore
            )

            # print(corrupted_logit_diff)

            # Cache the corrupted activations and gradients for all the nodes
            corrupted_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].save()
                for i in range(len(model.transformer.h))  # type: ignore
            ]

            corrupted_grad_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].grad.save()
                for i in range(len(model.transformer.h))  # type: ignore
            ]

            clean_ioi_score = ioi_metric(
                clean_logits,
                clean_logit_diff,
                corrupted_logit_diff,
                answer_token_indices,
            ).save()  # type: ignore

            corrupted_ioi_score = ioi_metric(
                corrupted_logits,
                clean_logit_diff,
                corrupted_logit_diff,
                answer_token_indices,
            ).save()  # type: ignore

            (corrupted_ioi_score + clean_ioi_score).backward()

    clean_cache = t.stack([value.value for value in clean_cache])
    clean_grad_cache = t.stack([value.value for value in clean_grad_cache])
    corrupted_cache = t.stack([value.value for value in corrupted_cache])
    corrupted_grad_cache = t.stack([value.value for value in corrupted_grad_cache])

    print("Clean Value:", clean_ioi_score.value.item())
    print("Corrupted Value:", corrupted_ioi_score.value.item())

    return (
        clean_cache,
        clean_grad_cache,
        corrupted_cache,
        corrupted_grad_cache,
        clean_logit_diff,
        corrupted_logit_diff,
    )


def create_attention_attr(clean_cache: t.Tensor, clean_grad_cache: t.Tensor) -> t.Tensor:
    attention_attr = clean_grad_cache * clean_cache
    attention_attr = rearrange(
        attention_attr,
        "layer batch head_index dest src -> batch layer head_index dest src",
    )
    return attention_attr


def main():
    prompt_store = build_prompt_store(tokeniser)
    clean_tokens, corrupted_tokens, answer_token_indices = prompt_store.prepare_tokens_and_indices()
    print(clean_tokens[0])
    print(corrupted_tokens[0])

    (
        clean_cache,
        clean_grad_cache,
        corrupted_cache,
        corrupted_grad_cache,
        clean_logit_diff,
        corrupted_logit_diff,
    ) = run_atp(model, clean_tokens, corrupted_tokens, answer_token_indices)

    print(clean_logit_diff)
    print(corrupted_logit_diff)

    attention_attr = create_attention_attr(clean_cache, clean_grad_cache)

    # plot_attention_attributions(
    #     attention_attr,
    #     clean_tokens,
    #     index=0,
    #     title="Attention Attribution for first sequence",
    # )


if __name__ == "__main__":
    # print(model)
    main()
