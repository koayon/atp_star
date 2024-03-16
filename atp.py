import torch as t
from einops import rearrange
from jaxtyping import Float, Int
from nnsight import LanguageModel
from transformers import PreTrainedTokenizer

from helpers import get_logit_diff, ioi_metric
from plot import plot_attention_attributions
from prompt_store import PROMPT_STORE

model = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
# model = LanguageModel("delphi-suite/v0-llama2-100k", device_map="mps", dispatch=True)
# model = LanguageModel("roneneldan/TinyStories-1M", device_map="cpu", dispatch=True)
tokeniser = model.tokenizer


def prepare_tokens_and_indices(
    tokeniser: PreTrainedTokenizer,
) -> tuple[Int[t.Tensor, "examples"], Int[t.Tensor, "examples"], Int[t.Tensor, "examples, 2"]]:

    clean_tokens = t.tensor(tokeniser(PROMPT_STORE.clean_prompts, return_tensors="pt")["input_ids"])
    corrupted_tokens = t.tensor(
        tokeniser(PROMPT_STORE.corrupted_prompts, return_tensors="pt")["input_ids"]
    )

    correct_answer_token_ids = t.tensor(
        tokeniser(PROMPT_STORE.correct_answers, return_tensors="pt")["input_ids"]
    )  # examples
    incorrect_answer_token_ids = t.tensor(
        tokeniser(PROMPT_STORE.incorrect_answers, return_tensors="pt")["input_ids"]
    )  # examples

    answer_token_indices = t.stack(
        [correct_answer_token_ids, incorrect_answer_token_ids], dim=1
    )  # examples, 2

    return clean_tokens, corrupted_tokens, answer_token_indices


def run_atp(
    model: LanguageModel,
    clean_tokens: Int[t.Tensor, "examples"],
    corrupted_tokens: Int[t.Tensor, "examples"],
    answer_token_indices: Int[t.Tensor, "examples, 2"],
):
    print(model)
    with model.trace() as tracer:
        with tracer.invoke((clean_tokens,)) as invoker:
            clean_logits = model.lm_head.output

            clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).item().save()  # type: ignore

            # print(clean_logit_diff)

            clean_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].save()
                for i in range(len(model.transformer.h))  # type: ignore
            ]

            clean_grad_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].grad.save()
                for i in range(len(model.transformer.h))  # type: ignore
            ]

        with tracer.invoke((corrupted_tokens,)) as invoker:
            corrupted_logits = model.lm_head.output

            corrupted_logit_diff = (
                get_logit_diff(corrupted_logits, answer_token_indices).item().save()  # type: ignore
            )

            # print(corrupted_logit_diff)

            # assert False

            corrupted_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].save()
                for i in range(len(model.transformer.h))  # type: ignore
            ]

            corrupted_grad_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].grad.save()
                for i in range(len(model.transformer.h))  # type: ignore
            ]

            clean_ioi_score = ioi_metric(
                clean_logits,  # type: ignore
                clean_logit_diff,
                corrupted_logit_diff,
                answer_token_indices,
            ).save()  # type: ignore

            corrupted_ioi_score = ioi_metric(
                corrupted_logits,  # type: ignore
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
    clean_tokens, corrupted_tokens, answer_token_indices = prepare_tokens_and_indices(tokeniser)
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
