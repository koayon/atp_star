import torch as t
from einops import rearrange
from nnsight import LanguageModel
from transformers import PreTrainedTokenizer

from helpers import get_logit_diff, ioi_metric
from plot import plot_attention_attributions

PROMPTS = [
    "When John and Mary went to the shops, John gave the bag to",
    "When John and Mary went to the shops, Mary gave the bag to",
    "When Tom and James went to the park, James gave the ball to",
    "When Tom and James went to the park, Tom gave the ball to",
    "When Dan and Sid went to the shops, Sid gave an apple to",
    "When Dan and Sid went to the shops, Dan gave an apple to",
    "After Martin and Amy went to the park, Amy gave a drink to",
    "After Martin and Amy went to the park, Martin gave a drink to",
]
ANSWERS = [
    (" Mary", " John"),
    (" John", " Mary"),
    (" Tom", " James"),
    (" James", " Tom"),
    (" Dan", " Sid"),
    (" Sid", " Dan"),
    (" Martin", " Amy"),
    (" Amy", " Martin"),
]

model = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
# model = LanguageModel("delphi-suite/v0-llama2-100k", device_map="mps", dispatch=True)
# model = LanguageModel("roneneldan/TinyStories-1M", device_map="cpu", dispatch=True)
tokeniser = model.tokenizer


def prepare_tokens_and_indices(
    tokeniser: PreTrainedTokenizer,
) -> tuple[t.Tensor, t.Tensor, t.LongTensor]:

    clean_tokens: t.Tensor = tokeniser(PROMPTS, return_tensors="pt")["input_ids"]  # type: ignore

    corrupted_tokens = clean_tokens[
        [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
    ]

    answer_token_indices: t.LongTensor = t.tensor(
        [
            [tokeniser(ANSWERS[i][j])["input_ids"][0] for j in range(2)]  # type: ignore
            for i in range(len(ANSWERS))
        ]
    )

    return clean_tokens, corrupted_tokens, answer_token_indices


def run_atp(
    model: LanguageModel,
    clean_tokens: t.Tensor,
    corrupted_tokens: t.Tensor,
    answer_token_indices: t.LongTensor,
):
    print(model)
    with model.trace() as tracer:
        with tracer.invoke((clean_tokens,)) as invoker:
            clean_logits = model.lm_head.output

            clean_logit_diff = (
                get_logit_diff(clean_logits, answer_token_indices).item().save()
            )

            # print(clean_logit_diff)

            clean_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].save()
                for i in range(len(model.transformer.h))
            ]

            clean_grad_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].grad.save()
                for i in range(len(model.transformer.h))
            ]

        with tracer.invoke((corrupted_tokens,)) as invoker:
            corrupted_logits = model.lm_head.output

            corrupted_logit_diff = (
                get_logit_diff(corrupted_logits, answer_token_indices).item().save()
            )

            # print(corrupted_logit_diff)

            # assert False

            corrupted_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].save()
                for i in range(len(model.transformer.h))
            ]

            corrupted_grad_cache = [
                model.transformer.h[i].attn.attn_dropout.input[0][0].grad.save()
                for i in range(len(model.transformer.h))
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


def create_attention_attr(
    clean_cache: t.Tensor, clean_grad_cache: t.Tensor
) -> t.Tensor:
    attention_attr = clean_grad_cache * clean_cache
    attention_attr = rearrange(
        attention_attr,
        "layer batch head_index dest src -> batch layer head_index dest src",
    )
    return attention_attr


def main():
    clean_tokens, corrupted_tokens, answer_token_indices = prepare_tokens_and_indices(
        tokeniser
    )
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
