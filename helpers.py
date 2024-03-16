import torch as t
from jaxtyping import Float, Int
from nnsight import LanguageModel


def ioi_metric(
    logits: Float[t.Tensor, "batch seq_len vocab"],
    clean_baseline_logit_diff: Float[t.Tensor, "batch"],
    corrupted_baseline_logit_diff: Float[t.Tensor, "batch"],
    answer_token_indices: Int[t.Tensor, "batch 2"],
) -> Float[t.Tensor, "batch"]:
    numerator = get_logit_diff(logits, answer_token_indices) - corrupted_baseline_logit_diff
    denominator = clean_baseline_logit_diff - corrupted_baseline_logit_diff

    normalised_ioi_metric = numerator / denominator
    return normalised_ioi_metric


def get_logit_diff(
    logits: t.Tensor,
    answer_token_indices: Int[t.Tensor, "batch 2"],
) -> Float[t.Tensor, "batch"]:
    """Compares the difference between the logits of the correct and incorrect answers.
    E.g. what's the difference between the answer being John or Mary in the IOI task?

    Parameters
    ----------
    logits : t.Tensor
        _description_
    answer_token_indices : t.LongTensor
        _description_

    Returns
    -------
    t.Tensor
        _description_
    """
    if len(logits.shape) == 3:  # batch, seq_len, vocab
        # Get final logits only
        logits = logits[:, -1, :]
    elif len(logits.shape) == 4:  # _, batch, seq_len, vocab
        logits = logits[0, :, -1, :]

    print("logits.shape", logits.shape)
    print("answer_token_indices.shape", answer_token_indices.shape)

    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()


def get_num_layers_heads(model: LanguageModel) -> tuple[int, int, list[str]]:
    num_layers = len(model.transformer.h)  # type: ignore
    num_heads = int(model.transformer.h[0].attn.num_heads)  # type: ignore

    head_names = [f"L{l}H{h}" for l in range(num_layers) for h in range(num_heads)]
    head_names_signed = [f"{name}{sign}" for name in head_names for sign in ["+", "-"]]
    # HEAD_NAMES_QKV = [f"{name}{act_name}" for name in HEAD_NAMES for act_name in ["Q", "K", "V"]]

    return num_layers, num_heads, head_names_signed
