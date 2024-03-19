import torch as t
from jaxtyping import Float, Int
from nnsight import LanguageModel


def ioi_metric(
    clean_logit_diff: Float[t.Tensor, "1"],
    corrupted_logit_diff: Float[t.Tensor, "1"],
    off_distribution_logit_diff: Float[t.Tensor, "1"],
) -> Float[t.Tensor, "1"]:
    """Calculates the IOI metric for a given batch of examples.
    Returns a scalar value."""

    numerator = clean_logit_diff - corrupted_logit_diff
    denominator = t.abs(clean_logit_diff - off_distribution_logit_diff)
    # denominator = off_distribution_logit_diff

    # print("Numerator", numerator)
    # print("Denominator", denominator)

    normalised_ioi_metric = numerator / denominator
    # normalised_ioi_metric = numerator
    return normalised_ioi_metric


def mean_logit_diff(
    logits: t.Tensor,
    answer_token_indices: Int[t.Tensor, "batch 2"],
) -> Float[t.Tensor, "1"]:
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

    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))  # examples 1
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))  # examples 1

    logit_diffs = (correct_logits - incorrect_logits).squeeze(1)  # examples
    mean_logit_diff = t.mean(logit_diffs)  # scalar
    return mean_logit_diff


def get_num_layers_heads(model: LanguageModel) -> tuple[int, int, list[str]]:
    """Given model, return the number of layers, number of heads and head names.

    Parameters
    ----------
    model : LanguageModel

    Returns
    -------
    num_layers: int
        The number of layers in the model.
    num_heads: int
        The number of heads in the model.
    head_names_signed: list[str]
        The names of the heads in the model.
    """
    num_layers = len(model.transformer.h)  # type: ignore
    num_heads = int(model.transformer.h[0].attn.num_heads)  # type: ignore

    head_names = [f"L{l}H{h}" for l in range(num_layers) for h in range(num_heads)]
    head_names_signed = [f"{name}{sign}" for name in head_names for sign in ["+", "-"]]
    # HEAD_NAMES_QKV = [f"{name}{act_name}" for name in HEAD_NAMES for act_name in ["Q", "K", "V"]]

    return num_layers, num_heads, head_names_signed
