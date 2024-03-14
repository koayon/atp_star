import torch as t
from jaxtyping import Float, Int


def ioi_metric(
    logits: Float[t.Tensor, "batch seq_len vocab"],
    CLEAN_BASELINE: float,
    CORRUPTED_BASELINE: float,
    answer_token_indices: Int[t.Tensor, "batch 2"],
) -> Float[t.Tensor, "batch"]:
    numerator = get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE
    denominator = CLEAN_BASELINE - CORRUPTED_BASELINE

    normalised_ioi_metric = numerator / denominator
    return normalised_ioi_metric


def get_logit_diff(
    logits: Float[t.Tensor, "batch seq_len vocab"],
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
    if len(logits.shape) == 3:
        # Get final logits only
        logits = logits[:, -1, :]

    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()
