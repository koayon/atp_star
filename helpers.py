import torch as t
from jaxtyping import Float, Int


def ioi_metric(
    logits: Float[t.Tensor, "batch seq_len vocab"],
    clean_baseline_logit_diff: float,
    corrupted_baseline_logit_diff: float,
    answer_token_indices: Int[t.Tensor, "batch 2"],
) -> Float[t.Tensor, "batch"]:
    # logger.debug(clean_baseline_logit_diff)
    # logger.debug(corrupted_baseline_logit_diff)
    numerator = (
        get_logit_diff(logits, answer_token_indices) - corrupted_baseline_logit_diff
    )
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
