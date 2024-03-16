import circuitsvis as cv
import torch as t
from einops import rearrange
from transformers import PreTrainedTokenizer


def plot_attention_attributions(
    attention_attr,
    tokens: t.Tensor,
    tokeniser: PreTrainedTokenizer,
    head_names_signed: list[str],
    top_k=20,
    index=0,
    # title="",
):
    if len(tokens.shape) == 2:
        token_strs = tokens[index]
    if len(attention_attr.shape) == 5:
        attention_attr = attention_attr[index]

    attention_attr_pos = attention_attr.clamp(min=-1e-5)
    attention_attr_neg = -attention_attr.clamp(max=1e-5)
    attention_attr_signed = t.stack([attention_attr_pos, attention_attr_neg], dim=0)
    attention_attr_signed = rearrange(
        attention_attr_signed,
        "sign layer head_index dest src -> (layer head_index sign) dest src",
    )
    attention_attr_signed = attention_attr_signed / attention_attr_signed.max()
    attention_attr_indices: t.LongTensor = (  # type: ignore
        attention_attr_signed.max(-1).values.max(-1).values.argsort(descending=True)
    )

    attention_attr_signed = attention_attr_signed[attention_attr_indices, :, :]
    head_labels = [head_names_signed[i.item()] for i in attention_attr_indices]  # type: ignore

    token_strs = [tokeniser.decode(token) for token in tokens]

    return cv.circuitsvis.attention.attention_heads(
        tokens=token_strs,
        attention=attention_attr_signed[:top_k],
        attention_head_names=head_labels[:top_k],
    )
