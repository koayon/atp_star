import circuitsvis as cv
import torch as t
from einops import rearrange
from nnsight import LanguageModel
from transformers import PreTrainedTokenizer


def get_num_layers_heads(model: LanguageModel):
    n_layers = len(model.transformer.h)  # type: ignore
    n_heads = model.transformer.h[0].attn.num_heads
    return n_layers, n_heads


def _plot_attention_attr(
    attention_attr,
    tokens: t.Tensor,
    tokeniser: PreTrainedTokenizer,
    head_names_signed: list[str],
    top_k=20,
    index=0,
    title="",
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


def plot_attention_attributions(
    model: LanguageModel,
    attention_attr,
    tokens: t.Tensor,
    tokeniser: PreTrainedTokenizer,
    head_names_signed: list[str],
    top_k=20,
    index=0,
    title="",
):
    n_layers, n_heads = get_num_layers_heads(model)

    HEAD_NAMES = [
        f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)  #  type: ignore
    ]  #  type: ignore
    HEAD_NAMES_SIGNED = [f"{name}{sign}" for name in HEAD_NAMES for sign in ["+", "-"]]
    HEAD_NAMES_QKV = [
        f"{name}{act_name}" for name in HEAD_NAMES for act_name in ["Q", "K", "V"]
    ]
    print(HEAD_NAMES[:5])
    print(HEAD_NAMES_SIGNED[:5])
    print(HEAD_NAMES_QKV[:5])

    _plot_attention_attr(
        attention_attr,
        tokens,
        tokeniser,
        head_names_signed,
        top_k,
        index,
        title,
    )
