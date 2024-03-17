import tempfile
import webbrowser

import circuitsvis as cv
import plotly.express as px
import seaborn as sns
import torch as t
from circuitsvis.utils.render import RenderedHTML
from einops import rearrange
from jaxtyping import Float
from matplotlib import pyplot as plt


def show_html_in_browser(html_plot: RenderedHTML) -> None:
    # Save the HTML to a temporary file
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html") as f:
        url = "file://" + f.name
        f.write(str(html_plot))

    # Open the URL in a web browser
    webbrowser.open(url)


def plot_attention_attributions(
    attention_attr: Float[t.Tensor, "layer head dest src"],
    token_strs: list[str],
    head_names_signed: list[str],
    top_k=20,
    # title="",
) -> None:
    # if len(tokens.shape) == 2:
    #     token_strs = tokens[index]
    # if len(attention_attr.shape) == 5:
    #     attention_attr = attention_attr[index]

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

    html_plot = cv.circuitsvis.attention.attention_heads(
        tokens=token_strs,
        attention=attention_attr_signed[:top_k],
        attention_head_names=head_labels[:top_k],
    )
    show_html_in_browser(html_plot)


def plot_single_attention_pattern(attention_pattern: t.Tensor) -> None:
    # Convert the PyTorch tensor to a numpy array
    attention_pattern_np = attention_pattern.detach().numpy()

    # Use seaborn to create a heatmap
    sns.heatmap(attention_pattern_np)

    # Display the plot
    plt.show()
