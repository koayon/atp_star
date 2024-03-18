import tempfile
import webbrowser

import circuitsvis as cv
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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


def plot_tensor_2d(input: t.Tensor) -> None:
    # Convert the PyTorch tensor to a numpy array
    input_np = input.detach().numpy()

    # Use seaborn to create a heatmap
    sns.heatmap(input_np)

    # Display the plot
    plt.show()


def plot_tensor_3d(
    input: Float[t.Tensor, "layer head seq"], title: str, sequence_positions: list[str]
) -> None:
    input_np = input.detach().numpy()

    fig = go.Figure()

    # Add each 2D tensor slice as a separate image trace
    for i in range(input_np.shape[0]):
        fig.add_trace(
            go.Heatmap(z=input_np[i], visible=(i == 0))  # Only the first slice is visible initially
        )

    # Create buttons to toggle visibility
    buttons = []
    for i in range(input_np.shape[0]):
        visibility = [False] * input_np.shape[0]
        visibility[i] = True  # Only the current slice is visible
        button = dict(label=f"Layer {i+1}", method="update", args=[{"visible": visibility}])
        buttons.append(button)

    # Update layout with buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top",
            )
        ]
    )

    # Set axis titles
    # fig.update_xaxes(title_text="Sequence Position")
    # fig.update_yaxes(title_text="Head")

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Sequence Position",
            tickvals=list(range(len(sequence_positions))),
            ticktext=sequence_positions,
        ),
        yaxis=dict(title="Head"),
    )

    fig.show(renderer="browser")


if __name__ == "__main__":
    # Create a 3D tensor with random values
    input = t.rand(3, 4, 5)

    # Plot the tensor
    plot_tensor_3d(input, "Testing 3d plotting", ["a", "b", "c", "d", "e"])
