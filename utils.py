# many are copied from https://github.com/mattneary/attention/blob/master/attention/attention.py
# here it nullifies the attention over the first token (<bos>)
# which in practice we find to be a good idea
from io import BytesIO
from PIL import Image
import requests
import torch
import numpy as np
import cv2


def aggregate_llm_attention(attn):
    """Extract average attention vector"""
    avged = []
    for layer in attn:
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = torch.concat(
            (
                # We zero the first entry because it's what's called
                # null attention (https://aclanthology.org/W19-4808.pdf)
                torch.tensor([0.0]),
                # usually there's only one item in attns_per_head but
                # on the first generation, there's a row for each token
                # in the prompt as well, so take [-1]
                attns_per_head[-1][1:].cpu(),
                # attns_per_head[-1].cpu(),
                # add zero for the final generated token, which never
                # gets any attention
                torch.tensor([0.0]),
            )
        )
        avged.append(vec / vec.sum())
    return torch.stack(avged).mean(dim=0)


def aggregate_vit_attention(attn, select_layer=-2, all_prev_layers=True):
    """Assuming LLaVA-style `select_layer` which is -2 by default"""
    if all_prev_layers:
        avged = []
        for i, layer in enumerate(attn):
            if i > len(attn) + select_layer:
                break
            layer_attns = layer.squeeze(0)
            attns_per_head = layer_attns.mean(dim=0)
            # vec = attns_per_head[1:, 1:].cpu()  # the first token is <CLS>
            # attns_per_head[1:, 1:] = 0
            vec = attns_per_head.cpu()
            avged.append(vec / vec.sum(-1, keepdim=True))
        return torch.stack(avged).mean(dim=0).to(torch.float16)
    else:
        layer = attn[select_layer]
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = attns_per_head[1:, 1:].cpu()
        return vec / vec.sum(-1, keepdim=True)


import torch
import torch.nn.functional as F


def interpolate_visual_tokens(
    x: torch.Tensor, target_tokens: int = 400
) -> torch.Tensor:
    """
    Interpolates visual tokens from shape [N, D] to [target_tokens, D]
    by reshaping to 2D, bilinearly upsampling, then flattening.

    Args:
        x: [N, D] tensor (e.g., [101, 768])
        target_tokens: number of desired output tokens (e.g., 400)

    Returns:
        [target_tokens, D] tensor
    """

    if x.ndim == 1:
        x = x.unsqueeze(-1)

    N, D = x.shape

    # Compute input 2D grid size (H_in, W_in) to hold N tokens
    H_in = W_in = int((N - 1) ** 0.5) + 1  # ceil(sqrt(N))
    pad_len = H_in * W_in - N
    x_padded = F.pad(x, (0, 0, 0, pad_len))  # [H_in*W_in, D]

    # Reshape to [D, H, W]
    x_2d = x_padded.view(H_in, W_in, D).permute(2, 0, 1).unsqueeze(0)  # [1, D, H, W]

    # Interpolate to target 2D size
    H_out = W_out = int(target_tokens**0.5)
    x_up = F.interpolate(
        x_2d, size=(H_out, W_out), mode="bilinear", align_corners=False
    )  # [1, D, H_out, W_out]

    # Flatten and transpose back to [target_tokens, D]
    x_flat = x_up.squeeze(0).permute(1, 2, 0).reshape(-1, D)  # [H_out*W_out, D]

    return x_flat[:target_tokens].squeeze(-1)  # Trim if necessary


def aggregate_qwen25vl_full_attention(attn, full_layers=[-4, -3, -2, -1]):
    """
    Aggregate attention from full-attention layers of Qwen2.5-VL.

    Args:
        attn: List/Tuple of tensors per layer [(1, heads, seq_len, seq_len), ...]
        full_layers: Indices of layers using global attention (default: last 4)

    Returns:
        Tensor of shape (seq_len, seq_len)
    """

    avged = []
    layers = [attn[i] for i in full_layers]
    for layer in layers:
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = attns_per_head[1:, 1:].cpu()  # the first token is <CLS>
        attns_per_head[1:, 1:] = 0
        vec = attns_per_head.cpu()
        avged.append(vec / vec.sum(-1, keepdim=True))
    return torch.stack(avged).mean(dim=0).to(torch.float16)


def heterogenous_stack(vecs):
    """Pad vectors with zeros then stack"""
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack(
        [torch.concat((v, torch.zeros(max_length - v.shape[0]))) for v in vecs]
    )


def load_image(image_path_or_url):
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith(
        "https://"
    ):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path_or_url).convert("RGB")
    return image


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_HSV)
    hm = np.float32(heatmap) / 255

    cam = hm + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam), heatmap
