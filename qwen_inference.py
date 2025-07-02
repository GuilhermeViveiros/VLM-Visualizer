import os
import sys

sys.path.append("./models")
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import requests
from io import BytesIO

import torch
import torch.nn.functional as F


from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# It's recommended to install this utility from the Qwen-VL repository
# pip install qwen-vl-utils
from qwen_vl_utils import process_vision_info

from utils import (
    load_image,
    aggregate_llm_attention,
    aggregate_vit_attention,
    heterogenous_stack,
    show_mask_on_image,
    aggregate_qwen25vl_full_attention,
    interpolate_visual_tokens,
)

# Create output directory for saved images
output_dir = "/home/gviveiros/MMInsights/images"
os.makedirs(output_dir, exist_ok=True)

# ===> specify the model path
model_path = "Qwen/Qwen2.5-VL-3B-Instruct"

# load the model
load_8bit = False
load_4bit = False
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load the processor and model from Hugging Face
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",  # Necessary for attention extraction
)

# ===> specify the image path or url and the prompt text
image_path_or_url = (
    "https://github.com/open-compass/MMBench/blob/main/samples/MMBench/1.jpg?raw=true"
)
prompt_text = "What python code (do not use for loops or list comprehensions) can be used to generate the output in the image?"

################################################
# preparation for the generation
# Qwen-VL uses a chat template for formatting input
response = requests.get(image_path_or_url)
image = Image.open(BytesIO(response.content))
# lets reshape the image to 286x286
image = image.resize((286, 286))
image_size = image.size

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }
]

# Prepare input for the model
chat_text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)

# Process the inputs using the Qwen processor
inputs = processor(
    text=[chat_text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to("cuda")


################################################

print(prompt_text)


# ---
# generate the response
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
        output_attentions=True,
        output_hidden_states=True,
    )

generated_ids = outputs.sequences[0, len(inputs["input_ids"][0]) :]
text = processor.decode(generated_ids, skip_special_tokens=True).strip()
print(text)

# ---

# constructing the llm attention matrix
# NOTE: The structure of `outputs.attentions` may vary. This is an adaptation.
# Qwen's attention output needs to be carefully inspected. We assume a similar
# structure to LLaVA for this adaptation.
aggregated_prompt_attention = []
# The attentions are structured as (batch, layer, head, seq_len, seq_len)
# We are interested in the attentions of the generated tokens.
# The `attentions` in the output are from the language model part.
for i, layer_attns in enumerate(outputs.attentions[0]):
    # Average across heads
    attns_per_head = layer_attns.squeeze(0).mean(dim=0)
    cur = attns_per_head.cpu().clone()
    # Zero out attention to the first <bos> token for stability, except for the first token itself
    cur[1:, 0] = 0.0
    # Re-normalize
    cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
    aggregated_prompt_attention.append(cur)
aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)


# llm_attn_matrix will be of torch.Size([N, N])
# where N is the total number of input (both image and text ones) + output tokens
llm_attn_matrix = heterogenous_stack(
    [torch.tensor([1])]
    + list(aggregated_prompt_attention)
    + list(map(aggregate_llm_attention, outputs["attentions"]))
)


# ---

# visualize the llm attention matrix
# ===> adjust the gamma factor to enhance the visualization
#      higer gamma brings out more low attention values
gamma_factor = 1
enhanced_attn_m = np.power(llm_attn_matrix.numpy(), 1 / gamma_factor)

fig, ax = plt.subplots(figsize=(10, 20), dpi=150)
ax.imshow(
    enhanced_attn_m,
    vmin=enhanced_attn_m.min(),
    vmax=enhanced_attn_m.max(),
    interpolation="nearest",
)

# Save the LLM attention matrix visualization
plt.savefig(
    os.path.join(output_dir, "llm_attention_matrix.png"), bbox_inches="tight", dpi=150
)
plt.close()

# ---

# identify length or index of tokens
input_token_len = aggregated_prompt_attention.shape[0]
print("Shape of the prompt LLM attention matrix: ", outputs["attentions"][0][0].shape)
print("Input token length: ", input_token_len)


input_ids = inputs["input_ids"][0]
image_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
image_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

image_start_token_index = input_ids.tolist().index(image_start_token_id)
image_end_token_index = input_ids.tolist().index(image_end_token_id)

# The number of visual tokens can be obtained from the vision encoder config
num_patches = inputs["pixel_values"].shape[0]


vision_token_start = image_start_token_index
vision_token_end = image_end_token_index
output_token_len = len(generated_ids)

num_patch_tokens = 400
num_visual_tokens = vision_token_end - vision_token_start
patches_per_visual_token = num_patch_tokens // (num_visual_tokens - 2)


# The attention matrix rows correspond to generated tokens (0 to output_token_len-1)
# The columns correspond to input tokens (0 to input_token_len-1)
# ---

output_token_start = input_token_len
output_token_end = input_token_len + output_token_len
# ---

# Plot attention weights trend (no changes in this section)
print("Plotting attention weights trend...")
# look at the attention weights over the vision tokens
overall_attn_weights_over_vis_tokens = []
for i, (row, token) in enumerate(
    zip(llm_attn_matrix[input_token_len:], generated_ids.tolist())
):
    overall_attn_weights_over_vis_tokens.append(
        row[vision_token_start:vision_token_end].sum().item()
    )

# plot the trend of attention weights over the vision tokens
fig, ax = plt.subplots(figsize=(20, 5))
ax.plot(overall_attn_weights_over_vis_tokens)
ax.set_xticks(range(len(overall_attn_weights_over_vis_tokens)))
ax.set_xticklabels(
    [
        processor.decode(token, skip_special_tokens=True).strip()
        for token in generated_ids.tolist()
    ],
    rotation=75,
)
ax.set_title("at each token, the sum of attention weights over all the vision tokens")

# Save the attention weights trend plot
plt.savefig(
    os.path.join(output_dir, "attention_weights_trend.png"),
    bbox_inches="tight",
    dpi=150,
)
plt.close()

# ---

# vis_attn_matrix will be of torch.Size([N, N])
# where N is the number of vision tokens/patches
# `all_prev_layers=True` will average attention from all layers until the selected layer
# otherwise only the selected layer's attention will be used

vis_attn_matrix = aggregate_vit_attention(
    attn=[block.attn.attn_weights for block in model.visual.blocks]
)

print("vis attn matrix size: ", len(vis_attn_matrix))
print("vis attn matrix shape: ", vis_attn_matrix[0].shape)
print("vis attn matrix type: ", vis_attn_matrix[0].dtype)

# grid_size = model.get_vision_tower().num_patches_per_side

# Placeholder for vision attention matrix - this needs to be correctly implemented
# based on how Qwen-VL exposes vision encoder attentions.
image_size_h, image_size_w = image_inputs[0].size
grid_size_h = image_size_h // 14
grid_size_w = image_size_w // 14
# A placeholder identity matrix for demonstration if direct attention is not available
# vis_attn_matrix = torch.eye(num_patches)


num_image_per_row = 8
image_ratio = image_size[0] / image_size[1]
num_rows = output_token_len // num_image_per_row + (
    1 if output_token_len % num_image_per_row != 0 else 0
)
fig, axes = plt.subplots(
    num_rows,
    num_image_per_row,
    figsize=(10, (10 / num_image_per_row) * image_ratio * num_rows),
    dpi=150,
)
plt.subplots_adjust(wspace=0.05, hspace=0.2)

# whether visualize the attention heatmap or
# the image with the attention heatmap overlayed
vis_overlayed_with_attn = True


# ------------------------------------------------------------
# we need to create a mapping between the visual tokens and the ViT tokens.
# Qwen2.5VL uses 400 patches for this image.
# However the merger will reduce this patches to 101 visual tokens.
# Assume 400 ViT tokens are mapped uniformly to 100 visual tokens


output_token_inds = list(range(output_token_start, output_token_end))
for i, ax in enumerate(axes.flatten()):
    if i >= output_token_len:
        ax.axis("off")
        continue

    target_token_ind = output_token_inds[i]
    attn_weights_over_vis_tokens = llm_attn_matrix[target_token_ind][
        vision_token_start:vision_token_end
    ]
    attn_weights_over_vis_tokens = (
        attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.sum()
    )

    # Expand to patch-level attention
    attn_weights_patch_level = interpolate_visual_tokens(attn_weights_over_vis_tokens)

    # ------------------------------------------------------------

    attn_over_image = []
    for weight, vis_attn in zip(attn_weights_patch_level, vis_attn_matrix):
        vis_attn = vis_attn.reshape(grid_size_h, grid_size_w)
        # vis_attn = vis_attn / vis_attn.max()
        attn_over_image.append(vis_attn * weight)
        # attn_over_image.append(weight)
    attn_over_image = torch.stack(attn_over_image).sum(dim=0)
    attn_over_image = attn_over_image / attn_over_image.max()

    def interpolate_attn_over_image(attn_over_image, mode: str = "bicubic"):
        return F.interpolate(
            attn_over_image.unsqueeze(0).unsqueeze(0), size=image.size, mode=mode
        ).squeeze()

    attn_over_image = interpolate_attn_over_image(attn_over_image, mode="nearest")

    np_img = np.array(image)[:, :, ::-1]

    img_with_attn, heatmap = show_mask_on_image(np_img, attn_over_image.numpy())
    ax.imshow(heatmap if not vis_overlayed_with_attn else img_with_attn)
    ax.set_title(
        processor.decode(generated_ids[i], skip_special_tokens=True).strip(),
        fontsize=7,
        pad=1,
    )
    ax.axis("off")

# Save the token-specific attention visualizations
plt.savefig(
    os.path.join(output_dir, "token_attention_visualizations.png"),
    bbox_inches="tight",
    dpi=150,
)
plt.close()

print(f"All visualizations have been saved to the '{output_dir}' directory:")
print(f"- llm_attention_matrix.png")
print(f"- attention_weights_trend.png")
print(f"- token_attention_visualizations.png")

# ---
