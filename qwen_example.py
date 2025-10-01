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

from utils import (
    aggregate_llm_attention, aggregate_vit_attention,
    heterogenous_stack,
    show_mask_on_image
)

from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor

# ===> specify the model path
model_path = "Qwen/Qwen2.5-VL-3B-Instruct"

# load the processor and model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",  # <<< this line is key for extracting the attention weights
    device_map="cuda:0",
)

# ===> specify the image path or url and the prompt text
image_path_or_url = "https://github.com/open-compass/MMBench/blob/main/samples/MMBench/1.jpg?raw=true"
prompt_text = "What python code can be used to generate the output in the image?"

################################################
# Download the image from the URL
response = requests.get(image_path_or_url)
image = Image.open(BytesIO(response.content)).convert("RGB")
image_size = image.size

# Prepare messages in the expected format
messages = [
    {"role": "user", 
    "content": [
        {"type": "image", "image": image}, 
        {"type": "text", "text": prompt_text}]
    }
]

# Use processor to format the chat and process images
chat_text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)

# process the inputs
inputs = processor(
    text=[chat_text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(device)

# print the length of the input and output tokens
print("Number of input tokens: ", len(inputs["input_ids"][0]))
################################################

print(prompt_text)


# generate the response
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=512,
        use_cache=True,
        return_dict_in_generate=True,
        output_attentions=True,
    )

text = processor.decode(outputs["sequences"][0], skip_special_tokens=True)
print(text)

# Below we are aggregating LLM's attention aross heads and layers (simply averaging them).
# See the `aggregate_llm_attention` in `utils.py` for details. 


# lets print some stats about the attention size
print("Number of generated tokens: ", len(outputs.attentions))
print("Number of layers: ", len(outputs.attentions[0]))
print("Number of heads: ", len(outputs.attentions[0][0][0]))

# constructing the llm attention matrix for the input sequence (only attending to attentions[0])
aggregated_prompt_attention = []

for i, layer in enumerate(outputs["attentions"][0]):
    layer_attns = layer.squeeze(0)
    attns_per_head = layer_attns.mean(dim=0)
    cur = attns_per_head[:-1].cpu().clone()
    # following the practice in `aggregate_llm_attention`
    # we are zeroing out the attention to the first <bos> token
    # for the first row `cur[0]` (corresponding to the next token after <bos>), however,
    # we don't do this because <bos> is the only token that it can attend to
    cur[1:, 0] = 0. # TODO: careful here, the <bos> token is inherently use in this particular tokenizer from llama-2
    cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
    aggregated_prompt_attention.append(cur)

aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)

print("Aggregated prompt attention shape: ", aggregated_prompt_attention.shape)

# llm_attn_matrix will be of torch.Size([N, N])
# where N is the total number of input (both image and text ones) + output tokens
llm_attn_matrix = heterogenous_stack(
    [torch.tensor([1])]
    + list(aggregated_prompt_attention) 
    + list(map(aggregate_llm_attention, outputs["attentions"]))
)

print("LLM attention matrix shape: ", llm_attn_matrix.shape)

# visualize the llm attention matrix
# ===> adjust the gamma factor to enhance the visualization
#      higer gamma brings out more low attention values
gamma_factor = 1
enhanced_attn_m = np.power(llm_attn_matrix.numpy(), 1 / gamma_factor)

fig, ax = plt.subplots(figsize=(10, 20), dpi=150)
ax.imshow(enhanced_attn_m, vmin=enhanced_attn_m.min(), vmax=enhanced_attn_m.max(), interpolation="nearest")
plt.savefig('images/llm_attention_matrix.png')
plt.close()


# identify length or index of tokens

image_patch_token_id = 151655
input_ids = inputs["input_ids"][0]
vision_token_indices = (input_ids == image_patch_token_id).nonzero().squeeze(-1)
vision_token_start = vision_token_indices[0].item()
vision_token_end = vision_token_indices[-1].item() + 1
vision_token_len = vision_token_end - vision_token_start

output_token_start = input_ids.shape[0]
output_token_len = outputs["sequences"][0].shape[0] - input_ids.shape[0]
output_token_end = output_token_start + output_token_len


# input_token_len = model.get_vision_tower().num_patches + len(inputs["input_ids"][0]) - 1 # -1 for the <image> token
# vision_token_start = len(tokenizer(prompt.split("<image>")[0], return_tensors='pt')["input_ids"][0])
# vision_token_end = vision_token_start + model.get_vision_tower().num_patches
# output_token_len = len(outputs[0])
# output_token_start = input_token_len
# output_token_end = input_token_len + output_token_len

# look at the attention weights over the vision tokens
overall_attn_weights_over_vis_tokens = []
for i, (row, token) in enumerate(
    zip(
        llm_attn_matrix[vision_token_start:], 
        outputs[0].tolist()
    )
):
    # print(
    #     i + input_token_len, 
    #     f"{tokenizer.decode(token, add_special_tokens=False).strip():<15}", 
    #     f"{row[vision_token_start:vision_token_end].sum().item():.4f}"
    # )

    overall_attn_weights_over_vis_tokens.append(
        row[vision_token_start:vision_token_end].sum().item()
    )
# plot the trend of attention weights over the vision tokens
fig, ax = plt.subplots(figsize=(20, 5))
ax.plot(overall_attn_weights_over_vis_tokens)
ax.set_xticks(range(len(overall_attn_weights_over_vis_tokens)))
ax.set_xticklabels(
    [processor.decode(token, add_special_tokens=False).strip() for token in outputs[0].tolist()],
    rotation=75
)
ax.set_title("at each token, the sum of attention weights over all the vision tokens")
plt.savefig('images/attention_over_vision_tokens.png')
plt.close()

# connect with the vision encoder attention
# to visualize the attention over the image

# vis_attn_matrix will be of torch.Size([N, N])
# where N is the number of vision tokens/patches
# `all_prev_layers=True` will average attention from all layers until the selected layer
# otherwise only the selected layer's attention will be used
# vis_attn_matrix = aggregate_vit_attention(
#     model.get_vision_tower().image_attentions,
#     select_layer=model.get_vision_tower().select_layer,
#     all_prev_layers=True
# )




# Now this has shape [1, heads, N, N] where N = 1 + num_patches
# Compatible with `aggregate_vit_attention`
# Call the function
vis_attn_matrix = aggregate_vit_attention(
    outputs.attentions[0],
    select_layer=-2,
    all_prev_layers=True
)

T, H, W = inputs["image_grid_thw"][0].tolist()
grid_size = H * W
grid_size = 57

num_image_per_row = 8
image_ratio = image_size[0] / image_size[1]
num_rows = output_token_len // num_image_per_row + (1 if output_token_len % num_image_per_row != 0 else 0)

# Increase figure size and DPI for better resolution
fig, axes = plt.subplots(
    num_rows, num_image_per_row, 
    figsize=(20, (20 / num_image_per_row) * image_ratio * num_rows), 
    dpi=300  # Increased DPI for better resolution
)
plt.subplots_adjust(wspace=0.05, hspace=0.2)

# whether visualize the attention heatmap or 
# the image with the attention heatmap overlayed
vis_overlayed_with_attn = True

output_token_inds = list(range(output_token_start, output_token_end))
for i, ax in enumerate(axes.flatten()):
    if i >= output_token_len:
        ax.axis("off")
        continue

    target_token_ind = output_token_inds[i]
    attn_weights_over_vis_tokens = llm_attn_matrix[target_token_ind][vision_token_start:vision_token_end]
    attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.sum()

    attn_over_image = []
    for weight, vis_attn in zip(attn_weights_over_vis_tokens, vis_attn_matrix):
        vis_attn = vis_attn.reshape(6, 19) # TODO: hardcoded for now
        attn_over_image.append(vis_attn * weight)
    attn_over_image = torch.stack(attn_over_image).sum(dim=0)
    attn_over_image = attn_over_image / attn_over_image.max()

    attn_over_image = F.interpolate(
        attn_over_image.unsqueeze(0).unsqueeze(0), 
        size=image.size, 
        mode='nearest', 
    ).squeeze()

    np_img = np.array(image)[:, :, ::-1]
    attn_over_image = attn_over_image.to(torch.float32).T
    img_with_attn, heatmap = show_mask_on_image(np_img, attn_over_image.cpu().numpy())
    
    # Save individual images
    token_text = processor.decode(outputs["sequences"][0][i], add_special_tokens=False).strip()
    # if vis_overlayed_with_attn:
    #    cv2.imwrite(f'images/token_{i}_{token_text}_overlay.png', img_with_attn)
    # else:
    #    cv2.imwrite(f'images/token_{i}_{token_text}_heatmap.png', heatmap)
    
    # Display in subplot
    ax.imshow(heatmap if not vis_overlayed_with_attn else img_with_attn)
    ax.set_title(token_text, fontsize=10, pad=1)  # Increased font size
    ax.axis("off")

# Save the entire figure as a single high-resolution image
plt.savefig('images/all_attention_visualizations.png', bbox_inches='tight', dpi=300)
plt.close()