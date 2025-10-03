import os
import sys
from functools import partial

sys.path.append("./models")
import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import torch
import torch.nn.functional as F

from utils import (
    load_image,
    aggregate_llm_attention,
    aggregate_vit_attention,
    heterogenous_stack,
    show_mask_on_image,
)

try:
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
    )
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import (
        process_images,
        tokenizer_image_token,
        get_model_name_from_path,
    )
except:
    raise ImportError("LLaVA-NeXT not found, make sure to install it -> https://github.com/GuilhermeViveiros/LLaVA-NeXT.git")

# Create output directory for saved images
# get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "images")
os.makedirs(output_dir, exist_ok=True)

# ===> specify the model path
model_path = "/mnt/scratch-artemis/gviveiros/TowerVision/llava-next-native/towerp_2b_base_full/"

# load the model
device = "cuda" if torch.cuda.is_available() else "cpu"

disable_torch_init()

model_name = get_model_name_from_path(model_path)

print("Model name: ", model_name)
print("Model path: ", model_path)

llava_args = {
    "multimodal": True,
    "attn_implementation": "eager" #"sdpa" if torch.version.cuda and torch.__version__ >= "2.1.2" else "eager"
}

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    None,
    model_name,
    device_map=device,
    torch_dtype="bfloat16",
    **llava_args
)
model.to(device)



# ===> specify the image path or url and the prompt text
image_path_or_url = (
    "https://github.com/open-compass/MMBench/blob/main/samples/MMBench/1.jpg?raw=true"
)
prompt_text = "What code can be used to generate the output in the image?"

################################################
# preparation for the generation
# unlikely that you need to change anything here
if "llama-2" in model_name.lower():
    conv_mode = "llava_llama_2"
elif "mistral" in model_name.lower():
    conv_mode = "mistral_instruct"
elif "v1.6-34b" in model_name.lower():
    conv_mode = "chatml_direct"
elif "v1" in model_name.lower():
    conv_mode = "llava_v1"
elif "mpt" in model_name.lower():
    conv_mode = "mpt"
elif "tower" in model_name.lower():
    conv_mode = "gemma2_instruct"
else:
    conv_mode = "llava_v0"

conv = conv_templates[conv_mode].copy()
if "mpt" in model_name.lower():
    roles = ("user", "assistant")
else:
    roles = conv.roles

image = load_image(image_path_or_url)
image_size = image.size
print("Image size: ", image_size)
# image_tensor, images = process_images([image], image_processor, model.config)
images = process_images([image], image_processor, model.config)
# image = images[0]
if type(images) is list:
    image_tensor = [
        image.to(model.device, dtype=torch.float16) for image in images
    ]
else:
    image_tensor = images.to(model.device, dtype=torch.bfloat16)

if model.config.mm_use_im_start_end:
    inp = (
        DEFAULT_IM_START_TOKEN
        + DEFAULT_IMAGE_TOKEN
        + DEFAULT_IM_END_TOKEN
        + "\n"
        + prompt_text
    )
else:
    inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text

conv.append_message(conv.roles[0], inp)
conv.append_message(conv.roles[1], None)

prompt = conv.get_prompt()

def pad_sequence(input_ids, batch_first, padding_value):
    if tokenizer.padding_side == "left":
        input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
    if tokenizer.padding_side == "left":
        input_ids = torch.flip(input_ids, [1])
    return input_ids

input_ids = (
    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    .unsqueeze(0)
    .to(model.device)
)
################################################

# pad the input ids
pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_ids).to(device)
attention_masks = input_ids.ne(pad_token_ids).to(device)
attention_masks = attention_masks.to(device)

# carefull here, ensure that no system prompt is added
print("Prompt: ", prompt)

pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
# ---
# generate the response
torch._inductor.cudagraph_mark_step_begin()
with torch.inference_mode():
    outputs = model.generate(
        input_ids,
        attention_mask=attention_masks,
        images=image_tensor,
        image_sizes=[image_size],
        do_sample=False,
        max_new_tokens=512,
        eos_token_id=107,
        pad_token_id=pad_token_ids,
        use_cache=True,
        return_dict_in_generate=True,
        output_attentions=True,
    )

# lets save the outputs
# outputs_file = os.path.join(output_dir, "outputs.pkl")
# with open(outputs_file, "wb") as f:
#     pickle.dump(outputs["sequences"].detach().cpu().numpy(), f)
#     pickle.dump(outputs["attentions"].detach().cpu().numpy(), f)

# import pdb; pdb.set_trace()
# # read back the outputs
# with open(outputs_file, "rb") as f:
#     outputs["sequences"] = torch.from_numpy(pickle.load(f)).to(device)
#     outputs["attentions"] = torch.from_numpy(pickle.load(f)).to(device)

text = tokenizer.decode(outputs["sequences"][0]).strip()
# how much tokens on the output tokens?
print("Number of output tokens: ", len(outputs["sequences"][0]))
print(text)

# ---
print("Constructing the LLM attention matrix...")
print("Number of layers: ", len(outputs["attentions"][0]))
for i, token in enumerate(outputs["attentions"]):
    print(f"Token {i} shape: {token[0].shape}")
    # gemma2 uses maximum context length during the generation (4096 tokens)


# FIXME
# gemma2 uses sliding window & global attention, so we need to pad the vectors into max_length
# lets replace outputs[0] by a new version padded with maxsize
max_length = max(attention.shape[-1] for attention in outputs["attentions"][0])
new_attentions = []
# clone the attentions and detach them
for idx, attention in enumerate(outputs["attentions"]): # for each output token
    padded_attentions = []
    for idx_,layer_attn in enumerate(attention):
        pad_len = max_length - layer_attn.shape[-1]
        if pad_len > 0:
            pad_shape = (*layer_attn.shape[:-1], pad_len)
            padding = torch.zeros(pad_shape, device=layer_attn.device, dtype=layer_attn.dtype)
            padded_attention = torch.cat([layer_attn, padding], dim=-1)
        else:
            padded_attention = layer_attn
        padded_attentions.append(padded_attention)
    new_attentions.append(padded_attentions)

outputs["padded_attentions"] = tuple(new_attentions)


# constructing the llm attention matrix
aggregated_prompt_attention = []
for i, layer in enumerate(outputs["padded_attentions"][0]):
    layer_attns = layer.squeeze(0)
    attns_per_head = layer_attns.mean(dim=0)
    cur = attns_per_head[:-1].cpu().clone()
    
    # following the practice in `aggregate_llm_attention`
    # we are zeroing out the attention to the first <bos> token
    # for the first row `cur[0]` (corresponding to the next token after <bos>), however,
    # we don't do this because <bos> is the only token that it can attend to
    cur[1:, 0] = 0.0
    cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
    aggregated_prompt_attention.append(cur)

print("Attention for every layer aggregated, for the prompt attention sequence")
print("We have", len(aggregated_prompt_attention), "attention heads")
print(
    "Each attention head is of shape: ",
    aggregated_prompt_attention[0].shape,
    " we removed the last row because it is the <eos> token",
)

aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)
print("Aggregated prompt attention shape: ", aggregated_prompt_attention.shape)


# llm_attn_matrix will be of torch.Size([N, N])
# where N is the total number of input (both image and text ones) + output tokens
llm_attn_matrix = heterogenous_stack(
    [torch.tensor([1])]
    + list(aggregated_prompt_attention)
    + list(map(aggregate_llm_attention, outputs["padded_attentions"]))
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
input_token_len = (
    model.get_vision_tower().num_patches + len(input_ids[0]) - 1
)  # -1 for the <image> token

vision_token_start = len(
    tokenizer(prompt.split("<image>")[0], return_tensors="pt")["input_ids"][0]
)
vision_token_end = vision_token_start + model.get_vision_tower().num_patches
visual_token_len = vision_token_end - vision_token_start
output_token_len = len(outputs["sequences"][0])
output_token_start = input_token_len
output_token_end = input_token_len + output_token_len

# ---
import pdb; pdb.set_trace()
# look at the attention weights over the vision tokens
overall_attn_weights_over_vis_tokens = []
for i, (row, token) in enumerate(
    zip(llm_attn_matrix[input_token_len:], outputs["sequences"][0].tolist())
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
    [
        tokenizer.decode(token, add_special_tokens=False).strip()
        for token in outputs["sequences"][0].tolist()
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

# connect with the vision encoder attention
# to visualize the attention over the image

# vis_attn_matrix will be of torch.Size([N, N])
# where N is the number of vision tokens/patches
# `all_prev_layers=True` will average attention from all layers until the selected layer
# otherwise only the selected layer's attention will be used

import pdb; pdb.set_trace()
vis_attn_matrix = aggregate_vit_attention(
    model.get_vision_tower().image_attentions,
    select_layer=model.get_vision_tower().select_layer,
    all_prev_layers=True,
)
grid_size = model.get_vision_tower().num_patches_per_side

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

    attn_over_image = []
    
    for weight, vis_attn in zip(attn_weights_over_vis_tokens, vis_attn_matrix):
        vis_attn = vis_attn.reshape(grid_size, grid_size)
        # vis_attn = vis_attn / vis_attn.max()
        attn_over_image.append(vis_attn * weight)
    
    attn_over_image = torch.stack(attn_over_image).sum(dim=0)
    attn_over_image = attn_over_image / attn_over_image.max()



    attn_over_image = F.interpolate(
        attn_over_image.unsqueeze(0).unsqueeze(0),
        size=( image_size[1], image_size[0] ),
        mode="nearest",
        # mode='bicubic', align_corners=False
    ).squeeze()
    
    # lets change the attn_over_image to height x width
    # attn_over_image = attn_over_image.transpose(1, 0)
   
    
    np_img = np.array(image)[:, :, ::-1]
    img_with_attn, heatmap = show_mask_on_image(np_img, attn_over_image.numpy())
    ax.imshow(heatmap if not vis_overlayed_with_attn else img_with_attn)
    ax.set_title(
        tokenizer.decode(outputs["sequences"][0][i], add_special_tokens=False).strip(),
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
