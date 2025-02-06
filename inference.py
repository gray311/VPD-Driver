from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from huggingface_hub import hf_hub_download

model_path = "gray311/VPD-Driver"
checkpoint_path = hf_hub_download("gray311/VPD-Driver", "./checkpoint/checkpoint.pt")
media_type = "video"

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    # device_map="auto",
)


from peft import LoraConfig, get_peft_model

target_modules=r'.*model.*\.(up_proj|k_proj|down_proj|v_proj|q_proj|o_proj|gate_proj)'
config = LoraConfig(
    r=64, 
    lora_alpha=128, 
    target_modules=target_modules, 
    lora_dropout=0.05,
    bias="none", 
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
model.load_state_dict(torch.load(checkpoint_path), strict=False)
model = model.merge_and_unload()


# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")



# you can set your own prompt for testing
prompt = "What is the visual description of the object [1]? Please describe it in detail."
image_path = "https://huggingface.co/gray311/VPD-Driver/resolve/main/data/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915358912465.jpg"
video_path = [
    "https://huggingface.co/gray311/VPD-Driver/resolve/main/data/video/n008-2018-08-31-11-56-46-0400__CAM_FRONT__1535731155162404.jpg",
    "https://huggingface.co/gray311/VPD-Driver/resolve/main/data/video/n008-2018-08-31-11-56-46-0400__CAM_FRONT__1535731155662404.jpg",
    "https://huggingface.co/gray311/VPD-Driver/resolve/main/data/video/n008-2018-08-31-11-56-46-0400__CAM_FRONT__1535731156262404.jpg",
    "https://huggingface.co/gray311/VPD-Driver/resolve/main/data/video/n008-2018-08-31-11-56-46-0400__CAM_FRONT__1535731156762404.jpg",
    "https://huggingface.co/gray311/VPD-Driver/resolve/main/data/video/n008-2018-08-31-11-56-46-0400__CAM_FRONT__1535731157262404.jpg",
    "https://huggingface.co/gray311/VPD-Driver/resolve/main/data/video/n008-2018-08-31-11-56-46-0400__CAM_FRONT__1535731157762404.jpg",
    "https://huggingface.co/gray311/VPD-Driver/resolve/main/data/video/n008-2018-08-31-11-56-46-0400__CAM_FRONT__1535731158262404.jpg",
    "https://huggingface.co/gray311/VPD-Driver/resolve/main/data/video/n008-2018-08-31-11-56-46-0400__CAM_FRONT__1535731158762404.jpg",
]

if media_type == "image":
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text", 
                    "text": prompt
                },
            ],
        },
    ]
elif media_type == "video":
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                },
                {
                    "type": "text", 
                    "text": prompt
                },
            ],
        },
    ]

model.cuda()
# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text)


