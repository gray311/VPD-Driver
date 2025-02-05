from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model_path = "./models/final_ft_10_epochs_lr1e-05_qwen2.5-vl_retain/step_1800"
model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    # device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

image_path = "n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915358912465.jpg"
# image_path = "./EST/positive_face/generated.photos_0139935.png"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {"type": "text", "text": "What is the visual description of the object [1]?"},
        ],
    },
]
model.cuda(2)
# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False
)

print(text)


text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print(text)


image_inputs, video_inputs = process_vision_info(messages)


inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

print(model.device)

inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=512)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
