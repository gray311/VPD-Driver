import os
import json
import torch
import random
import transformers
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from rouge_score import rouge_scorer
import argparse


random.seed(233)

def parse_args():
    parser = argparse.ArgumentParser(description="Testing Script")
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--lora_path', type=str, default=None)
    args = parser.parse_args()
    return args

args = parse_args()

model_path = args.model_path
# model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    # device_map="auto",
)


if args.lora_path is not None:
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
    model.load_state_dict(torch.load(args.lora_path), strict=False)
    model = model.merge_and_unload()


# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model.cuda(1)

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    # device_map="auto",
)

from collections import defaultdict

scores = {
    "rougel": defaultdict(list),
    "llm-eval": defaultdict(list)
}

def llm_eval(gt, pred):
  
    messages = [
        {"role": "system", 
        "content": """You are an evaluator assessing the semantic similarity between two given passages. Your task is to analyze their meanings and determine how closely they align.

Provide a similarity score between 0 and 1, where 1 means the two passages are completely semantically identical, and 0 means they have no semantic similarity at all.
Consider aspects such as meaning, intent, and core information rather than exact wording or phrasing.
If the passages express the same idea but with different wording, they should receive a high score.
If they differ in key details, intent, or implications, assign a lower score accordingly."""
        },
        {"role": "user", "content": """Here are the two passages for evaluation:

Sentence 1: [gt]

Sentence 2: [pred]

Provide your response in the following format:

On the first line, only output the similarity score (a number between 0 and 1).
From the second line onward, explain your reasoning for the given score."""},
    ]

    messages[1]['content'] = messages[1]['content'].replace("[gt]", gt).replace("[pred]", pred)

    outputs = pipeline(
        messages,
        max_new_tokens=128,
    )
    outputs = outputs[0]["generated_text"][-1]['content']

    try:
        # print(outputs)
        return float(outputs.split("\n")[0])
    except:
        print(outputs)
        return 0.0


def model_inference(messages):
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

    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    return output_text[0].split("\n")[-1]


def eval_step(image, question, answer, question_type):
    # image = image[-1]
    if isinstance(image, list):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": image,
                    },
                    {
                        "type": "text", 
                        "text": "What is the visual description of the object [1]? Please describe it in detail."
                    },
                ],
            },
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text", 
                        "text": "What is the visual description of the object [1]? Please describe it in detail."
                    },
                ],
            },
        ]


    gt = answer
    pred = model_inference(messages)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(gt, pred)
    llm_scores = llm_eval(gt, pred)

    print(rouge_scores['rougeL'].precision, llm_scores)
    scores["rougel"][question_type].append(rouge_scores['rougeL'].precision)
    scores["llm-eval"][question_type].append(llm_scores)

"""
python test.py \
    --data_file ./workspace/data/vpd_mix470k_ins-it379k_drivelm91k_V1.json \
    --model_path ./VPD-Driver \
    --lora_path ./models/final_ft_3_epochs_lr5e-05_qwen2.5-vl_retain/step_3600/checkpoint.pt


"""

if __name__ == "__main__":
    with open(args.data_file, "r") as f:
        data = json.load(f)


    from tqdm import tqdm
    for i, line in tqdm(enumerate(data)):
        if "n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915359412465" not in line['image'][-1]: continue
        print(line)
        eval_step(line['image'], line['conversations'][0]['value'], line['conversations'][1]['value'], line['source'])

        break




 
    
