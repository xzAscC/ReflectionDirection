from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from inference import set_seed, config, Logger
from tqdm import tqdm
import datasets
import torch
import os
import json
import argparse
import numpy as np
import random
args = argparse.ArgumentParser()
args.add_argument(
    "--seed",
    type=int,
    default=42,
    help="The name of the model to use.",
)
args.add_argument(
    "--model_name",
    type=str,
    default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    help="The name of the model to use.",
)
args.add_argument(
    "--dataset_name",
    type=str,
    default="HuggingFaceH4/MATH-500",
    help="The name of the dataset to use.",
)

args.add_argument(
    "--output_dir",
    type=str,
    default="./asset/insert_response/",
    help="The output directory where the model predictions and checkpoints will be written.",
)
args.add_argument(
    "--prompt",
    type=str,
    default="Please reason step by step, and put your final answer within \\boxed{{}}. <|im_start|>user: {problems}<|im_end|>\n<|im_start|>assistant:<think>",
    help="The prompt to use for the model.",
)
    
args.add_argument(
    "--max_new_tokens",
    type=int,
    default=32768,
    help="The maximum number of new tokens to generate.",
)
args = args.parse_args() 
set_seed(args.seed)
output_dir = os.path.join(
    args.output_dir,
    f"responseLength_acc_{args.model_name.split('/')[-1]}_{args.dataset_name.split('/')[-1]}_{len(args.prompt)}_{args.max_new_tokens}",
)
os.makedirs(output_dir, exist_ok=True)

logger = Logger(os.path.join(output_dir, "inference.log"))
logger.write(str(args) + "\n")

model = LLM(
    args.model_name,
    dtype="bfloat16",
)

tok = AutoTokenizer.from_pretrained(args.model_name)

stop_token_ids = tok("<|im_start|><|im_end|>")["input_ids"]
sampling_params = SamplingParams(
    max_tokens=32768,
    min_tokens=0,
    stop_token_ids=stop_token_ids,
    skip_special_tokens=False,
    temperature=0.6,
    top_p=0.95,
)
prompt = args.prompt
inputs = []
model_name = args.model_name
dataset_name = args.dataset_name

dataset = datasets.load_dataset(args.dataset_name)["test"]
for idx, example in enumerate(tqdm(dataset)):
    problem = example["problem"]
    formatted_prompt = prompt.format(problems=str(problem))
    inputs.append(formatted_prompt)
# Generate response using vllm's LLM API
ignore_str = "Wait"

with torch.no_grad():
    for idx, input_ in enumerate(inputs):
        if idx > 1:
            break
        res_length = 16384 - tok.encode(input_, return_tensors="pt").shape[1]
        output = model.generate(input_, sampling_params=sampling_params)
        response = output[0].outputs[0].text
        example = dataset[idx]
        with open(
            os.path.join(
                args.output_dir,
                f"responseLength_acc_{model_name.split('/')[-1]}_{dataset_name.split('/')[-1]}_{len(prompt)}_{args.max_new_tokens}",
                f"{idx}_0.json",
            ),
            "w",
        ) as f:
            json.dump(
                {
                    "problem": problem,
                    "response": response,
                    "answer": example["answer"],
                },
                f,
                indent=2,
            )

        for i in range(10):
            cur_length = tok.encode(input_, return_tensors="pt").shape[1]
            if cur_length <= res_length:
                input_ += output[0].outputs[0].text + ignore_str
                sampling_params = SamplingParams(
                    max_tokens=32768,
                    min_tokens=1,
                    stop_token_ids=stop_token_ids,
                    skip_special_tokens=False,
                    temperature=0.0,
                )
                output = model.generate(input_, sampling_params=sampling_params)
                res_length = res_length - cur_length
                response = output[0].outputs[0].text
                with open(
                    os.path.join(
                        args.output_dir,
                        f"responseLength_acc_{model_name.split('/')[-1]}_{dataset_name.split('/')[-1]}_{len(prompt)}_{args.max_new_tokens}",
                        f"{idx}_{i}.json",
                    ),
                    "w",
                ) as f:
                    json.dump(
                        {
                            "problem": problem,
                            "response": input_ + response,
                            "answer": example["answer"],
                        },
                        f,
                        indent=2,
            )

logger.close()
