from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import datasets
import os
from tqdm import tqdm
import json
import sys
import random
import numpy as np
from datetime import datetime
import vllm
from utils import preprocess_box_response_for_qwen_prompt


def config():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Name of the model to use for inference",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceH4/MATH-500",
        help="Name of the dataset to use for inference",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="""Please reason step by step, and put your final answer within \\boxed{{}}. <|im_start|>user: {problems}<|im_end|>\n<|im_start|>assistant:<think>""",
        help="Prompt to use for inference",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./asset/insert_response/",
        help="Directory to save the inference results",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8196,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling for generation",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Whether to use cache for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dataset_ratio",
        type=float,
        default=0.05,
        help="Ratio of the dataset to use for inference (0.0 to 1.0)",
    )
    parser.add_argument(
        "--only_eval",
        action="store_true",
        help="Whether to only evaluate the model without training",
    )
    parser.add_argument(
        "--injection",
        action="store_true",
        help="Whether to use injection for inference",
    )
    parser.add_argument(
        "--injection_layer",
        type=int,
        default=20,
        help="Layer number for injection",
    )
    parser.add_argument(
        "--injection_alpha",
        type=float,
        default=0.1,
        help="Alpha value for injection",
    )
    args = parser.parse_args()
    return args


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
        self.terminal.write(timestamp + message)
        self.log.write(timestamp + message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def set_seed(seed):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inference_vllm(args, logger):
    # Set random seed for reproducibility
    set_seed(args.seed)
    model_name = args.model_name
    dataset_name = args.dataset_name
    prompt = args.prompt
    model = vllm.LLM(model=model_name, dtype="bfloat16")

    # Load the dataset
    dataset = datasets.load_dataset(dataset_name)["test"]

    os.makedirs(
        os.path.join(
            args.output_dir,
            f"{args.model_name.split('/')[-1]}_{args.dataset_name.split('/')[-1]}_{len(args.prompt)}_{args.max_new_tokens}",
        ),
        exist_ok=True,
    )
    sampling_param = vllm.SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=0.95 if args.do_sample else None,
    )
    score = 0
    boxed = 0
    overall_length = len(dataset) * args.dataset_ratio
    inputs = []
    with torch.no_grad():
        for idx, example in enumerate(tqdm(dataset)):
            problem = example["problem"]
            if args.only_eval:
                with open(
                    os.path.join(
                        args.output_dir,
                        f"{model_name.split('/')[-1]}_{dataset_name.split('/')[-1]}_{len(prompt)}_{args.max_new_tokens}",
                        f"{idx}.json",
                    ),
                    "r",
                ) as f:
                    response = json.load(f)["response"]
                    # answer = example["answer"]
            else:
                formatted_prompt = prompt.format(problems=str(problem))
                inputs.append(formatted_prompt)
        # Generate response using vllm's LLM API
        outputs = model.generate(
            inputs,
            sampling_param,
        )
        for idx, output in enumerate(outputs):
            example = dataset[idx]
            problem = output.prompt
            response = output.outputs[0].text
            with open(
                os.path.join(
                    args.output_dir,
                    f"{model_name.split('/')[-1]}_{dataset_name.split('/')[-1]}_{len(prompt)}_{args.max_new_tokens}",
                    f"{idx}.json",
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

            # Evaluate the response
            _, box_match, box = preprocess_box_response_for_qwen_prompt(
                response, example["answer"]
            )
            score += box_match
            boxed += box
            logger.write(
                f"Problem: {idx}\tResponse: {response}\tAnswer: {example['answer']}\tScore: {box_match}\tBoxed: {box}\n"
            )
    logger.write(f"Model: {model_name}\nDataset: {dataset_name}\nPrompt: {prompt}\n")
    logger.write(f"Total Problems: {overall_length}\nCorrect Answers: {score}\nBoxed: {boxed}\n")
    logger.write(f"Overall Score: {score / overall_length}\n")


class InsertLayer(torch.nn.Module):
    def __init__(self, vector, alpha):
        super(InsertLayer, self).__init__()
        self.vector = vector
        self.alpha = alpha

    def forward(self, x, **kwargs):  # Accept additional keyword arguments
        if self.vector is not None:
            r_hat = self.vector / torch.norm(self.vector, dim=-1, keepdim=True)
            x += self.alpha * torch.matmul(
                r_hat.unsqueeze(-1), r_hat.unsqueeze(-2)
            ).matmul(x.unsqueeze(-1)).squeeze(-1)
        return x


def find_module(block, keywords):
    """
    Try to find a module in a transformer block.
    Args:
        block: Transformer block (nn.Module).
        keywords: List of possible module names (str).
    Returns:
        The found module if found, else None.
    """
    for name, module in block.named_modules():
        if any(keyword in name for keyword in keywords):
            return module
    submodule_names = [name for name, _ in block.named_modules()]
    raise ValueError(f"Could not find keywords {keywords} in: {submodule_names}")


def inference_transformers(args, logger):
    # Set random seed for reproducibility
    set_seed(args.seed)
    model_name = args.model_name
    dataset_name = args.dataset_name
    prompt = args.prompt
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )  # Load the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load the dataset
    dataset = datasets.load_dataset(dataset_name)["test"]

    # dataset = dataset.select(
    #     range(int(len(dataset) * args.dataset_ratio))
    # )  # Select a subset for testing
    os.makedirs(
        os.path.join(
            args.output_dir,
            f"{args.model_name.split('/')[-1]}_{args.dataset_name.split('/')[-1]}_{len(args.prompt)}_{args.injection_layer}_{args.injection_alpha}",
        ),
        exist_ok=True,
    )
    score = 0
    boxed = 0
    overall_length = len(dataset) * args.dataset_ratio
    with torch.no_grad():
        if args.injection:
            mlp_keywords = ["mlp", "feedforward", "ffn"]
            w_wait = (
                torch.load(
                    f"./asset/response/DeepSeek-R1-Distill-Qwen-1.5B_MATH-500_140/DeepSeek-R1-Distill-Qwen-1.5B_hs/before_wait_DeepSeek-R1-Distill-Qwen-1.5B_{args.injection_layer}_-1.pt"
                )
                .cpu()
                .to(torch.float32)
            )
            w_wo_wait = (
                torch.load(
                    f"./asset/response/DeepSeek-R1-Distill-Qwen-1.5B_MATH-500_140/DeepSeek-R1-Distill-Qwen-1.5B_hs/before_wo_wait_DeepSeek-R1-Distill-Qwen-1.5B_{args.injection_layer}_-1.pt"
                )
                .cpu()
                .to(torch.float32)
            )
            insert_vector = w_wait.mean(dim=0) - w_wo_wait.mean(dim=0)

            original_mlp = find_module(model.model.layers[args.injection_layer], mlp_keywords)
            model.model.layers[19].mlp = torch.nn.Sequential(
                original_mlp,
                InsertLayer(insert_vector.to("cuda").to(torch.bfloat16), args.injection_alpha),
            )
            logger.write(str(model) + "\n")
        for idx, example in enumerate(tqdm(dataset)):
            problem = example["problem"]
            if idx >= overall_length:
                break
            if args.only_eval:
                with open(
                    os.path.join(
                        args.output_dir,
                        f"{model_name.split('/')[-1]}_{dataset_name.split('/')[-1]}_{len(prompt)}_{args.injection_layer}_{args.injection_alpha}",
                        f"{idx}.json",
                    ),
                    "r",
                ) as f:
                    response = json.load(f)["response"]
                    # answer = example["answer"]
            else:
                formatted_prompt = prompt.format(problems=str(problem))
                inputs = tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                ).to(model.device)
                # Generate response using transformers API
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature if args.do_sample else 0.0,
                    do_sample=args.do_sample,
                    use_cache=args.use_cache,
                    return_dict_in_generate=True,
                    attention_mask=torch.ones_like(inputs["input_ids"]),
                    pad_token_id=tokenizer.eos_token_id,
                )
                response = tokenizer.decode(
                    outputs.sequences[0], skip_special_tokens=True
                )  # Extract the generated text
                with open(
                    os.path.join(
                        args.output_dir,
                        f"{model_name.split('/')[-1]}_{dataset_name.split('/')[-1]}_{len(prompt)}_{args.injection_layer}_{args.injection_alpha}",
                        f"{idx}.json",
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

            # Evaluate the response
            _, box_match, box = preprocess_box_response_for_qwen_prompt(
                response, example["answer"]
            )
            score += box_match
            boxed += box
            logger.write(
                f"Problem: {idx}\tResponse: {response}\tAnswer: {example['answer']}\tScore: {box_match}\tBoxed: {boxed}\n"
            )
    logger.write(f"Model: {model_name}\nDataset: {dataset_name}\nPrompt: {prompt}\n")
    logger.write(f"Total Problems: {overall_length}\nCorrect Answers: {score}\n")
    logger.write(f"Overall Score: {score / overall_length}\n")

def inference_transformers_s1(args, logger):
    # Set random seed for reproducibility
    set_seed(args.seed)
    model_name = args.model_name
    dataset_name = args.dataset_name
    prompt = args.prompt
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )  # Load the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load the dataset
    dataset = datasets.load_dataset(dataset_name)["test"]

    # dataset = dataset.select(
    #     range(int(len(dataset) * args.dataset_ratio))
    # )  # Select a subset for testing
    os.makedirs(
        os.path.join(
            args.output_dir,
            f"{args.model_name.split('/')[-1]}_{args.dataset_name.split('/')[-1]}_{len(args.prompt)}_{args.injection_layer}_{args.injection_alpha}",
        ),
        exist_ok=True,
    )
    score = 0
    boxed = 0
    overall_length = len(dataset) * args.dataset_ratio
    with torch.no_grad():
        for idx, example in enumerate(tqdm(dataset)):
            problem = example["problem"]
            if idx >= overall_length:
                break
            if args.only_eval:
                with open(
                    os.path.join(
                        args.output_dir,
                        f"{model_name.split('/')[-1]}_{dataset_name.split('/')[-1]}_{len(prompt)}_{args.injection_layer}_{args.injection_alpha}",
                        f"{idx}.json",
                    ),
                    "r",
                ) as f:
                    response = json.load(f)["response"]
                    # answer = example["answer"]
            else:
                formatted_prompt = prompt.format(problems=str(problem))
                inputs = tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                ).to(model.device)
                # Generate response using transformers API
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=args.max_new_tokens,
                    temperature=args.temperature if args.do_sample else 0.0,
                    do_sample=args.do_sample,
                    use_cache=args.use_cache,
                    return_dict_in_generate=True,
                    attention_mask=torch.ones_like(inputs["input_ids"]),
                    pad_token_id=tokenizer.eos_token_id,
                )
                response = tokenizer.decode(
                    outputs.sequences[0], skip_special_tokens=True
                )  # Extract the generated text
                if outputs.sequences[0].shape[0] < 4096:
                    input_prompt = response + "Wait"
                    inputs = tokenizer(
                        input_prompt,
                        return_tensors="pt",
                    ).to(model.device)
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_length=args.max_new_tokens,
                        temperature=args.temperature if args.do_sample else 0.0,
                        do_sample=args.do_sample,
                        use_cache=args.use_cache,
                        return_dict_in_generate=True,
                        attention_mask=torch.ones_like(inputs["input_ids"]),
                        pad_token_id=tokenizer.eos_token_id,
                    )
                with open(
                    os.path.join(
                        args.output_dir,
                        f"{model_name.split('/')[-1]}_{dataset_name.split('/')[-1]}_{len(prompt)}_{args.injection_layer}_{args.injection_alpha}",
                        f"{idx}.json",
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

            # Evaluate the response
            _, box_match, box = preprocess_box_response_for_qwen_prompt(
                response, example["answer"]
            )
            score += box_match
            boxed += box
            logger.write(
                f"Problem: {idx}\tResponse: {response}\tAnswer: {example['answer']}\tScore: {box_match}\tBoxed: {box}\n"
            )
    logger.write(f"Model: {model_name}\nDataset: {dataset_name}\nPrompt: {prompt}\n")
    logger.write(f"Total Problems: {overall_length}\nCorrect Answers: {score}\nBoxed: {boxed}\n")
    logger.write(f"Overall Score: {score / overall_length}\n")


if __name__ == "__main__":
    args = config()
    output_dir = os.path.join(
        args.output_dir,
        f"{args.model_name.split('/')[-1]}_{args.dataset_name.split('/')[-1]}_{len(args.prompt)}_{args.max_new_tokens}_{args.injection_layer}_{args.injection_alpha}",
    )
    os.makedirs(output_dir, exist_ok=True)
    logger = Logger(os.path.join(output_dir, "inference.log"))
    logger.write(str(args) + "\n")
    inference_vllm(args, logger)
    logger.close()
