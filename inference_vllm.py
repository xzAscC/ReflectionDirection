from inference import config, Logger, set_seed
import os
import torch
import json
import vllm
import datasets
from tqdm import tqdm
from utils import preprocess_box_response_for_qwen_prompt


def inference_vllm(args, logger):
    # Set random seed for reproducibility
    set_seed(args.seed)
    model_name = args.model_name
    dataset_name = args.dataset_name
    prompt = args.prompt
    model = vllm.LLM(model=model_name, dtype="bfloat16")

    # Load the dataset
    dataset = datasets.load_dataset(dataset_name)["train"]

    os.makedirs(
        os.path.join(
            args.output_dir,
            f"{args.model_name.split('/')[-1]}_{args.dataset_name.split('/')[-1]}_{len(args.prompt)}_{args.max_new_tokens}_{args.injection_layer}_{args.injection_alpha}",
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
            problem = example["Problem"]
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
                    f"{model_name.split('/')[-1]}_{dataset_name.split('/')[-1]}_{len(prompt)}_{args.max_new_tokens}_{args.injection_layer}_{args.injection_alpha}",               
                    f"{idx}.json",
                ),
                "w",
            ) as f:
                json.dump(
                    {
                        "Problem": problem,
                        "response": response,
                        "answer": example["Answer"],
                    },
                    f,
                    indent=2,
                )

        # Evaluate the response
        _, box_match, box = preprocess_box_response_for_qwen_prompt(
            response, example["Answer"]
        )
        score += box_match
        boxed += box
        logger.write(
            f"Problem: {idx}\tResponse: {response}\tAnswer: {example['Answer']}\tScore: {box_match}\tBoxed: {box}\n"
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
