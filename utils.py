import datasets
import torch
import os
import json
import huggingface_hub as hf_hub
from typing import Tuple, List
from collections import Counter, defaultdict
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from torch.nn import DataParallel
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from math_grader import boxed_reward_fn
import numpy as np
import umap
from matplotlib.gridspec import GridSpec


def numerical_rank(A):
    # Compute singular values
    # A = torch.stack(A).to(torch.float32)
    singular_values = torch.linalg.svdvals(A.to(torch.float32))
    singular_values = singular_values / singular_values.sum()
    # Compute the numerical rank formula
    num_rank = (singular_values.sum() ** 2) / (singular_values.square().sum())

    return num_rank.item()


def plot_numerical_rank():
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(29),
        [
            numerical_rank(
                torch.load(
                    f"../asset/response/DeepSeek-R1-Distill-Qwen-1.5B_MATH-500_140/hs/before_wait_Qwen2.5-1.5B_{l}_-1.pt"
                )
            )
            for l in range(29)
        ],
        label="qwen_w_wait",
    )
    plt.plot(
        range(29),
        [
            numerical_rank(
                torch.load(
                    f"../asset/response/DeepSeek-R1-Distill-Qwen-1.5B_MATH-500_140/hs/before_wo_wait_DeepSeek-R1-Distill-Qwen-1.5B_{l}_-1.pt"
                )
            )
            for l in range(29)
        ],
        label="ds_wo_wait",
    )
    plt.plot(
        range(29),
        [
            numerical_rank(
                torch.load(
                    f"../asset/response/DeepSeek-R1-Distill-Qwen-1.5B_MATH-500_140/hs/before_wait_DeepSeek-R1-Distill-Qwen-1.5B_{l}_-1.pt"
                )
            )
            for l in range(29)
        ],
        label="ds_w_wait",
    )
    plt.plot(
        range(29),
        [
            numerical_rank(
                torch.load(
                    f"../asset/response/DeepSeek-R1-Distill-Qwen-1.5B_MATH-500_140/hs/before_wo_wait_Qwen2.5-1.5B_{l}_-1.pt"
                )
            )
            for l in range(29)
        ],
        label="qwen_wo_wait",
    )
    plt.xlabel("Layer")
    plt.ylabel("Numerical Rank")
    plt.title("Numerical Rank Across Layers")
    plt.legend()
    plt.show()


def test_prompt_boxed(
    prompt: str, model_name: str, dataset_name: str, save_respnses: bool = True
) -> tuple[int, int, int]:
    """
    Test the performance of a prompt by checking how many answers are in the boxed format.

    Args:
        prompt (str): The input prompt to the model.
        model_name (str): The name or path of the model to test.
        tokenizer_name (str): The name or path of the tokenizer to use.

    Returns:
        float: The percentage of answers in the boxed format.
    """

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    )
    model = DataParallel(model)  # Enable multi-GPU support
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the dataset
    dataset = datasets.load_dataset(dataset_name)["test"]

    # dataset = dataset.select(
    #     range(int(len(dataset) * 1e-2))
    # )  # Select a subset for testing
    dataset_length = len(dataset)
    model_name = model_name.split("/")[-1]
    dataset_name = dataset_name.split("/")[-1]
    if save_respnses:
        os.makedirs(
            f"./asset/insert_response/{model_name}_{dataset_name}_{len(prompt)}",
            exist_ok=True,
        )
    with torch.no_grad():

        for idx, problem in enumerate(tqdm(dataset["problem"])):
            formatted_prompt = prompt.format(problems=str(problem))
            input_ids = tokenizer(formatted_prompt, return_tensors="pt")[
                "input_ids"
            ].to("cuda:0")

            outputs = model.module.generate(  # Use model.module for DataParallel
                input_ids,
                max_new_tokens=8196,
                use_cache=True,
                temperature=0.7,
                do_sample=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                attention_mask=torch.ones_like(input_ids),
                pad_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            with open(
                f"./asset/insert_response/{model_name}_{dataset_name}_{len(prompt)}/{idx}.json",
                "w",
            ) as f:
                json.dump({"problem": problem, "response": response}, f, indent=2)

    return dataset_length


def extract_key_token(
    response_dir: str,
    model_name: str,
    token_position: int = -2,
    layer: int = 9,
    token_list: list[int] = [382],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract key token embeddings from model responses.

    Args:
        response_path (str): Path to the response files
        model_name (str): Name of the model to load
        token_position (int): Position of the token to extract

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Extracted token embeddings and their labels
    """
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    last_token_before_wait = []
    last_token_before_wo_wait = []
    for idx, path in enumerate(tqdm(os.listdir(response_dir))):
        if not path.endswith(".json"):
            continue
        with open(os.path.join(response_dir, path), "r") as f:
            response = json.load(f)
        # layers = len(hidden_states[-1])
        input_ids = tokenizer(response["response"], return_tensors="pt")[
            "input_ids"
        ].to(
            "cuda:0"
        )  # shape: (1, seq_length)
        problem_length = tokenizer(response["problem"], return_tensors="pt")[
            "input_ids"
        ].shape[1]

        input_length = input_ids.shape[1]
        wait_word = ["wait", "Wait", " wait", " Wait"]
        wait_list = []
        outs = model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            do_sample=True,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            attention_mask=torch.ones_like(input_ids),
            pad_token_id=tokenizer.eos_token_id,
        )  # shape: (1, layers, outputs_len, seq_length, dim)

        wait_list = tokenizer(wait_word, return_tensors="pt")["input_ids"][:, -1]

        indices = []
        for word in wait_list:
            index = (input_ids[0] == word.item()).nonzero().squeeze()
            if index.dim() == 0:  # if it's a scalar, add a dimension
                index = index.unsqueeze(0)
            indices.append(index)
        res = torch.cat(indices)
        for idy in res:
            last_token_before_wait.append(
                outs.hidden_states[0][layer][0][idy + token_position]
            )
        for token in token_list:
            index = (input_ids[0] == token).nonzero().squeeze()
            if index.dim() == 0:  # if it's a scalar, add a dimension
                index = index.unsqueeze(0)
            for i in range(index.shape[0]):
                input_length = input_ids[0].shape[0]
                if index[i] + 50 > input_length:
                    search_end_index = input_length
                else:
                    search_end_index = index[i] + 50
                flag = False
                for word in wait_list:
                    if word in input_ids[0][index[i] : search_end_index]:
                        flag = True
                        break
                if not flag:
                    if index[i] + 10 >= input_length:
                        continue
                    last_token_before_wo_wait.append(
                        outs.hidden_states[0][layer][0][index[i] + token_position]
                    )
    short_model_name = model_name.split("/")[-1]
    hs_dir = os.path.join(
        response_dir, f"{short_model_name}_hs"
    )  # Create a directory path without leading slash
    os.makedirs(hs_dir, exist_ok=True)

    wait_path = os.path.join(
        hs_dir, f"before_wait_{short_model_name}_{layer}_{token_position}.pt"
    )
    wo_wait_path = os.path.join(
        hs_dir, f"before_wo_wait_{short_model_name}_{layer}_{token_position}.pt"
    )

    torch.save(torch.stack(last_token_before_wait, dim=0), wait_path)
    torch.save(torch.stack(last_token_before_wo_wait, dim=0), wo_wait_path)
    return last_token_before_wait, last_token_before_wo_wait


def numerical_rank(A):
    # Compute singular values
    singular_values = torch.linalg.svdvals(A)
    singular_values = singular_values / singular_values.sum()
    # Compute the numerical rank formula
    num_rank = (singular_values.sum() ** 2) / (singular_values.square().sum())

    return num_rank


def neumann_entropy(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute the Neumann entropy of a matrix using PyTorch.
    Args:
        matrix: Matrix (torch.Tensor).
    Returns:
        Neumann entropy of the matrix as a torch.Tensor.
    """
    # matrix = normalized_empirical_covariance(matrix)
    s = torch.linalg.svdvals(matrix)
    # threshold_ = s[0] * 1e-5
    # s = s[s > threshold_]
    normalized_s = s / s.sum()
    return -torch.sum(
        normalized_s
        * (
            torch.log(normalized_s)
            / torch.log(torch.tensor(3584, device=normalized_s.device))
        )
    )


def plot_umap_embeddings(
    layers, asset_path, n_neighbors=10, min_dist=0.1, random_state=42, figsize=(20, 20)
):
    """
    Plots UMAP embeddings for given layers.

    Parameters:
        layers (int): Number of layers to process.
        asset_path (str): Path to the asset directory.
        n_neighbors (int): Number of neighbors for UMAP.
        min_dist (float): Minimum distance for UMAP.
        random_state (int): Random state for reproducibility.
        figsize (tuple): Size of the figure.
    """

    fig, axes = plt.subplots(8, 4, figsize=figsize)
    axes = axes.flatten()
    colors = {
        "wait token": "red",
        "no wait token": "orange",
    }

    for layer in tqdm(range(layers)):
        token_before_wait = torch.load(
            f"{asset_path}/hs/before_wait_Qwen2.5-1.5B_{layer}.pt"
        )
        token_wo_wait = torch.load(
            f"{asset_path}/hs/before_wo_wait_Qwen2.5-1.5B_{layer}.pt"
        )

        reducer = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state
        )
        embeddings = reducer.fit_transform(
            np.concatenate(
                [
                    token_before_wait.cpu().to(torch.float32).numpy(),
                    token_wo_wait.cpu().to(torch.float32).numpy(),
                ]
            )
        )

        wait_labels = ["wait token"] * token_before_wait.shape[0]
        nonwait_labels = ["no wait token"] * token_wo_wait.shape[0]
        all_labels = wait_labels + nonwait_labels

        for label in set(all_labels):
            indices = [i for i, x in enumerate(all_labels) if x == label]
            axes[layer].scatter(
                embeddings[indices, 0],
                embeddings[indices, 1],
                label=label,
                alpha=0.5,
                s=10,
                color=colors[label],
            )
        axes[layer].set_title(f"Layer {layer}")
        axes[layer].set_xlabel("UMAP Dimension 1")
        axes[layer].set_ylabel("UMAP Dimension 2")
        axes[layer].legend()
        axes[layer].grid(True)

    plt.tight_layout()
    plt.show()


def run_test_prompt_boxed():
    for model_name in [
        "Qwen/Qwen2.5-1.5B",
        # "Qwen/Qwen2.5-7B",
        # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        # "meta-llama/Llama-3.1-8B",
    ]:
        if model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":
            test_prompt_boxed(
                prompt="""Please reason step by step, and put your final answer within \\boxed{{}}.<|im_start|>user: {problems}<|im_end|>\n<|im_start|>assistant: <think>""",
                model_name=model_name,
                dataset_name="HuggingFaceH4/MATH-500",
                save_respnses=True,
            )
        else:
            test_prompt_boxed(
                prompt="""Please reason step by step, and put your final answer within \\boxed{{}}.<|im_start|>user: {problems}<|im_end|>\n<|im_start|>assistant:""",
                model_name=model_name,
                dataset_name="HuggingFaceH4/MATH-500",
                save_respnses=True,
            )


def average_response_length(
    response_dir: str,
    model_name: str,
    keywords_list: List[str] = [
        "wait",
        "re-check",
        "recheck",
        "rethink",
        "re-think",
        "reconsider",
        "re-consider",
        "re-evaluat",
        "reevaluat",
        "rethink",
        "re-think",
        "re-examine",
        "reexamine",
        "check again",
        "try again",
        "think again",
        "consider again",
        "evaluate again",
        "examine again",
    ],
) -> float:
    """
    Calculate the average response length.

    Args:
        response_path (str): Path to the response files
        model_name (str): Name of the model to load

    Returns:
        float: Average response length
    """
    total_length_with_wait = 0
    total_length_without_wait = 0
    total_length = 0
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    wait_numbers = 0
    problem_numbers = 0
    for idx, path in enumerate(tqdm(os.listdir(response_dir))):
        if not path.endswith(".json"):
            continue
        with open(os.path.join(response_dir, path), "r") as f:
            response = json.load(f)
        assistant_response = response["response"].split("<|im_start|>assistant:")[-1]
        input_ids = tokenizer(assistant_response, return_tensors="pt")["input_ids"].to(
            "cuda:0"
        )

        if any(keyword in assistant_response.lower() for keyword in keywords_list):
            total_length_with_wait += input_ids.shape[1]
            wait_numbers += 1
        else:
            total_length_without_wait += input_ids.shape[1]
        total_length += input_ids.shape[1]
        problem_numbers += 1
    if wait_numbers == 0:
        return (
            total_length / problem_numbers,
            0,
            total_length_without_wait / (problem_numbers - wait_numbers),
            wait_numbers,
            problem_numbers,
        )
    elif problem_numbers == wait_numbers:
        return (
            total_length / problem_numbers,
            total_length_with_wait / wait_numbers,
            0,
            wait_numbers,
            problem_numbers,
        )
    else:
        return (
            total_length / problem_numbers,
            total_length_with_wait / wait_numbers,
            total_length_without_wait / (problem_numbers - wait_numbers),
            wait_numbers,
            problem_numbers,
        )


def run_average_response_length():
    response_dirs = [
        "./asset/response/Qwen2.5-1.5B_MATH-500_133",
        "./asset/response/Qwen2.5-7B_MATH-500_133",
        "./asset/response/DeepSeek-R1-Distill-Qwen-7B_MATH-500_141",
        "./asset/response/DeepSeek-R1-Distill-Qwen-1.5B_MATH-500_140",
    ]
    for response_dir in response_dirs:
        model_name = response_dir.split("/")[-1]
        if "DeepSeek" in model_name:
            model_name = f"deepseek-ai/{model_name.split('_')[0]}"
        else:
            model_name = f"Qwen/{model_name.split('_')[0]}"
        tmp = average_response_length(
            response_dir=response_dir,
            model_name=model_name,
        )
        print(tmp)


def wait_number(
    response_dir: str,
    model_name: str,
    keywords_list: List[str] = [
        "wait",
        "re-check",
        "recheck",
        "rethink",
        "re-think",
        "reconsider",
        "re-consider",
        "re-evaluat",
        "reevaluat",
        "rethink",
        "re-think",
        "re-examine",
        "reexamine",
        "check again",
        "try again",
        "think again",
        "consider again",
        "evaluate again",
        "examine again",
    ],
) -> list[int]:
    """
    Determine whether the keyword 'wait' appears in the <think> process or after it.

    Args:
        response_dir (str): Path to the response files
        model_name (str): Name of the model to load

    Returns:
        list[int]: Positions of the keyword 'wait' in the responses
    """
    wait_number = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for idx, path in enumerate(tqdm(os.listdir(response_dir))):
        if not path.endswith(".json"):
            continue
        with open(os.path.join(response_dir, path), "r") as f:
            response = json.load(f)
        assistant_response = response["response"].split("<|im_start|>assistant:")[-1]
        input_ids = tokenizer(assistant_response, return_tensors="pt")["input_ids"].to(
            "cuda:0"
        )
        word_count = 0
        for word in keywords_list:
            word_count += assistant_response.lower().count(word)
        wait_number.append(
            [
                word_count,
                input_ids.shape[1],
            ]
        )
    return wait_number


def run_wait_number():
    response_dirs = [
        # "./asset/response/Qwen2.5-1.5B_MATH-500_133",
        # "./asset/response/Qwen2.5-7B_MATH-500_133",
        "./asset/response/DeepSeek-R1-Distill-Qwen-7B_MATH-500_141",
        "./asset/response/DeepSeek-R1-Distill-Qwen-1.5B_MATH-500_140",
    ]
    for response_dir in response_dirs:
        model_name = response_dir.split("/")[-1]
        if "DeepSeek" in model_name:
            model_name = f"deepseek-ai/{model_name.split('_')[0]}"
        else:
            model_name = f"Qwen/{model_name.split('_')[0]}"
        tmp = wait_number(
            response_dir=response_dir,
            model_name=model_name,
        )
        print(tmp)


def plot_wait_number_vs_response_length(response_dir: str, model_name: str):
    wait_data = wait_number(response_dir, model_name)
    wait_counts = [item[0] for item in wait_data]
    response_lengths = [item[1] for item in wait_data]

    plt.figure(figsize=(10, 6))
    plt.scatter(response_lengths, wait_counts, alpha=0.6, edgecolors="k")
    plt.title("Relationship Between Wait Number and Response Length")
    plt.xlabel("Response Length")
    plt.ylabel("Wait Number")
    plt.grid(True)
    plt.show()


def compute_correlation(wait_data):
    """
    Compute the Pearson correlation coefficient between response lengths and wait counts.

    Args:
        wait_data (list[list[int]]): A list of [wait_count, response_length] pairs.

    Returns:
        tuple: Pearson correlation coefficient and p-value.
    """
    wait_counts = [item[0] for item in wait_data]
    response_lengths = [item[1] for item in wait_data]

    # Calculate Pearson correlation coefficient
    correlation, p_value = pearsonr(response_lengths, wait_counts)

    return correlation, p_value


def extract_keywords(
    keywords_list: List[str] = [
        "wait",
        "re-check",
        "recheck",
        "rethink",
        "re-think",
        "reconsider",
        "re-consider",
        "re-evaluat",
        "reevaluat",
        "rethink",
        "re-think",
        "re-examine",
        "reexamine",
        "check again",
        "try again",
        "think again",
        "consider again",
        "evaluate again",
        "examine again",
    ],
    response_dir: str = "./reflect_responses",
):
    """
    Extracts keywords from responses stored in JSON files within the specified directory.

    Args:
        keywords_list (List[str], optional): A list of keywords to search for in the responses.
            Defaults to [
                "wait", "re-check", "recheck", "rethink", "re-think", "reconsider",
                "re-consider", "re-evaluat", "reevaluat", "rethink", "re-think",
                "re-examine", "reexamine", "check again", "try again", "think again",
                "consider again", "evaluate again", "examine again",
            ].
        response_dir (str, optional): The directory containing JSON response files.
            Defaults to "./reflect_responses".

    Returns:
        dict: A dictionary where keys are keywords and values are their respective counts
            in the responses.
    """
    # most keywords only appear in responses containing the word "wait"
    # Moreover, we observe that the majority of these instances involve the word "wait" preceding other keywords.
    # Furthermore, nearly all identified keywords co-occur with the word "wait" within the same sentence.
    keywords = []
    for idx, response_file in enumerate(tqdm(os.listdir(response_dir))):
        if not response_file.endswith(".json"):
            continue
        with open(os.path.join(response_dir, response_file), "r") as f:
            response = json.load(f)["response"]
            sentences = re.split(r"(?<=[.!?:])\s+", response)
            for idy, sentence in enumerate(sentences):
                for keyword in keywords_list:
                    if keyword in sentence.lower():
                        keywords.append(keyword)

    return (dict(Counter(keywords)),)


def run_extract_keywords():

    response_dirs = [
        "asset/response/Qwen2.5-1.5B_MATH-500_133",
        "asset/response/Qwen2.5-7B_MATH-500_133",
        "asset/response/DeepSeek-R1-Distill-Qwen-7B_MATH-500_141",
        "asset/response/DeepSeek-R1-Distill-Qwen-1.5B_MATH-500_140",
    ]
    for response_dir in response_dirs:
        keywords_count = extract_keywords(response_dir=response_dir)
    print(keywords_count)


def preprocess_box_response_for_qwen_prompt(sequence, answer):
    model_output = re.sub(
        r"^.*?<\|im_start\|>assistant",
        "<|im_start|>assistant",
        sequence,
        flags=re.DOTALL,
        count=1,
    )
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    if "boxed" in model_output:
        boxed = 1
    else:
        boxed = 0
    # grader
    _, box_match = boxed_reward_fn(model_response=model_output, gt_answer=answer)

    return "", box_match, boxed


def grader_w_keywords(response_dir: str):
    dataset = datasets.load_dataset("HuggingFaceH4/MATH-500")["test"]
    score = 0
    boxed = 0
    for idx, example in enumerate(tqdm(dataset)):
        with open(os.path.join(response_dir, f"{idx}.json"), "r") as f:
            response = json.load(f)["response"]
        _, box_match, box = preprocess_box_response_for_qwen_prompt(
            response, example["answer"]
        )
        score += box_match
        boxed += box
    return score / boxed


def plot_thinking_time_vs_accuracy():
    # Sample data - replace with your actual data
    thinking_times = {
        'DeepSeek-1.5B': [7220.81, 16211.72, 24777.92],
        'Qwen-1.5B': [1740.9, 2908.27, 7409.45]
    }
    
    accuracies = {
        'DeepSeek-1.5B': [83.2, 84.7, 85.6],
        'Qwen-1.5B': [17, 26, 39]
    }
    
    # Create figure with broken y-axis
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.05)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Plot data in both subplots
    ax1.plot(thinking_times['DeepSeek-1.5B'], accuracies['DeepSeek-1.5B'], 
             marker='o', linestyle='-', label='DeepSeek-1.5B')
    ax1.plot(thinking_times['Qwen-1.5B'], accuracies['Qwen-1.5B'], 
             marker='s', linestyle='-', label='Qwen-1.5B')
    
    ax2.plot(thinking_times['DeepSeek-1.5B'], accuracies['DeepSeek-1.5B'], 
             marker='o', linestyle='-', label='DeepSeek-1.5B')
    ax2.plot(thinking_times['Qwen-1.5B'], accuracies['Qwen-1.5B'], 
             marker='s', linestyle='-', label='Qwen-1.5B')
    
    # Set y-axis limits for the desired ranges
    ax1.set_ylim(80, 90)
    ax2.set_ylim(10, 40)
    
    # Hide the spines between ax1 and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(labelbottom=False)
    ax2.xaxis.tick_bottom()
    
    # Add diagonal lines to indicate broken y-axis
    d = 0.01
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    
    # Add labels and legend
    ax2.set_xlabel('Average Thinking Time (s)')
    fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical')
    fig.suptitle('Accuracy vs. Average Thinking Time', fontsize=14)
    ax1.legend()
    
    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    
    return fig


def run_grader_w_keywords():
    response_dirs = [
        # "./asset/response/Qwen2.5-1.5B_MATH-500_133",
        # "./asset/response/Qwen2.5-7B_MATH-500_133",
        # "./asset/response/DeepSeek-R1-Distill-Qwen-1.5B_MATH-500_458",
        # "./asset/response/DeepSeek-R1-Distill-Qwen-7B_MATH-500_141",
        "./asset/response/DeepSeek-R1-Distill-Qwen-1.5B_MATH-500_140",
    ]
    dataset = datasets.load_dataset("HuggingFaceH4/MATH-500")["test"]
    for response_dir in response_dirs:
        model_name = response_dir.split("/")[-1]
        if "DeepSeek" in model_name:
            model_name = f"deepseek-ai/{model_name.split('_')[0]}"
        else:
            model_name = f"Qwen/{model_name.split('_')[0]}"
        wait_stat = wait_number(
            response_dir=response_dir,
            model_name=model_name,
        )
    score_stat = []
    for idx, example in enumerate(tqdm(dataset)):
        with open(os.path.join(response_dir, f"{idx}.json"), "r") as f:
            response = json.load(f)["response"]
        _, box_match, box = preprocess_box_response_for_qwen_prompt(
            response, example["answer"]
        )
        score_stat.append([wait_stat[idx][0], wait_stat[idx][1], box_match, box])
    score = 0
    boxed = 0
    for itm in score_stat:
        if itm[2] == 1:
            score += 1
            
        if itm[3] == 1:
            boxed += 1
    print(f"score: {score}, boxed: {boxed}, total: {score/boxed}")
            
    return score_stat


def vis_err_wait(score_stat):
    # [(wait_number, ..., score), ...]
    wait_numbers = [item[0] for item in score_stat]
    scores = [item[2] for item in score_stat]

    # 自定义分组边界和标签
    bin_labels = ["0", "1-2", "3-4", "5-6", "7-8", "9-10", "11-20", ">20"]

    def get_bin_label(w):
        if w == 0:
            return "0"
        elif 1 <= w <= 2:
            return "1-2"
        elif 3 <= w <= 4:
            return "3-4"
        elif 5 <= w <= 6:
            return "5-6"
        elif 7 <= w <= 8:
            return "7-8"
        elif 9 <= w <= 10:
            return "9-10"
        elif 11 <= w <= 20:
            return "11-20"
        else:
            return ">20"

    # 初始化分组统计
    bin_stats = defaultdict(lambda: [0, 0])  # [correct_count, incorrect_count]

    # 分组统计
    for w, s in zip(wait_numbers, scores):
        label = get_bin_label(w)
        if s == 1:
            bin_stats[label][0] += 1
        else:
            bin_stats[label][1] += 1

    # 保持顺序
    correct_counts = [bin_stats[label][0] for label in bin_labels]
    incorrect_counts = [bin_stats[label][1] for label in bin_labels]

    # 计算准确率
    accuracies = []
    for c, i in zip(correct_counts, incorrect_counts):
        total = c + i
        if total == 0:
            accuracies.append(None)
        else:
            accuracies.append(c / total)

    # 绘图
    x = np.arange(len(bin_labels))
    width = 0.6

    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(
        x, correct_counts, width, label="Correct (score=1)", color="skyblue"
    )
    bars2 = plt.bar(
        x,
        incorrect_counts,
        width,
        bottom=correct_counts,
        label="Incorrect (score=0)",
        color="salmon",
    )

    # 添加准确率标签
    for i, (x_pos, acc, c, i_count) in enumerate(
        zip(x, accuracies, correct_counts, incorrect_counts)
    ):
        total = c + i_count
        if acc is not None:
            y_pos = c + i_count + 1  # 稍微高于顶部
            plt.text(
                x_pos,
                y_pos,
                f"{acc:.1%}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    plt.xticks(x, bin_labels)
    plt.xlabel("Wait Number Range")
    plt.ylabel("Count")
    plt.title("Correct and Incorrect Counts per Wait Number Group (with Accuracy)")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


def vis_err_diff(score_stat):
    # [(wait_number, ..., score), ...]
    wait_numbers = [item[4] for item in score_stat]
    scores = [item[2] for item in score_stat]

    # 自定义分组边界和标签
    bin_labels = ["1", "2", "3", "4", "5"]

    def get_bin_label(w):
        if w == 1:
            return "1"
        elif w == 2:
            return "2"
        elif w == 3:
            return "3"
        elif w == 4:
            return "4"
        elif w == 5:
            return "5"

    # 初始化分组统计
    bin_stats = defaultdict(lambda: [0, 0])  # [correct_count, incorrect_count]

    # 分组统计
    for w, s in zip(wait_numbers, scores):
        label = get_bin_label(w)
        if s == 1:
            bin_stats[label][0] += 1
        else:
            bin_stats[label][1] += 1

    # 保持顺序
    correct_counts = [bin_stats[label][0] for label in bin_labels]
    incorrect_counts = [bin_stats[label][1] for label in bin_labels]

    # 计算准确率
    accuracies = []
    for c, i in zip(correct_counts, incorrect_counts):
        total = c + i
        if total == 0:
            accuracies.append(None)
        else:
            accuracies.append(c / total)

    # 绘图
    x = np.arange(len(bin_labels))
    width = 0.6

    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(
        x, correct_counts, width, label="Correct (score=1)", color="skyblue"
    )
    bars2 = plt.bar(
        x,
        incorrect_counts,
        width,
        bottom=correct_counts,
        label="Incorrect (score=0)",
        color="salmon",
    )

    # 添加准确率标签
    for i, (x_pos, acc, c, i_count) in enumerate(
        zip(x, accuracies, correct_counts, incorrect_counts)
    ):
        total = c + i_count
        if acc is not None:
            y_pos = c + i_count + 1  # 稍微高于顶部
            plt.text(
                x_pos,
                y_pos,
                f"{acc:.1%}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    plt.xticks(x, bin_labels)
    plt.xlabel("Wait Number Range")
    plt.ylabel("Count")
    plt.title("Correct and Incorrect Counts per Wait Number Group (with Accuracy)")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # run_extract_keywords()
    # run_test_prompt_boxed()
    # run_average_response_length()
    # run_wait_number()
    # run_grader_w_keywords()
    for model_name in [
        # "Qwen/Qwen2.5-1.5B",
        # "Qwen/Qwen2.5-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        # "meta-llama/Llama-3.1-8B",
    ]:
        for layer in tqdm(range(29)):
            extract_key_token(
                response_dir="./asset/response/DeepSeek-R1-Distill-Qwen-1.5B_MATH-500_140",
                model_name=model_name,
                token_position=-1,
                layer=layer,
            )

    # layers = 29
    # for model_name in [
    #     # "Qwen/Qwen2.5-1.5B",
    #     # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    #     "meta-llama/Llama-3.1-8B",
    # ]:
    #     if model_name == "meta-llama/Llama-3.1-8B":
    #         layers = 33
    #     for layer in range(layers):
    #         for position in [-2, -1]:
    #             extract_key_token(
    #                 response_dir="./asset/response/DeepSeek-R1-Distill-Qwen-1.5B_MATH-500_140",
    #                 model_name=model_name,
    #                 token_position=position,
    #                 layer=layer,
    #             )
