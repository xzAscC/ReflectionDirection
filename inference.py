from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import json
import os
from tqdm import tqdm
from functools import lru_cache

# TODO: wait number and the model size
# TODO: wait helps or deteriorates the performance
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
dataset_name = "HuggingFaceH4/MATH-500"
# TODO: why wait and space wait is different, why .\n\n is a token instead of .
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.bfloat16
)

########################################
# last_token_before_wait = torch.load("last_token_before_wait.pt")
# last_token_before_wo_wait = torch.load("last_token_before_wo_wait.pt")
# layers = 29
# icv = torch.zeros(layers, last_token_before_wo_wait[0][0].shape[-1])
# for layer in tqdm(range(layers)):
#     token_rep_one_layer = torch.cat([
#         x[layer].mean(dim=1, keepdim=True) if x[layer].shape[1] > 1 else x[layer]
#         for x in last_token_before_wo_wait
#     ]).to("cuda:0").squeeze()
#     token_rep_one_layer = token_rep_one_layer.to(torch.float32)
#     token_before_wait = torch.cat([
#         x[layer].mean(dim=1, keepdim=True) if x[layer].shape[1] > 1 else x[layer]
#         for x in last_token_before_wait
#     ]).to("cuda:0").squeeze()
#     mean_token_rep = token_rep_one_layer.mean(dim=0)
#     mean_token_before = token_before_wait.mean(dim=0)
#     diff_means = mean_token_before - mean_token_rep
#     icv[layer] = diff_means

# lam = 0.12

# sim_list = []
# layers = 29
# icv = torch.zeros(layers, last_token_before_wo_wait[0][0].shape[-1])
# import random

# selected_examples = random.sample(
#     last_token_before_wo_wait, len(last_token_before_wait)
# )
# print(f"Number of selected examples: {len(selected_examples)}")
# for layer in tqdm(range(layers)):
#     token_rep_one_layer = torch.cat([
#         x[layer].mean(dim=1, keepdim=True) if x[layer].shape[1] > 1 else x[layer]
#         for x in selected_examples
#     ]).to("cuda:0").squeeze()
#     token_rep_one_layer = token_rep_one_layer.to(torch.float32)
#     token_before_wait = torch.cat([
#         x[layer].mean(dim=1, keepdim=True) if x[layer].shape[1] > 1 else x[layer]
#         for x in last_token_before_wait
#     ]).to("cuda:0").squeeze()
#     diff_means = token_before_wait - token_rep_one_layer
#     res_mat = torch.matmul(diff_means.T, diff_means)
#     # Compute eigen decomposition (PCA) on res_mat and select the first eigenvector
#     eigenvalues, eigenvectors = torch.linalg.eigh(res_mat)
#     first_eigenvector = eigenvectors[:, -1]
#     icv[layer] = first_eigenvector
# lam = 0.12
# add_icv_layers(model, icv[1:].cuda(), [lam])
########################################
# Load the MATH-500 dataset
dataset = load_dataset("HuggingFaceH4/MATH-500")

# Create output directory if it doesn't exist
os.makedirs("responses", exist_ok=True)
os.makedirs("long_responses2", exist_ok=True)
os.makedirs("long_hidden_state2", exist_ok=True)
problems = dataset["test"]["problem"]
os.makedirs("icv_pca", exist_ok=True)
torch.set_grad_enabled(False)

# Process each problem and get response
for idx, problem in enumerate(tqdm(problems)):
    # Prepare prompt for inference
    prompt = f"""<|im_start|>system
    Please reason step by step, and put your final answer within \\boxed{{}}.
    <|im_end|>
    <|im_start|>user
    {problem}
    <|im_end|>
    <|im_start|>assistant
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    
    # # Generate output text with output_hidden_states=True to get hidden states
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=8196,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

    # # Get the generated tokens and hidden states
    generated_tokens = outputs.sequences[0]
    hidden_states = outputs.hidden_states

    # Print the structure of hidden states
    # print("\n###### Hidden States Structure ########")
    # print(f"Number of generation steps: {len(hidden_states)}")
    # for i, hidden_state in enumerate(hidden_states):
    #     print(f"Length of hidden states at step {i}: {len(hidden_state)}")
    #     print(f"One layer shape ", hidden_state[0].shape)

    generated_tokens, hidden_states = generate_response(problem)

    # Decode the response and save results
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Save hidden states (only keep necessary layers)
    hidden_file = f"long_hidden_state2/problem_{idx:04d}.pt"
    torch.save([h[0] for h in hidden_states], hidden_file)
    
    # Create and save response object
    output_file = f"long_responses2/problem_{idx:04d}.json"
    with open(output_file, "w") as f:
        json.dump({
            "problem": problem,
            "response": response
        }, f)
