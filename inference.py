from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import json
import tqdm
import os

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model in model-parallel mode:
# "device_map='auto'" automatically distributes the model across available GPUs.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"  # Automatically assigns model parts to GPUs
)

# Load the MATH-500 dataset
dataset = load_dataset("HuggingFaceH4/MATH-500")


# Create output directory if it doesn't exist
os.makedirs("responses", exist_ok=True)

problems = dataset["test"]["problem"]

# Process each problem and get response
for idx, problem in enumerate(tqdm.tqdm(problems)):
    # Prepare prompt for inference
    inputs = tokenizer(problem, return_tensors="pt").to("cuda:0")
    
    # Generate output text with output_hidden_states=True to get hidden states
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=1000,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        output_hidden_states=True,
        return_dict_in_generate=True
    )
    
    # Get the generated tokens and hidden states
    generated_tokens = outputs.sequences[0]
    hidden_states = outputs.hidden_states

    # Print the structure of hidden states
    # print("\n###### Hidden States Structure ########")
    # print(f"Number of generation steps: {len(hidden_states)}")
    # for i, hidden_state in enumerate(hidden_states):
    #     print(f"Length of hidden states at step {i}: {len(hidden_state)}")
    #     print(f"One layer shape ", hidden_state[0].shape)
    
    
    # Decode the response
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Create response object
    response_obj = {
        "problem": problem,
        "response": response
    }
    # print("###### Response ########")
    # print(response)

    # Check if response contains "Wait" or "wait"
    wait_count = response.count("Wait") + response.count("wait")
    print("Total number of 'wait' occurrences:", wait_count)


    # Check if response contains "Wait" or "wait"
    if "Wait" in response or "wait" in response:
        # Save hidden states and response to reflect_responses folder
        output_file = f"reflect_responses/problem_{idx:04d}.json"
        # Convert hidden states to list for JSON serialization
        # hidden_states_list = [[layer.detach().cpu().numpy().tolist() for layer in h] for h in hidden_states]
        # response_obj["hidden_states"] = hidden_states_list
    else:
        # Save to responses folder
        output_file = f"responses/problem_{idx:04d}.json"

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
    # Save response to JSON file
    with open(output_file, "w") as f:
        json.dump(response_obj, f, indent=2)
