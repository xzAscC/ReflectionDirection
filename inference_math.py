import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, load_dataset

# Step 1: Load the tokenizer and model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.bfloat16
)

# Step 2: Load the dataset
dataset_name = "HuggingFaceH4/MATH-500"
dataset = load_dataset(dataset_name)


# Step 3: Define a prompt function
# TODO: Create a function to generate prompts for the model
def create_prompt(problem):
    return f"Solve the following math problem:\n{problem}\n\nShow your thinking process and provide the final answer in \\boxed{{}} format."


# Step 3: Tokenize the dataset
def tokenize_function(examples):
    prompts = [create_prompt(problem) for problem in examples["problem"]]
    return tokenizer(prompts, padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)


# Step 4: Perform inference
def generate_response(input_ids):
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids.to(model.device))
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


# Example of generating responses for the test set
test_problems = tokenized_datasets["test"]["input_ids"]
responses = [generate_response(torch.tensor([problem]))[0] for problem in test_problems]

# Print some example responses
for i, response in enumerate(responses[:5]):
    print(f"Problem {i+1}: {dataset['test']['problem'][i]}")
    print(f"Response: {response}\n")
