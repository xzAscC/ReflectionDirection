import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, load_dataset
import logging

# Step 1: Set up logging
def setup_logger():
    logger = logging.getLogger('inference_logger')
    logger.setLevel(logging.INFO)

    # Create file handler which logs even debug messages
    fh = logging.FileHandler('inference_results.log')
    fh.setLevel(logging.INFO)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

# Get the logger instance
logger = setup_logger()

# Step 2: Load the tokenizer and model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.bfloat16
)

# Step 3: Load the dataset
dataset_name = "HuggingFaceH4/MATH-500"
dataset = load_dataset(dataset_name)


# Step 4: Define a prompt function
def create_prompt(problem):
    return f"""
        Solve the following math problem:\n{problem}\n\n
        Show your thinking process and provide the final answer in \\boxed{{}} format.
    """


# Step 5: Tokenize the dataset
def tokenize_function(examples):
    prompts = [create_prompt(problem) for problem in examples["problem"]]
    return tokenizer(prompts, padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)


# Step 6: Perform inference
def generate_response(input_ids):
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids.to(model.device))
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


# Example of generating responses for the test set
test_problems = tokenized_datasets["test"]["input_ids"]
responses = [generate_response(torch.tensor([problem]))[0] for problem in test_problems]

# Save the results to a logger file and stdout
for i, response in enumerate(responses):
    logger.info(f"Problem {i+1}: {dataset['test']['problem'][i]}")
    logger.info(f"Response: {response}\n")

logger.info("Results have been saved to inference_results.log")
