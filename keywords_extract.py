import os
import re
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
from collections import Counter


# extract the keywords list from the response
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
    # TODO: get response from the 7B
    # TODO: most keywords only appear in responses containing the word "wait"
    # Moreover, we observe that the majority of these instances involve the word "wait" preceding other keywords.
    # TODO: Furthermore, nearly all identified keywords co-occur with the word "wait" within the same sentence.
    keywords = []
    for response_file in tqdm(os.listdir(response_dir)):
        with open(os.path.join(response_dir, response_file), "r") as f:
            response = json.load(f)["response"]
            sentences = re.split(r"(?<=[.!?:])\s+", response)
            for sentence in sentences:
                for keyword in keywords_list:
                    if keyword in sentence:
                        keywords.append(keyword)
                        break
    return dict(Counter(keywords))


def extract_token_before_wait(
    response_dir: str = "./reflect_responses",
    hidden_state_dir: str = "./hidden_state",
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
):
    # TODO: analyse the 7B response and extract the token position
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    last_token_before_wait = []
    last_token_before_wo_wait = []
    for idx, path in enumerate(tqdm(os.listdir(response_dir))):
        with open(os.path.join(response_dir, path), "r") as f:
            response = json.load(f)["response"]
        hidden_states = torch.load(os.path.join(hidden_state_dir, path.split('.')[0] + '.pt'))
        input_ids = tokenizer(response['response'], return_tensors="pt")['input_ids'].to("cuda:0")
        problem_length = tokenizer(response['problem'], return_tensors="pt")['input_ids'].shape[1]
        input_length = input_ids.shape[1]
        wait_word = ['wait', 'Wait', ' wait', ' Wait']
        wait_list = []
        for word in wait_word:
            wait_list.append(tokenizer(word, return_tensors="pt")['input_ids'][0][1].item())
        indices = []
        for word in wait_list:
            index = (input_ids[0] == word).nonzero().squeeze()
            if index.dim() == 0:  # if it's a scalar, add a dimension
                index = index.unsqueeze(0)
            indices.append(index)
        res = torch.cat(indices)
        last_token_before_wait.extend(input_ids[0][res-1].tolist())
    # ## find other position for last token
    # index = (input_ids[0] == 193).nonzero().squeeze()
    # if index.dim() == 0:  # if it's a scalar, add a dimension
    #     index = index.unsqueeze(0)
    # token_wo_wait = []
    # for i in range(index.shape[0]):
    #     input_length = input_ids[0].shape[0]
    #     if index[i] + 100 > input_length:
    #         search_end_index = input_length
    #     else:
    #         search_end_index = index[i] + 100
    #     flag = False
    #     for word in wait_list:
    #         if word in input_ids[0][index[i]:search_end_index]:
    #             flag = True
    #             print(tokenizer.decode(input_ids[0][index[i]:search_end_index]))
    #             break
    #     if not flag:
    #         token_wo_wait.append(index[i].item())
    return last_token_before_wait