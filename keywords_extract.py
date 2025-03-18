import os
import re
import json
import torch
import seaborn as sns
import matplotlib.pyplot as plt
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
    # most keywords only appear in responses containing the word "wait"
    # Moreover, we observe that the majority of these instances involve the word "wait" preceding other keywords.
    # Furthermore, nearly all identified keywords co-occur with the word "wait" within the same sentence.
    keywords = []
    first_occurrence = []
    first_occurrence_this_sentence = []
    first_occurrence_wo_wait = []
    last_sentence_flag = False
    for idx, response_file in enumerate(tqdm(os.listdir(response_dir))):
        with open(os.path.join(response_dir, response_file), "r") as f:
            response = json.load(f)["response"]
            sentences = re.split(r"(?<=[.!?:])\s+", response)
            for idy, sentence in enumerate(sentences):
                this_sentence_flag = False
                for keyword in keywords_list:
                    if keyword in sentence.lower():
                        keywords.append(keyword)
                        if not this_sentence_flag:
                            this_sentence_flag = True
                            first_occurrence_this_sentence.append(keyword)
                            if not last_sentence_flag:
                                first_occurrence.append(keyword)
                            if idy > 0 and "wait" not in sentences[idy - 1].lower():
                                first_occurrence_wo_wait.append(keyword)
                last_sentence_flag = this_sentence_flag

    return (
        dict(Counter(keywords)),
        dict(Counter(first_occurrence)),
        dict(Counter(first_occurrence_wo_wait)),
        dict(Counter(first_occurrence_this_sentence)),
    )


def extract_token_before_wait(
    response_dir: str = "./reflect_responses",
    hidden_state_dir: str = "./hidden_state",
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
):
    # TODO: analyse the 7B response and extract the token position
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    last_token_before_wait = []
    last_token_before_wo_wait = []
    for idx, path in enumerate(tqdm(os.listdir(response_dir))):
        with open(os.path.join(response_dir, path), "r") as f:
            response = json.load(f)
        # hidden_states = torch.load(
        #     os.path.join(hidden_state_dir, path.split(".")[0] + ".pt")
        # )
        input_ids = tokenizer(response["response"], return_tensors="pt")[
            "input_ids"
        ].to("cuda:0")
        # problem_length = tokenizer(response["problem"], return_tensors="pt")[
        #     "input_ids"
        # ].shape[1]
        # input_length = input_ids.shape[1]
        wait_word = ["wait", "Wait", " wait", " Wait"]
        wait_list = []
        for word in wait_word:
            wait_list.append(
                tokenizer(word, return_tensors="pt")["input_ids"][0][1].item()
            )
        indices = []
        for word in wait_list:
            index = (input_ids[0] == word).nonzero().squeeze()
            if index.dim() == 0:  # if it's a scalar, add a dimension
                index = index.unsqueeze(0)
            indices.append(index)
        res = torch.cat(indices)
        last_token_before_wait.extend(input_ids[0][res - 1].tolist())
    last_token_before_wait_length = len(last_token_before_wait)
    last_token_before_wait_dict = dict(Counter(last_token_before_wait))
    last_token_before_wait_dict = dict(
        sorted(
            last_token_before_wait_dict.items(), key=lambda item: item[1], reverse=True
        )
    )
    # Tokenize the keys in last_token_before_wait_dict
    last_token_before_wait_dict_tokenized = {
        tokenizer.decode([key]): value
        for key, value in last_token_before_wait_dict.items()
    }
    return last_token_before_wait_dict_tokenized, last_token_before_wait_length


def token_rank_before_wait(
    response_dir: str = "./reflect_responses",
    hidden_state_dir: str = "./hidden_state",
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    token_list: List[int] = [382, 3983, 13],
):
    # TODO: extract the token hidden state and compute the rank
        # TODO: analyse the 7B response and extract the token position
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    last_token_before_wait = []
    layers = 0
    last_token_before_wo_wait = []
    for idx, path in enumerate(tqdm(os.listdir(response_dir))):
        if idx > 5:
            break
        with open(os.path.join(response_dir, path), "r") as f:
            response = json.load(f)
        hidden_states = torch.load(
            os.path.join(hidden_state_dir, path.split(".")[0] + ".pt")
        )
        layers = len(hidden_states[-1])
        input_ids = tokenizer(response["response"], return_tensors="pt")[
            "input_ids"
        ].to("cuda:0")
        problem_length = tokenizer(response["problem"], return_tensors="pt")[
            "input_ids"
        ].shape[1]
        input_length = input_ids.shape[1]
        wait_word = ["wait", "Wait", " wait", " Wait"]
        wait_list = []
        for word in wait_word:
            wait_list.append(
                tokenizer(word, return_tensors="pt")["input_ids"][0][1].item()
            )
        indices = []
        for word in wait_list:
            index = (input_ids[0] == word).nonzero().squeeze()
            if index.dim() == 0:  # if it's a scalar, add a dimension
                index = index.unsqueeze(0)
            indices.append(index)
        res = torch.cat(indices)
        for idy in res:
            token_before_wait = input_ids[0][idy-1].item()
            if token_before_wait == token_list[0]:
                last_token_before_wait.append(hidden_states[idy-1-problem_length])
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
                    if word in input_ids[0][index[i]:search_end_index]:
                        flag = True
                        break
                if not flag:
                    last_token_before_wo_wait.append(hidden_states[index[i]-problem_length])
    # rank_list = []
    # for layer in range(layers):
    #     token_rep_one_layer = torch.cat([
    #         x[layer].mean(dim=1, keepdim=True) if x[layer].shape[1] > 1 else x[layer]
    #         for x in last_token_before_wait
    #     ]).squeeze()
    #     token_rep_one_layer = token_rep_one_layer.to(torch.float32)
    #     rank = compute_rank(token_rep_one_layer)
    #     rank_list.append(rank)

    # # Visualize the rank list
    # # TODO: use wait have 4 forms
    # # TODO: why some token has 1, 47, 3584 dim
    # plt.figure(figsize=(10, 6))
    # sns.lineplot(x=range(len(rank_list)), y=rank_list, marker="o")
    # plt.title("Rank of Token Representations Across Layers")
    # plt.xlabel("Layer")
    # plt.ylabel("Rank")
    # plt.grid(True)
    # plt.show()
    
    return last_token_before_wait, last_token_before_wo_wait
    
        
def compute_rank(hidden_states, threshold=1e-5):
    """
    Computes the numerical rank of the hidden states matrix using SVD.

    Parameters:
        hidden_states (torch.Tensor): The hidden state representations of shape (batch_size, seq_len, hidden_dim)
        threshold (float): Singular values below this threshold are considered zero.

    Returns:
        rank (int): Estimated rank of the hidden states
    """
    if hidden_states.dim() == 3:
        # Reshape to 2D: (batch_size * seq_len, hidden_dim)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    # Compute Singular Value Decomposition (SVD)
    U, S, V = torch.svd(hidden_states)

    # Compute rank based on threshold
    rank = torch.sum(S > threshold).item()

    return rank


if __name__ == "__main__":
    # we observe that the majority of these instances involve the word "wait" preceding other keywords.
    # most keywords only appear in responses containing the word "wait"
    # nearly all identified keywords co-occur with the word "wait" within the same sentence.
    # print(extract_keywords())

    # keywords in no wait responses
    # print(extract_keywords(response_dir="./responses"))

    # the token before wait
    print(extract_token_before_wait())

    # analyse the rank of the token before wait
    # token_rank_before_wait()
