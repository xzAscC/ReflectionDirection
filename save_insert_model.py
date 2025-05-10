import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
with torch.no_grad():
    layer = 20
    alpha = -0.1

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # Load the tokenizer        

    model1 = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # w_wait = (
    #     torch.load(
    #         f"./asset/response/DeepSeek-R1-Distill-Qwen-1.5B_MATH-500_140/DeepSeek-R1-Distill-Qwen-1.5B_hs/before_wait_DeepSeek-R1-Distill-Qwen-1.5B_{layer}_-1.pt"
    #     )
    #     .to("cuda")
    #     .to(torch.float32)
    # )
    # w_wo_wait = (
    #     torch.load(
    #         f"./asset/response/DeepSeek-R1-Distill-Qwen-1.5B_MATH-500_140/DeepSeek-R1-Distill-Qwen-1.5B_hs/before_wo_wait_DeepSeek-R1-Distill-Qwen-1.5B_{layer}_-1.pt"
    #     )
    #     .to("cuda")
    #     .to(torch.float32)
    # )
    # insert_vector = w_wait.mean(dim=0) - w_wo_wait.mean(dim=0)
    # insert_vector = insert_vector.to("cuda").to(torch.bfloat16)
    # tmp = alpha * torch.matmul(
    #                 insert_vector.unsqueeze(-1), insert_vector.unsqueeze(-2)
    #             ).matmul(model.model.layers[layer - 1].mlp.down_proj.weight)
    # model.model.layers[layer - 1].mlp.down_proj.weight += tmp
    model_name = "Qwen/Qwen2.5-1.5B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    for layer in range(10, 21):
        model.model.layers[layer - 1].mlp.down_proj.weight -= 0.1*(model1.model.layers[layer - 1].mlp.down_proj.weight - model.model.layers[layer - 1].mlp.down_proj.weight)
    model.save_pretrained(f"./models/insert_model_{layer}_{alpha}")
    tokenizer.save_pretrained(f"./models/insert_model_{layer}_{alpha}")
