import os

import torch

import torch.nn.functional as F
from bitdelta.diff2 import compress_diff, save_diff, save_full_model,Delta
from bitdelta.misc import find_corr_stddev

from bitdelta.utils import get_model, parse_args, get_tokenizer
from tqdm import tqdm
from bitdelta.data import get_dataset, get_dataloader
from transformers import AutoModelForCausalLM
import json

args = parse_args()

# create save_dir if it doesn't exist
os.makedirs(args.save_dir, exist_ok=True)

tokenizer = get_tokenizer(args.finetuned_model)

with torch.no_grad():
    base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to("cuda:0")
    
    finetuned_model = AutoModelForCausalLM.from_pretrained(
            args.finetuned_model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to("cuda:0")
    
    # base_model = get_model(args.base_model, args.base_model_device, args.base_model_memory_map).to(torch.float32)
    # finetuned_model = get_model(args.finetuned_model, args.finetuned_model_device, args.finetuned_model_memory_map).to(torch.float32)

# finetuned_compressed_model = get_model(args.finetuned_model, args.finetuned_compressed_model_device, args.finetuned_compressed_model_memory_map)

finetuned_compressed_model = AutoModelForCausalLM.from_pretrained(            
                                args.finetuned_model,
                                torch_dtype=torch.bfloat16,
                                low_cpu_mem_usage=True,
                                device_map="auto")

print(f"compressing diff...")
compress_diff(base_model, finetuned_model, finetuned_compressed_model,args.save_dir)

if args.train:
    
    train_num_samples = args.batch_size * args.num_steps
    train_dataset = get_dataset(
        args.dataset_name,
        args.subset,
        "train",
        size=train_num_samples,
    )
    train_dataloader = get_dataloader(
        train_dataset,
        tokenizer,
        args.batch_size,
        num_workers=4,
        max_length=args.max_length,
    )   
    
    optimizer = torch.optim.AdamW(finetuned_compressed_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_steps)

    bar = tqdm(train_dataloader)

    train_loss_list = []

    # Train loop
    for step, batch in enumerate(bar):
        batch1 = {k: v.to(finetuned_model.device) for k, v in batch.items()}
        with torch.inference_mode():
            finetuned_outputs = finetuned_model(**batch1)

        # import pdb; pdb.set_trace()
        
        batch2 = {k: v.to(finetuned_compressed_model.device) for k, v in batch.items()}
        finetuned_compressed_outputs = finetuned_compressed_model(**batch2)

        loss = F.mse_loss(
            finetuned_outputs.logits.clone().to(finetuned_compressed_outputs.logits.device),
            finetuned_compressed_outputs.logits,
        )

        train_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        scheduler.step()

        bar.set_description(f"train loss: {loss.item()}")

def merge(finetuned_compressed_model, finetuned_model,save_dir):
    diff_dict = {}

    for name, module in finetuned_compressed_model.named_modules():
        if isinstance(module, Delta):
            diff_dict[name + ".weight"] = module.base + module.U @ torch.diag(module.S) @ module.V.T  # self.U @ torch.diag(self.S) @ self.V.T
            # import pdb; pdb.set_trace()

    finetuned_model.load_state_dict(diff_dict, strict=False)
    # import pdb; pdb.set_trace()
    finetuned_model.to(torch.bfloat16)
    
    finetuned_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


merge(finetuned_compressed_model, finetuned_model,args.save_dir)

# 打开文件
# file_path = '/home/pingbowen/workspace/lora-fusion/UltraEval/datasets/gsm8k/data/gsm8k.jsonl'
# with open(file_path, 'r') as file:
#     # 遍历文件中的每一行
#     for line in file:
#         # 解析JSON数据
#         data = json.loads(line)
#         text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n ### Instruction:\n{data['question']}\n\n### Response: Let's think step by step."
#         input_ids = tokenizer(text, return_tensors='pt').to(args.finetuned_compressed_model_device)
#         outputs = finetuned_compressed_model.generate(**input_ids, max_new_tokens=512, do_sample=True, top_p=1, temperature=0.0001)
#         outputs = tokenizer.batch_decode(
#            outputs.to("cpu"), skip_special_tokens=True)
#         for i in range(len(outputs)):
#             if outputs[i].startswith(text):
#                 outputs[i] = outputs[i][len(text):]