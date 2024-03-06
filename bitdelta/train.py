import os
import torch
from torch import nn
import gc
import torch.nn.functional as F
from bitdelta.diff import  save_diff, save_full_model
from bitdelta.misc import find_corr_stddev
from bitdelta.binary_gemm_kernel import pack, unpack, binary_bmm
from bitdelta.utils import get_model, parse_args, get_tokenizer
from tqdm import tqdm
from bitdelta.data import get_dataset, get_dataloader

import json
import transformers

args = parse_args()

# import pdb ; pdb.set_trace()

# create save_dir if it doesn't exist
os.makedirs(args.save_dir, exist_ok=True)

tokenizer = get_tokenizer(args.base_model)

with torch.no_grad():
     base_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            # device_map="auto"   
        ).to("cpu")
    
     finetuned_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.finetuned_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            # device_map="auto"   
        ).to("cpu")

finetuned_compressed_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.finetuned_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            # device_map="auto"   
        ).to("cpu")

print(f"compressing diff...")

class BinaryDiff(nn.Module):
    def __init__(self, base, finetune):
        super().__init__()
        diff = finetune - base
        # diff = decomposition(diff, 2048)
        quantile = diff.float().abs().mean()

        mask = torch.ones_like(diff)
        mask[diff < 0] = 0
        mask = pack(mask.bool().T)
     
        self.register_buffer("mask", mask)
        self.register_buffer("base", base.T)
        self.register_parameter(
            "coeff",
            nn.Parameter(
                torch.tensor(
                    quantile,
                    dtype=torch.float32,
                    requires_grad=True,
                    device=base.device,
                )
            ),
        )
        del base, finetune, diff

    def forward(self, x):
        # print(x.shape, self.base.shape, self.coeff.shape, self.mask.shape)
        # [B, seq, in] @ [in, out] + [B, seq, in] @ [B, in/32, out]

        # TODO: This can be faster
        repeated_mask = self.mask.unsqueeze(0).repeat(x.size(0), 1, 1)
        return x @ self.base + self.coeff * binary_bmm(x, repeated_mask)



def compress_diff(base_model, finetuned_model, finetuned_compressed_model,layers=None):
    def compress_submodule(name, subname, subsubname, module, submodule):
        target_device = submodule.weight.device
                    
        base_weight = base_model.get_submodule(f"{name}.0.{subsubname}").weight.detach().to(target_device)
        finetuned_weight = finetuned_model.get_submodule(f"{name}.{subname}.{subsubname}").weight.detach().to(target_device)
        
        compressed = BinaryDiff(
            base=base_weight,
            finetune=finetuned_weight,
        ).to(target_device)

        del submodule, base_weight
        setattr(module, subname, None)
        gc.collect()
        torch.cuda.empty_cache()
        setattr(module, subname, compressed)

    # TODO: this can be parallelized
    for name, module in finetuned_compressed_model.named_modules():
        if "experts" in name :
            for subname, submodule in module.named_children():
                if "0" not in subname:
                    for subsubname, subsubmodule in submodule.named_children():
                        if "w" in subsubname:
                            # import pdb ; pdb.set_trace()
                            compress_submodule(name, subname, subsubname,module, subsubmodule)

   
compress_diff(base_model, finetuned_model, finetuned_compressed_model,layers=args.layers)

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

# save untrained delta
save_diff(finetuned_compressed_model, os.path.join(args.save_dir, "diff_untrained.pt"),layers=args.layers)

if args.train:
    optimizer = torch.optim.AdamW(finetuned_compressed_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_steps)

    bar = tqdm(train_dataloader)

    train_loss_list = []

    # Train loop
    for step, batch in enumerate(bar):
        batch1 = {k: v.to(finetuned_model.device) for k, v in batch.items()}
        with torch.inference_mode():
            finetuned_outputs = finetuned_model(**batch1)

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


# # save loss list
# if args.debug:
#     with open(os.path.join(args.save_dir, f"train_loss_{args.num_groups}.json"), "w") as f:
#         json.dump(train_loss_list, f)

# # save trained delta
save_diff(finetuned_compressed_model, os.path.join(args.save_dir, "diff.pt"),layers=args.layers)

del base_model, finetuned_model, finetuned_compressed_model
torch.cuda.empty_cache()

if args.save_full_model:
    # print("saving uncalibrated model")
    # save_full_model(args.base_model, args.finetuned_model, os.path.join(args.save_dir, "diff_untrained.pt"), os.path.join(args.save_dir, f"uncalibrated_test"), device="cpu",layers=args.layers)
    # print("saving calibrated model")
    save_full_model(args.base_model, args.finetuned_model, os.path.join(args.save_dir, "diff.pt"), os.path.join(args.save_dir, "calibrated_mixtral"), device="cpu")
