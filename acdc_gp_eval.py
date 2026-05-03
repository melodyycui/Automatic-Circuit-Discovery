"""
ACDC for GT - finds circuit and saves to JSON.
"""
import torch
from tqdm import tqdm
from datasets import load_from_disk
import json
import random
import argparse
import pickle
from transformer_lens.HookedTransformer import HookedTransformer
from acdc.TLACDCExperiment import TLACDCExperiment

TOTAL_EDGES = 32923

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", "-t", default=0.001, type=float)
    parser.add_argument("--dataset-path", "-d", default="/content/drive/MyDrive/acdc_datasets/gp")
    parser.add_argument("--max-train-examples", "-n", default=150, type=int)
    parser.add_argument("--max-num-epochs", "-e", default=100000, type=int)
    parser.add_argument("--device", "-D", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--out-json-path", "-j", default=None)
    parser.add_argument("--out-pickle-path-final", "-f", default=None)
    args = parser.parse_args()
    if args.out_json_path is None:
        args.out_json_path = f"/content/drive/MyDrive/acdc_results/gp-t{args.threshold}-graph.json"
    if args.out_pickle_path_final is None:
        args.out_pickle_path_final = f"/content/drive/MyDrive/acdc_results/gp-t{args.threshold}-graph.pkl"

    return args

args = parse_args()

random.seed(42)
torch.random.manual_seed(42)
torch.autograd.set_grad_enabled(False)

model = HookedTransformer.from_pretrained(
    'gpt2',
    center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device=args.device,
)
model.set_use_hook_mlp_in(True)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)
tokenizer = model.tokenizer
tokenizer.pad_token = tokenizer.eos_token

dataset = load_from_disk(args.dataset_path)
train_data = dataset["train"]
if args.max_train_examples < len(train_data):
    train_data = train_data.select(range(args.max_train_examples))

train_sentences = [train_data[i]['prefix'] for i in range(len(train_data))]
train_corr_sentences = [train_data[i]['corr_prefix'] for i in range(len(train_data))]
train_toks = tokenizer(train_sentences, return_tensors="pt", padding=True).input_ids
train_corr_toks = tokenizer(train_corr_sentences, return_tensors="pt", padding=True).input_ids

@torch.no_grad()
def pred(model, tokens, device, batch_size=32):
    logits = []
    for i in range(0, tokens.shape[0], batch_size):
        logits.append(model(tokens[i:i+batch_size].to(device)).cpu())
    return torch.cat(logits, dim=0)

train_pred = pred(model, train_toks, args.device)

def kl_metric(logits, full_model_logits):
    logits = logits[:, -1, :]
    full_model_logits = full_model_logits[:, -1, :].to(logits.device)
    return torch.nn.functional.kl_div(
        torch.log_softmax(logits, dim=-1),
        torch.log_softmax(full_model_logits, dim=-1),
        log_target=True, reduction="batchmean"
    )

metric = lambda logits: kl_metric(logits, train_pred)

import os
os.makedirs("/content/drive/MyDrive/acdc_results", exist_ok=True)

model.reset_hooks()
experiment = TLACDCExperiment(
    model=model,
    ds=train_toks,
    ref_ds=train_corr_toks,
    threshold=args.threshold,
    metric=metric,
    online_cache_cpu=False,
    corrupted_cache_cpu=False,
    verbose=True,
)

bar = tqdm(range(args.max_num_epochs))
for i in bar:
    experiment.step()
    edge_count = experiment.count_no_edges()
    bar.set_description(f"Epoch {i+1}: {edge_count} edges")
    if experiment.current_node is None:
        break

experiment.save_edges(args.out_pickle_path_final)
n_edges = experiment.count_no_edges()
edge_sparsity = 1 - (n_edges / TOTAL_EDGES)

print(f"[i] Overall Edge Count: {n_edges}")
print(f"[i] Edge Sparsity: {edge_sparsity}")

graph = experiment.save_subgraph(return_it=True)

good_graph = []
good_graph_extra = []
for to_name, to_idx, from_name, from_idx in graph:
    to_parts = to_name.split(".")
    from_parts = [from_name] if "." not in from_name else from_name.split(".")
    if from_parts[0] in ["hook_embed", "hook_pos_embed"]:
        continue
    to_layer_num = int(to_parts[1])
    from_layer_num = int(from_parts[1])
    if to_parts[2] == "attn":
        good_graph_extra.append({"from": from_name, "to": to_name})
        continue
    elif to_parts[2] == "hook_mlp_out":
        good_graph_extra.append({"from": from_name, "to": to_name})
        continue
    elif to_parts[2] == "hook_resid_post":
        to_name = "resid_post"
    elif to_parts[2] == "hook_mlp_in":
        to_name = f"mlp.{to_layer_num}"
    elif to_parts[2] == "hook_q_input":
        to_name = f"head.{to_layer_num}.{to_idx[2]}.q"
    elif to_parts[2] == "hook_k_input":
        to_name = f"head.{to_layer_num}.{to_idx[2]}.k"
    elif to_parts[2] == "hook_v_input":
        to_name = f"head.{to_layer_num}.{to_idx[2]}.v"
    else:
        continue
    if from_parts[2] == "attn":
        from_name = f"head.{from_layer_num}.{from_idx[2]}"
    elif from_parts[2] == "hook_mlp_out":
        from_name = f"mlp.{from_layer_num}"
    elif from_parts[2] == "hook_resid_pre":
        good_graph_extra.append({"from": from_name, "to": to_name})
        continue
    else:
        continue
    good_graph.append({"from": from_name, "to": to_name})

json.dump({"original": good_graph, "extra": good_graph_extra}, open(args.out_json_path, "w+"), indent=4)
print(f"Saved circuit to {args.out_json_path}")
print(f"No. edges: {len(good_graph)}")