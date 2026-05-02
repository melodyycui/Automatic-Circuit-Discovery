"""
ACDC for IOI with evaluation metrics (KL, logit difference, accuracy).
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", "-t", default=0.001, type=float)
    parser.add_argument("--dataset-path", "-d", default="/scratch/network/mc3803/Edge-Pruning/data/datasets/ioi/")
    parser.add_argument("--max-train-examples", "-n", default=200, type=int)
    parser.add_argument("--max-test-examples", default=10000, type=int)
    parser.add_argument("--batch-size", "-b", default=32, type=int)
    parser.add_argument("--max-num-epochs", "-e", default=100000, type=int)
    parser.add_argument("--device", "-D", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--out-json-path", "-j", default=None)
    parser.add_argument("--out-pickle-path-final", "-f", default=None)
    args = parser.parse_args()
    if args.out_json_path is None:
        args.out_json_path = f"results/ioi-sweep/ioi-t{args.threshold}-graph.json"
    if args.out_pickle_path_final is None:
        args.out_pickle_path_final = f"results/ioi-sweep/ioi-t{args.threshold}-graph.pkl"
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

# Load train data for ACDC
train_data = dataset["train"]
if args.max_train_examples < len(train_data):
    train_data = train_data.select(range(args.max_train_examples))

def get_ioi_data(data, tokenizer):
    tokens = tokenizer([d['ioi_sentences'] for d in data], return_tensors="pt", padding=True).input_ids
    corr_tokens = tokenizer([d['corr_ioi_sentences'] for d in data], return_tensors="pt", padding=True).input_ids
    cont_indices = []
    correct_tokens = []
    distractor_tokens = []
    for d in data:
        prefix = d['ioi_sentences'][:d['ioi_sentences'].rfind(" ")]
        idx = len(tokenizer.tokenize(prefix)) - 1
        cont_indices.append(idx)
        correct_tokens.append(tokenizer.encode(" " + d['a'])[0])
        distractor_tokens.append(tokenizer.encode(" " + d['b'])[0])
    return (tokens, corr_tokens,
            torch.LongTensor(cont_indices),
            torch.LongTensor(correct_tokens),
            torch.LongTensor(distractor_tokens))

train_toks, train_corr_toks, train_idx, train_correct, train_distractor = get_ioi_data(train_data, tokenizer)

@torch.no_grad()
def pred(model, tokens, device, batch_size=32):
    logits = []
    for i in range(0, tokens.shape[0], batch_size):
        logits.append(model(tokens[i:i+batch_size].to(device)).cpu())
    return torch.cat(logits, dim=0)

train_pred = pred(model, train_toks, args.device)

def kl_divergence_metric(logits, full_model_logits, indices):
    indices = indices.to(logits.device)
    logits = torch.gather(logits, 1, indices.reshape(-1,1,1).repeat(1,1,logits.shape[-1])).squeeze(1)
    full_model_logits = torch.gather(full_model_logits, 1, indices.reshape(-1,1,1).repeat(1,1,full_model_logits.shape[-1])).squeeze(1).to(logits.device)
    log_probs = torch.log_softmax(logits, dim=-1)
    full_log_probs = torch.log_softmax(full_model_logits, dim=-1)
    return torch.nn.functional.kl_div(log_probs, full_log_probs, log_target=True, reduction="batchmean")

metric = lambda logits: kl_divergence_metric(logits, train_pred, train_idx)

import os
os.makedirs("results/ioi-sweep", exist_ok=True)

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

# Evaluate on test set
test_data = dataset["test"]
if args.max_test_examples < len(test_data):
    test_data = test_data.select(range(args.max_test_examples))

test_toks, test_corr_toks, test_idx, test_correct, test_distractor = get_ioi_data(test_data, tokenizer)
full_model_pred = pred(model, test_toks, args.device)  # hooks still active = circuit

# Run full (unhooked) model for KL reference
model_copy = HookedTransformer.from_pretrained('gpt2', center_writing_weights=False, center_unembed=False, fold_ln=False, device=args.device)
model_copy.set_use_hook_mlp_in(True)
model_copy.set_use_split_qkv_input(True)
model_copy.set_use_attn_result(True)
ref_pred = pred(model_copy, test_toks, args.device)

# Compute metrics
kl_total = 0
ld_total = 0
acc_total = 0

for i in range(0, len(test_data), args.batch_size):
    idx = test_idx[i:i+args.batch_size]
    circuit_logits = torch.gather(full_model_pred[i:i+args.batch_size], 1, idx.reshape(-1,1,1).repeat(1,1,full_model_pred.shape[-1])).squeeze(1)
    ref_logits = torch.gather(ref_pred[i:i+args.batch_size], 1, idx.reshape(-1,1,1).repeat(1,1,ref_pred.shape[-1])).squeeze(1)
    
    log_probs = torch.log_softmax(circuit_logits, dim=-1)
    ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
    kl = torch.nn.functional.kl_div(log_probs, ref_log_probs, log_target=True, reduction="sum")
    kl_total += kl.item()
    
    correct = test_correct[i:i+args.batch_size]
    distractor = test_distractor[i:i+args.batch_size]
    ld = (circuit_logits.gather(1, correct.reshape(-1,1)) - circuit_logits.gather(1, distractor.reshape(-1,1))).sum()
    ld_total += ld.item()
    
    acc = (circuit_logits.argmax(dim=-1) == correct).sum()
    acc_total += acc.item()

n = len(test_data)
print(f"\n[i] Overall Edge Count: {n_edges}")
print(f"[i]     KL Divergence: {kl_total/n}")
print(f"[i]     Logit difference: {ld_total/n}")
print(f"[i]     Accuracy: {acc_total/n}")
