"""
ACDC for Gender Pronoun (GP) with evaluation metrics.
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
    parser.add_argument("--dataset-path", "-d", default="/scratch/network/mc3803/Edge-Pruning/data/datasets/gp/")
    parser.add_argument("--max-train-examples", "-n", default=150, type=int)
    parser.add_argument("--batch-size", "-b", default=32, type=int)
    parser.add_argument("--max-num-epochs", "-e", default=100000, type=int)
    parser.add_argument("--device", "-D", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--out-json-path", "-j", default=None)
    parser.add_argument("--out-pickle-path-final", "-f", default=None)
    args = parser.parse_args()
    if args.out_json_path is None:
        args.out_json_path = f"results/gp-sweep/gp-t{args.threshold}-graph.json"
    if args.out_pickle_path_final is None:
        args.out_pickle_path_final = f"results/gp-sweep/gp-t{args.threshold}-graph.pkl"
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
os.makedirs("results/gp-sweep", exist_ok=True)

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
test_sentences = [test_data[i]['prefix'] for i in range(len(test_data))]
test_toks = tokenizer(test_sentences, return_tensors="pt", padding=True).input_ids

# Get correct/distractor tokens (pronouns)
targets = []
distractors = []
prefix_lengths = []
for i in range(len(test_data)):
    pronoun = test_data[i]['pronoun']
    corr_pronoun = test_data[i]['corr_pronoun']
    targets.append(tokenizer.encode(" " + pronoun)[0])
    distractors.append(tokenizer.encode(" " + corr_pronoun)[0])
    sentence = test_data[i]['prefix']
    prefix = sentence[:sentence.rfind(" ")]
    prefix_lengths.append(len(tokenizer.tokenize(prefix)) - 1)

targets = torch.LongTensor(targets)
distractors = torch.LongTensor(distractors)
prefix_lengths = torch.LongTensor(prefix_lengths)

circuit_pred = pred(model, test_toks, args.device)  # hooks still active

model_copy = HookedTransformer.from_pretrained('gpt2', center_writing_weights=False, center_unembed=False, fold_ln=False, device=args.device)
model_copy.set_use_hook_mlp_in(True)
model_copy.set_use_split_qkv_input(True)
model_copy.set_use_attn_result(True)
ref_pred_logits = pred(model_copy, test_toks, args.device)

kl_total = 0
ld_total = 0
acc_total = 0
n = len(test_data)

for i in range(n):
    pl = prefix_lengths[i].item()
    circuit_logits = circuit_pred[i, pl, :]
    ref_logits = ref_pred_logits[i, pl, :]
    
    ld_total += (circuit_logits[targets[i]] - circuit_logits[distractors[i]]).item()
    acc_total += (circuit_logits.argmax() == targets[i]).int().item()
    
    log_p = torch.log_softmax(circuit_logits, dim=-1)
    ref_log_p = torch.log_softmax(ref_logits, dim=-1)
    kl_total += torch.nn.functional.kl_div(log_p, ref_log_p, log_target=True, reduction="sum").item()

print(f"\n[i] Overall Edge Count: {n_edges}")
print(f"[i]     KL Divergence: {kl_total/n}")
print(f"[i]     Logit difference: {ld_total/n}")
print(f"[i]     Accuracy: {acc_total/n}")
