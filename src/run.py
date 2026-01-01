import os
from tqdm import tqdm
from process_data import create_dataset
import torch
from options import Options
import numpy as np
import torch
import zlib
import torch.nn.functional as F
from transformers import set_seed
import transformers
import torch 
import random
import openai 
from accelerate import Accelerator
import torch 
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import csv


# Initialize vulnerable tokens
VULNERABLE_TOKENS = None
VULNERABLE_TOKEN_IDS = None

def load_vulnerable_tokens(tokenizer, filepath):
    global VULNERABLE_TOKENS, VULNERABLE_TOKEN_IDS
    if VULNERABLE_TOKENS is None:
        vulnerable_list = set()
        with open(filepath, 'r', encoding='utf-8') as f:
            current_text = []
            for line in f:
                if line.strip() == '':
                    if current_text:
                        text = ' '.join(current_text)
                        tokens = tokenizer.tokenize(text)
                        vulnerable_list.update(tokens)
                        current_text = []
                else:
                    current_text.append(line.strip())
            if current_text:
                text = ' '.join(current_text)
                tokens = tokenizer.tokenize(text)
                vulnerable_list.update(tokens)
        VULNERABLE_TOKENS = vulnerable_list
        VULNERABLE_TOKEN_IDS = set(tokenizer.convert_tokens_to_ids(vulnerable_list))
    return VULNERABLE_TOKEN_IDS

def load_model(model_name, use_float16=True):    
    accelerator = Accelerator()

    def load_specific_model(name, use_float16=False):
        if name == "google-bert/bert-base-uncased" or name == "huawei-noah/TinyBERT_General_4L_312D":
            model = transformers.BertForMaskedLM.from_pretrained(name)
        elif name == "distilbert/distilbert-base-uncased":
            model = transformers.DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
        elif name == "albert/albert-base-v2":
            model = transformers.AlbertForMaskedLM.from_pretrained("albert-base-v2")
        elif name == "google/mobilebert-uncased":
            model = transformers.MobileBertForMaskedLM.from_pretrained("google/mobilebert-uncased")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                name, return_dict=True, trust_remote_code=True,
                torch_dtype = torch.float16 if use_float16 and name != "google/gemma-2-27b-it" else torch.bfloat16,
                device_map="auto",
                attn_implementation="eager"
            )
        return model

    # Load model
    model = load_specific_model(model_name, use_float16)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.eval()

    model = accelerator.prepare(model)

    return model, tokenizer, accelerator

def api_key_setup(key_path):
    openai.api_key = open(key_path, "r").read()

def fix_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)

def get_ll(sentence, model, tokenizer, device, vulnerable_file_path, extra_step = 'none'):
    max_length = model.config.max_position_embeddings


    input_ids = tokenizer.encode(sentence, max_length=max_length, truncation=True)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    input_ids = input_ids.long()
    input_ids = input_ids.to(device, non_blocking=True)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        
    loss, logits = outputs[:2]
    
    if torch.isnan(loss).any():
        return None
    
    if extra_step == "red_list":

        adjusted_logits = apply_red_list(logits, tokenizer, device, vulnerable_file_path)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Recompute loss with adjusted logits
        loss = F.cross_entropy(
            adjusted_logits.view(-1, adjusted_logits.size(-1)),
            input_ids.view(-1),                             
            ignore_index=tokenizer.pad_token_id
        )   

        logits = adjusted_logits
    
    if extra_step == "temp":

        scaled_logits = apply_temp(logits, tokenizer, device, vulnerable_file_path)
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Recompute loss with adjusted logits
        loss = F.cross_entropy(
            scaled_logits.view(-1, scaled_logits.size(-1)),
            input_ids.view(-1),
            ignore_index=tokenizer.pad_token_id
        )

        logits = scaled_logits

    return get_all_prob(input_ids, loss, logits)

def apply_red_list(logits, tokenizer, device, filepath):
    vulnerable_ids = load_vulnerable_tokens(tokenizer, filepath)
    if not vulnerable_ids:
        return logits
    
    penalty_mask = torch.zeros_like(logits, device=device)
    penalty_mask[:, :, list(vulnerable_ids)] = 1
    return logits - penalty_mask

def apply_temp(logits, tokenizer, device, filepath):
    vulnerable_ids = load_vulnerable_tokens(tokenizer, filepath)
    if not vulnerable_ids:
        return logits
    
    temp_mask = torch.ones_like(logits, device=device)
    temp_mask[:, :, list(vulnerable_ids)] = 0.5
    return logits * temp_mask

def get_ll_bert(sentence, model, tokenizer, device, bert_percent_masked, vulnerable_file_path, extra_step = 'none', seed = 42):
    torch.manual_seed(seed)
    random.seed(seed)

    max_length = model.config.max_position_embeddings
    
    input_ids = torch.tensor(tokenizer.encode(sentence, max_length=max_length, truncation=True)).unsqueeze(0)
    input_ids = input_ids.long()
    input_ids = input_ids.to(device, non_blocking=True)

    masked_input_ids = input_ids.clone()

    num_tokens = input_ids.size(1)
    num_masked = int(num_tokens * bert_percent_masked)

    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    real_token_indices = torch.where(attention_mask == 1)[1].tolist()
    masked_indices = torch.tensor(
        random.sample(real_token_indices, min(num_masked, len(real_token_indices))),
        dtype=torch.long,
        device=device
    )
    attention_mask[0, masked_indices] = 0
    masked_input_ids[0, masked_indices] = tokenizer.mask_token_id

    with torch.no_grad():
        outputs = model(masked_input_ids, attention_mask=attention_mask, labels=input_ids)
        loss, logits = outputs[:2]

    if torch.isnan(loss).any():
        return None

    if extra_step == "red_list":
        adjusted_logits = apply_red_list(logits, tokenizer, device, vulnerable_file_path)
        
        # Recompute loss with adjusted logits
        loss = F.cross_entropy(
            adjusted_logits.view(-1, adjusted_logits.size(-1)),
            input_ids.view(-1),
            ignore_index=tokenizer.pad_token_id
        )   

        logits = adjusted_logits
    
    if extra_step == "temp":
        scaled_logits = apply_temp(logits, tokenizer, device, vulnerable_file_path)
        
        # Recompute loss with adjusted logits
        loss = F.cross_entropy(
            scaled_logits.view(-1, scaled_logits.size(-1)),
            input_ids.view(-1),
            ignore_index=tokenizer.pad_token_id
        )   

        logits = scaled_logits

    return get_all_prob(input_ids, loss, logits)

def get_conditional_ll(input_text, target_text, model, tokenizer, device):
    max_length = model.config.max_position_embeddings

    input_encodings = tokenizer(input_text, return_tensors="pt", max_length=max_length//2, truncation=True)
    target_encodings = tokenizer(target_text, return_tensors="pt", max_length=max_length//2, truncation=True)
    concat_ids = torch.cat((input_encodings.input_ids.to(device, non_blocking=True), target_encodings.input_ids.to(device, non_blocking=True)), dim=1)
    concat_ids = concat_ids.long()

    labels = concat_ids.clone().to(device, non_blocking=True)
    labels[:, : input_encodings.input_ids.size(1)] = -100

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    with torch.no_grad():
        outputs = model(concat_ids, labels=labels)
    loss, logits = outputs[:2]
    if torch.isnan(loss).any():
        return None
    return get_all_prob(labels, loss, logits)

def get_all_prob(input_ids, loss, logits):
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    ll = -loss.item()
    ppl = torch.exp(loss).item()
    prob = torch.exp(-loss).item()
    return prob, ll , ppl, all_prob, loss.item()

def inference(model, tokenizer, target_data, prefix, accelerator, num_shots, ex, bert_percent_masked, vulnerable_file_path, extra_step = "none"):
    pred = {}
    max_length = model.config.max_position_embeddings

    device = accelerator.device

    # BERT-based models
    if isinstance(model, transformers.BertForMaskedLM) or isinstance(model, transformers.DistilBertForMaskedLM) or isinstance(model, transformers.AlbertForMaskedLM) or isinstance(model, transformers.MobileBertForMaskedLM):
        # unconditional log-likelihood
        ll = get_ll_bert(target_data, model, tokenizer,device, bert_percent_masked, vulnerable_file_path, extra_step=extra_step)
        if ll == None:
            return None
        else:
            ll = ll[1]
    else:
        # unconditional log-likelihood
        ll = get_ll(target_data, model, tokenizer, device, vulnerable_file_path, extra_step=extra_step)
        if ll == None:
            return None
        else:
            ll = ll[1]
            
        # ReCaLL
        if int(num_shots) != 0:   
            # conditional log-likelihood with prefix     
            ll_nonmember = get_conditional_ll("".join(prefix), target_data, model, tokenizer, device)
            if ll_nonmember == None:
                return None
            else:
                ll_nonmember = ll_nonmember[1]
            pred["recall"] = ll_nonmember / ll

    # baselines 
    pred["loss"] = ll
    pred["zlib"] = ll / len(zlib.compress(bytes(target_data, "utf-8")))
    
    ex["pred"] = pred
    return ex


def generate_tables_to_file(all_output, output_path):
    metrics = all_output[0]["pred"].keys()

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        for metric in metrics:
            writer.writerow([f"Table for Metric: {metric}"])
            
            writer.writerow(["Data Point", "Ground Truth", "Score"])
            
            for ex in all_output:
                writer.writerow([
                    ex["input"],
                    ex["label"],
                    ex["pred"][metric]
                ])
            
            writer.writerow([])

def generate_prompt(example):
    return f"Generate a passage that is similar to the given text in length, domain, and style.\nGiven text:{example}\nPassage :"

def get_completion(prompt):
    message = [{"role": "user", "content": prompt}]
    responses = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=message,
        max_tokens=512,
        temperature=1,
    )
    return responses.choices[0]["message"]["content"]

def gpt_synthetic_prefix (original_prefixes):
    # default generate synthetic prefix from non-member data
    synthetic_prefixes = []
    for original_prefix in original_prefixes:
        prompt = generate_prompt(original_prefix)
        response = get_completion(prompt)      
        synthetic_prefixes.append(response)
    return synthetic_prefixes

    
def process_prefix(target_model, prefix, avg_length, pass_window, total_shots):
    if pass_window:
        return prefix
    max_length = target_model.config.max_position_embeddings
    
    token_counts = [len(tokenizer.encode(shot, max_length=max_length, truncation=True)) for shot in prefix]
    target_token_count = avg_length
    total_tokens = sum(token_counts) + target_token_count
    if total_tokens <= max_length:
        return prefix
    # Determine the maximum number of shots that can fit within the max_length
    max_shots = 0
    cumulative_tokens = target_token_count
    for count in token_counts:
        if cumulative_tokens + count <= max_length:
            max_shots += 1
            cumulative_tokens += count
        else:
            break
    # Truncate the prefix to include only the maximum number of shots
    truncated_prefix = prefix[-max_shots:]
    total_shots = max_shots
    return truncated_prefix

def evaluate_data(test_data, model, tokenizer, prefix, accelerator, total_shots, pass_window, synehtic_prefix, bert_percent_masked, vulnerable_file_path, extra_step = "none"):
    all_output = []
    max_length = model.config.max_position_embeddings
    if int(total_shots) != 0:  
        avg_length = int(np.mean([len(tokenizer.encode(ex["input"], max_length=max_length, truncation=True)) for ex in test_data])) 
        prefix = process_prefix(target_model, prefix, avg_length, pass_window, total_shots) 
        if synehtic_prefix:
            prefix = gpt_synthetic_prefix(prefix)
    for ex in tqdm(test_data):
        new_ex = inference(model, tokenizer, ex["input"], prefix, accelerator, total_shots, ex, bert_percent_masked, vulnerable_file_path, extra_step)
        if new_ex != None:
            all_output.append(new_ex)
    
    return all_output

if __name__ == "__main__":
    fix_seed(42)
    args = Options()
    args = args.parser.parse_args()
    
    output_dir = args.output_dir
    num_data_points = int(args.num_data_points)
    dataset = args.dataset
    target_model = args.target_model
    sub_dataset = args.sub_dataset
    num_shots = args.num_shots
    pass_window = args.pass_window
    synehtic_prefix = args.synehtic_prefix
    api_key_path = args.api_key_path
    vulnerable_file_path = args.vulnerable_file_path
    extra_step = args.post_distillation_step
    bert_percent_masked = 0.15

    if synehtic_prefix and api_key_path is not None:
        api_key_setup(api_key_path)

    print("Preparing dataset...")
    # process and prepare the data
    full_data, nonmember_prefix, member_data_prefix = create_dataset(dataset, sub_dataset, output_dir, num_shots, num_data_points)

    print("Loading models...")
    # load models
    model, tokenizer, accelerator = load_model(target_model)

    print("Evaluating data...")
    # evaluate the data
    all_output = evaluate_data(full_data, model, tokenizer, nonmember_prefix, accelerator, num_shots, pass_window, synehtic_prefix, bert_percent_masked, vulnerable_file_path, extra_step)

    print("Saving outputs...")
    # save the results
    if num_data_points > 0:
        data_limit_suffix = "_limit_"+str(num_data_points)+"_each"
    else:
        data_limit_suffix = ""
    
    if extra_step != "none":
        extra_step_suffix = "_"+extra_step
    else:
        extra_step_suffix = ""

    all_output_path = os.path.join(output_dir, f"{dataset}", f"{target_model.split('/')[-1]}", f"{sub_dataset}", f"{num_shots}_shot_{str(sub_dataset)+data_limit_suffix+extra_step_suffix}.csv")
    
    os.makedirs(os.path.dirname(all_output_path), exist_ok=True)
    generate_tables_to_file(all_output, all_output_path)
    print(f"Saved results to {all_output_path}")