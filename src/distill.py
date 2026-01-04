import argparse
import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    default_data_collator,
)
from torch.optim import AdamW
from types import SimpleNamespace


# --------------------
# Architectural helpers (NoNorm, bottleneck, replacements)
# --------------------
class NoNorm(nn.Module):
    def __init__(self, normalized_shape, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, (list, tuple)):
            n = normalized_shape[0]
        else:
            n = int(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(n))
            self.bias = nn.Parameter(torch.zeros(n))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        if self.elementwise_affine:
            return x * self.weight + self.bias
        else:
            return x

def make_bottleneck_mlp(in_dim, bottleneck_dim, out_dim, activation_fn=nn.GELU):
    return nn.Sequential(
        nn.Linear(in_dim, bottleneck_dim),
        activation_fn(),
        nn.Linear(bottleneck_dim, out_dim),
    )

def replace_layernorm_with_nonorm(model, initialize_new=True):
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.LayerNorm):
            # find parent and attribute name
            path = name.split('.')
            parent = model
            for p in path[:-1]:
                parent = getattr(parent, p)
            attr = path[-1]
            nonorm = NoNorm(module.normalized_shape, elementwise_affine=True)
            # init to LayerNorm-like defaults
            with torch.no_grad():
                if nonorm.elementwise_affine:
                    nonorm.weight.fill_(1.0)
                    nonorm.bias.zero_()
            setattr(parent, attr, nonorm)
            replaced += 1
    print(f"[NoNorm] Replaced {replaced} LayerNorm modules with NoNorm")
    return model


def apply_bottleneck(model, bottleneck_dim=384, init_only_new=True):
    cfg = getattr(model, "config", SimpleNamespace())
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or getattr(cfg, "d_model", None)
    intermediate_size = getattr(cfg, "intermediate_size", None) or getattr(cfg, "ffn_dim", None) or (4 * hidden_size if hidden_size else None)

    replacements = 0
    # Heuristics for common ffn naming: c_fc / fc_in / intermediate.dense
    for parent_name, parent_module in model.named_modules():
        for attr in dir(parent_module):
            if attr.startswith("_"):
                continue
            try:
                child = getattr(parent_module, attr)
            except Exception:
                continue
            if not isinstance(child, nn.Module):
                continue

            # GPT-style container: has attributes c_fc and c_proj inside a child module (child is the block)
            if hasattr(child, "c_fc") and isinstance(getattr(child, "c_fc"), nn.Linear):
                lin = getattr(child, "c_fc")
                H, I = lin.in_features, lin.out_features
                B = min(bottleneck_dim, H)
                new_mlp = make_bottleneck_mlp(H, B, I, activation_fn=nn.GELU)
                setattr(child, "c_fc", new_mlp)
                replacements += 1
                continue

            # GPT-NeoX style fc_in / fc_out
            if hasattr(child, "fc_in") and isinstance(getattr(child, "fc_in"), nn.Linear):
                lin = getattr(child, "fc_in")
                H, I = lin.in_features, lin.out_features
                B = min(bottleneck_dim, H)
                new_mlp = make_bottleneck_mlp(H, B, I, activation_fn=nn.GELU)
                setattr(child, "fc_in", new_mlp)
                replacements += 1
                continue

            # BERT-style: intermediate.dense
            if hasattr(child, "intermediate"):
                inter = getattr(child, "intermediate")
                if hasattr(inter, "dense") and isinstance(getattr(inter, "dense"), nn.Linear):
                    lin = getattr(inter, "dense")
                    H, I = lin.in_features, lin.out_features
                    B = min(bottleneck_dim, H)
                    new_mlp = make_bottleneck_mlp(H, B, I, activation_fn=nn.GELU)
                    setattr(inter, "dense", new_mlp)
                    replacements += 1
                    continue

            # Generic fallback: replace Linear layers that exactly match hidden_size -> intermediate_size
            if isinstance(child, nn.Linear) and hidden_size and intermediate_size:
                if child.in_features == hidden_size and child.out_features == intermediate_size:
                    H, I = child.in_features, child.out_features
                    B = min(bottleneck_dim, H)
                    new_mlp = make_bottleneck_mlp(H, B, I, activation_fn=nn.GELU)
                    setattr(parent_module, attr, new_mlp)
                    replacements += 1
                    continue

    # Initialize only the newly created Linear layers
    def _init_new(m):
        # If m belongs to Sequential created above, initialize its Linear modules
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
    model.apply(_init_new)
    print(f"[Bottleneck] Replaced {replacements} FFN first-projection(s) with H->B->I bottlenecks (B={bottleneck_dim})")
    try:
        model.config.bottleneck_dim = bottleneck_dim
    except Exception:
        pass
    return model

def apply_selected_architectures(student_model, args):
    mode = getattr(args, "mode", None)
    # interpret mode priority: explicit mode overrides boolean flags
    if mode is not None and mode != "none":
        if mode == "bottleneck":
            apply_bottleneck(student_model, bottleneck_dim=args.bottleneck_dim)
        elif mode == "nonorm":
            replace_layernorm_with_nonorm(student_model)
        elif mode == "all":
            apply_bottleneck(student_model, bottleneck_dim=args.bottleneck_dim)
            replace_layernorm_with_nonorm(student_model)
        else:
            print(f"[apply_selected_architectures] Unknown mode: {mode} (no changes applied)")
    else:
        if getattr(args, "bottleneck", False):
            apply_bottleneck(student_model, bottleneck_dim=args.bottleneck_dim)
        if getattr(args, "nonorm", False):
            replace_layernorm_with_nonorm(student_model)
    return student_model


def reinitialize_model_weights(model):
    config = getattr(model, "config", None)
    init_range = getattr(config, "initializer_range", 0.02)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=init_range)
            if getattr(module, "padding_idx", None) is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, NoNorm):
            if module.elementwise_affine:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)


def distillation_loss(student_logits, teacher_logits, labels, pad_token_id, T=1.0, alpha=0.5):
    vocab = student_logits.size(-1)

    # soft targets KL: note kl_div expects log-prob (input) and probs (target)
    s_log_probs = F.log_softmax(student_logits / T, dim=-1)
    t_probs = F.softmax(teacher_logits / T, dim=-1)
    kl_loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T * T)

    # hard targets CE, ignoring padding tokens
    ce_loss = F.cross_entropy(
        student_logits.view(-1, vocab),
        labels.view(-1),
        ignore_index=pad_token_id,
    )

    return alpha * kl_loss + (1.0 - alpha) * ce_loss, kl_loss.detach(), ce_loss.detach()

START_SIGNAL = ""
END_SIGNAL = None

def load_start_signal_file(path, start_signal=START_SIGNAL, end_signal=END_SIGNAL):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if not line.startswith(start_signal):
                continue
            text = line[len(start_signal):].strip()
            if end_signal and text.endswith(end_signal):
                text = text[: -len(end_signal)].strip()
            examples.append(text)
    return examples

def compute_ppl(model, dl, device, pad_token_id):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            vocab = logits.size(-1)
            loss = F.cross_entropy(
                logits.view(-1, vocab),
                labels.view(-1),
                ignore_index=pad_token_id,
                reduction="sum"
            )
            # count non-pad tokens
            nonpad = (labels != pad_token_id).sum().item()
            total_loss += loss.item()
            total_tokens += nonpad
    model.train()
    ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
    return ppl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-name", default="EleutherAI/pythia-160m")
    parser.add_argument("--student-name", default="crumb/distilpythia")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--mode", choices=["bottleneck", "nonorm", "all", "none"], default="none")
    parser.add_argument("--bottleneck-dim", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--log-steps", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Using device:", device)

    # I/O Paths
    TEXT_FILE = args.dataset
    save_path = args.save_path
    


    # Load tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_name)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_name)
    student = AutoModelForCausalLM.from_pretrained(args.student_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    # freeze teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Apply architectural edits to student (these functions take care of initializing new layers)
    args.bottleneck_dim = args.bottleneck_dim
    student = apply_selected_architectures(student, args)

    print("[Init] Reinitializing student weights from scratch")
    reinitialize_model_weights(student)

    # Ensure only student params are trainable (teacher is frozen)
    for p in student.parameters():
        p.requires_grad = True

    
    # Load dataset
    texts = load_start_signal_file(TEXT_FILE, START_SIGNAL, END_SIGNAL)
    print(f"Loaded {len(texts)} examples from {TEXT_FILE}")
    if len(texts) == 0:
        raise ValueError("No examples loaded. Check TEXT_FILE and START_SIGNAL")

    ds = Dataset.from_dict({"text": texts})

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.max_length)

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=default_data_collator)

    # small held-out set for PPL (use a subset)
    heldout_ds = ds.select(range(min(1000, len(ds))))
    heldout_dl = DataLoader(heldout_ds, batch_size=args.batch_size, shuffle=False, collate_fn=default_data_collator)

    # Optimizer & scheduler
    optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(dataloader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    # move models
    teacher.to(device)
    student.to(device)
    student.train()

    # training loop
    global_step = 0
    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone().to(device)

            # teacher logits (no grad)
            with torch.no_grad():
                t_out = teacher(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = t_out.logits

            s_out = student(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = s_out.logits

            loss, kl_term, ce_term = distillation_loss(
                student_logits, teacher_logits, labels,
                pad_token_id=pad_token_id,
                T=args.temperature,
                alpha=args.alpha
            )

            optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % args.log_steps == 0:
                avg_loss = running_loss / args.log_steps
                running_loss = 0.0
                # compute a quick heldout ppl
                ppl = compute_ppl(student, heldout_dl, device, pad_token_id)
                print(f"Epoch {epoch+1}/{args.epochs} step {global_step}/{total_steps} — loss {avg_loss:.4f} (kl {kl_term:.4f}, ce {ce_term:.4f}) heldout_ppl {ppl:.2f}")

        # epoch-end eval
        ppl = compute_ppl(student, heldout_dl, device, pad_token_id)
        print(f"==> End epoch {epoch+1}: heldout_ppl = {ppl:.2f}")

    # final save
    os.makedirs(save_path, exist_ok=True)
    student.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Finished training. Student model saved to {save_path}")

if __name__ == "__main__":
    main()
