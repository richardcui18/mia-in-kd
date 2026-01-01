from datasets import load_dataset
import os
import random

def create_dataset(dataset_name,sub_dataset_name, output_dir, num_shots, num_data_points):
    if num_data_points > 0:
        limit_data_size = True
    else:
        limit_data_size = False

    if dataset_name == "wikimia":
        dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{sub_dataset_name}")
        member_data = []
        nonmember_data = []
        for data in dataset:
            if data["label"] == 1:
                member_data.append(data["input"])
            elif data["label"] == 0:
                nonmember_data.append(data["input"])
        
        if limit_data_size:
            member_data = member_data[:num_data_points]
            nonmember_data = nonmember_data[:num_data_points]
        
        # shuffle the datasets
        random.shuffle(member_data)
        random.shuffle(nonmember_data)
        num_shots = int(num_shots)

        nonmember_prefix = nonmember_data[:num_shots]
        nonmember_data = nonmember_data[num_shots:]
        member_data_prefix = member_data[:num_shots]
        member_data = member_data[num_shots:]

    elif dataset_name == "arxiv":
        arxiv_dataset = load_dataset("iamgroot42/mimir", "arxiv", split="ngram_13_0.8", trust_remote_code=True)

        member_data = []
        nonmember_data = []
        for data in arxiv_dataset:
            member_data.append(data["member"])
            nonmember_data.append(data["nonmember"])

        if limit_data_size:
            member_data = member_data[:num_data_points]
            nonmember_data = nonmember_data[:num_data_points]

        # Ensure both datasets are of similar size
        min_size = min(len(member_data), len(nonmember_data))
        member_data = member_data[:min_size]
        nonmember_data = nonmember_data[:min_size]

        # Shuffle the datasets
        random.shuffle(member_data)
        random.shuffle(nonmember_data)
        num_shots = int(num_shots)

        nonmember_prefix = nonmember_data[:num_shots]
        nonmember_data = nonmember_data[num_shots:]
        member_data_prefix = member_data[:num_shots]
        member_data = member_data[num_shots:]

    elif dataset_name == "openwebtext":
        # Load OpenWebText for member data
        openwebtext_dataset = load_dataset("Skylion007/openwebtext", split="train", trust_remote_code=True)
        member_data = [example["text"] for example in openwebtext_dataset]

        # Load WebInstructSub (May 2024) for non-member data
        WebInstructSub_dataset = load_dataset("chargoddard/WebInstructSub-prometheus", split="train", trust_remote_code=True)
        nonmember_data = [example["generation"] for example in WebInstructSub_dataset if example["generation"].strip() or len(example["generation"])>=3]

        if limit_data_size:
            member_data = member_data[:num_data_points]
            nonmember_data = nonmember_data[:num_data_points]

        # Ensure both datasets are of similar size
        min_size = min(len(member_data), len(nonmember_data))
        member_data = member_data[:min_size]
        nonmember_data = nonmember_data[:min_size]

        # Shuffle the datasets
        random.shuffle(member_data)
        random.shuffle(nonmember_data)
        num_shots = int(num_shots)

        nonmember_prefix = nonmember_data[:num_shots]
        nonmember_data = nonmember_data[num_shots:]
        member_data_prefix = member_data[:num_shots]
        member_data = member_data[num_shots:]
    
    elif dataset_name == "bookcorpus":
        bookcorpus_dataset = load_dataset("bookcorpus", trust_remote_code=True)['train']
        member_data = [example["text"] for example in bookcorpus_dataset]

        # Load WebInstructSub (May 2024) for non-member data
        WebInstructSub_dataset = load_dataset("chargoddard/WebInstructSub-prometheus", split="train", trust_remote_code=True)
        nonmember_data = [example["generation"] for example in WebInstructSub_dataset if example["generation"].strip() or len(example["generation"])>=3]

        if limit_data_size:
            member_data = member_data[:num_data_points]
            nonmember_data = nonmember_data[:num_data_points]

        # Ensure both datasets are of similar size
        min_size = min(len(member_data), len(nonmember_data))
        member_data = member_data[:min_size]
        nonmember_data = nonmember_data[:min_size]

        # Shuffle the datasets
        random.shuffle(member_data)
        random.shuffle(nonmember_data)
        num_shots = int(num_shots)

        nonmember_prefix = nonmember_data[:num_shots]
        nonmember_data = nonmember_data[num_shots:]
        member_data_prefix = member_data[:num_shots]
        member_data = member_data[num_shots:]
    
    else: 
        raise ValueError(f"Unknown dataset: {dataset_name}. Please modify the code to include the dataset. Make sure the dataset is in the same format.")

    full_data = [] 
    # binary classification, the data need to be balanced. 
    if "only" in dataset_name or "both" in dataset_name:
        for m_data in member_data:
            full_data.append({"input": m_data, "label": 1})
    else:
        for nm_data, m_data in zip(nonmember_data, member_data):
            full_data.append({"input": nm_data, "label": 0})
            full_data.append({"input": m_data, "label": 1})

    return full_data, nonmember_prefix, member_data_prefix