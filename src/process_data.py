from datasets import load_dataset
import os
import random

def create_dataset(dataset_name,sub_dataset_name, output_dir, num_shots):
    if dataset_name == "wikimia":
        dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{sub_dataset_name}")
        member_data = []
        nonmember_data = []
        for data in dataset:
            if data["label"] == 1:
                member_data.append(data["input"])
            elif data["label"] == 0:
                nonmember_data.append(data["input"])
        
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