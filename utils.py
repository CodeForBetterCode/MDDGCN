import torch
import numpy as np
import pandas as pd
import json

def load_protein_to_idx(filepath="./data/protein_to_idx.json"):
    with open(filepath, "r") as f:
        return json.load(f)

def initialize_labels(num_nodes=17201):
    return torch.full((num_nodes,), -1, dtype=torch.long)

def load_probably_positive_samples(protein_to_idx, positive_samples_set):
    df = pd.read_csv('./data/MDD_STRINGID/potential_positive_sample.csv', sep='\t', header=0)
    mapped = {
        protein_to_idx[row['stringId']]
        for _, row in df.iterrows()
        if row['stringId'] in protein_to_idx and protein_to_idx[row['stringId']] not in positive_samples_set
    }
    return np.array(list(mapped))

def set_labels(y, indices, label_value):
    y[indices] = label_value

def PartC_PPST(data, experiment=0):
    protein_to_idx = load_protein_to_idx()
    y = initialize_labels()

    positive_samples = np.where(data.y == 1)[0]
    set_labels(y, positive_samples, 1)

    mapped_proteins = load_probably_positive_samples(protein_to_idx, set(positive_samples))

    all_indices = np.arange(len(data.y))
    all_negative_samples = np.setdiff1d(all_indices, np.concatenate([positive_samples, mapped_proteins]))

    np.random.seed(42 + experiment)
    negative_samples = np.random.choice(all_negative_samples, size=len(positive_samples), replace=False)
    set_labels(y, negative_samples, 0)

    train_indices = np.concatenate([positive_samples, negative_samples])
    test_negative_samples = np.setdiff1d(all_negative_samples, negative_samples)

    set_labels(y, test_negative_samples, 0)
    set_labels(y, mapped_proteins, 1)

    data.y = y

    print("train positive sample size：", len(positive_samples))
    print("train negative sample size：", len(negative_samples))
    print("test positive sample size：", len(mapped_proteins))
    print("test negative sample size：", len(test_negative_samples))
    print("train sample size：", train_indices.shape[0])
    print("test sample size：", (mapped_proteins.shape[0] + test_negative_samples.shape[0]))

    return data, train_indices, np.concatenate([mapped_proteins, test_negative_samples])

def NNPSampling(data, experiment=0):
    protein_to_idx = load_protein_to_idx()
    y = initialize_labels()

    positive_samples = np.where(data.y == 1)[0]
    set_labels(y, positive_samples, 1)

    np.random.seed(42 + experiment)
    test_positive_samples_size = round(len(positive_samples) * 0.1)
    test_positive_samples = np.random.choice(positive_samples, size=test_positive_samples_size, replace=False)

    mapped_proteins = load_probably_positive_samples(protein_to_idx, set(positive_samples))
    all_indices = np.arange(len(data.y))
    all_negative_samples = np.setdiff1d(all_indices, positive_samples)

    train_data_positive = np.setdiff1d(positive_samples, test_positive_samples)
    negative_samples = np.random.choice(all_negative_samples, size=len(train_data_positive), replace=False)
    set_labels(y, negative_samples, 0)

    test_negative_samples = np.setdiff1d(all_indices, positive_samples)
    set_labels(y, test_negative_samples, 0)

    data.y = y

    print(f'positive sample size：{np.sum(y == 1)}')
    print(f'negative sample size{np.sum(y == 0)}')
    print("train positive sample size：", len(train_data_positive))
    print("train negative sample size：", len(negative_samples))
    print("test positive sample size：", len(test_positive_samples))
    print("test negative sample size：", len(test_negative_samples))
    print("train sample size：", train_data_positive.shape[0] + negative_samples.shape[0])
    print("test sample size：", test_positive_samples.shape[0] + test_negative_samples.shape[0])

    return data, np.concatenate([train_data_positive, negative_samples]), np.concatenate([test_positive_samples, test_negative_samples])

def PNSampling(data, experiment=0):
    protein_to_idx = load_protein_to_idx()
    y = initialize_labels()

    positive_samples = np.where(data.y == 1)[0]
    set_labels(y, positive_samples, 1)

    np.random.seed(42 + experiment)
    test_positive_samples_size = round(len(positive_samples) * 0.1)
    test_positive_samples = np.random.choice(positive_samples, size=test_positive_samples_size, replace=False)

    mapped_proteins = load_probably_positive_samples(protein_to_idx, set(positive_samples))
    all_indices = np.arange(len(data.y))
    all_negative_samples = np.setdiff1d(all_indices, np.concatenate([positive_samples, mapped_proteins]))

    train_data_positive = np.setdiff1d(positive_samples, test_positive_samples)
    negative_samples = np.random.choice(all_negative_samples, size=len(train_data_positive), replace=False)
    set_labels(y, negative_samples, 0)

    test_negative_samples = np.setdiff1d(all_indices, positive_samples)
    set_labels(y, test_negative_samples, 0)

    data.y = y

    print(f'positive sample size：{np.sum(y == 1)}')
    print(f'negative sample size{np.sum(y == 0)}')
    print("train positive sample size：", len(train_data_positive))
    print("train negative sample size：", len(negative_samples))
    print("test positive sample size：", len(test_positive_samples))
    print("test negative sample size：", len(test_negative_samples))
    print("train sample size：", train_data_positive.shape[0] + negative_samples.shape[0])
    print("test sample size：", test_positive_samples.shape[0] + test_negative_samples.shape[0])

    return data, np.concatenate([train_data_positive, negative_samples]), np.concatenate([test_positive_samples, test_negative_samples])
