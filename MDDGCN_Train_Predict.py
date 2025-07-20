import argparse
import csv
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
import xgboost as xgb
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR,StepLR
from torch_geometric.data import Data
from utils import PartC_PPST,NNPSampling,PNSampling


random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Model Training')

parser.add_argument('--modeltype', type=str, default='MDDGCN', help='model of prdict',choices=['GCN', 'GAT', 'ChebNet', 'GraphSAGE','MDDGCN','CGMega','EMOGI'])
parser.add_argument('--epochs', type=int, default=2500, help='Number of epochs to train the model')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for the optimizer')
parser.add_argument('--hidden_channels',type=int, default=128, help='Number of hidden channels in the model')
parser.add_argument('--intermediate_channels', type=int, default=128, help='Number of intermediate_channels in the model')
parser.add_argument('--attention_hidden', type=int, default=128, help='Number of attention_hidden in the model')
parser.add_argument('--ChetK',type=int, default=2, help='ChebNet of K')
parser.add_argument('--GATHeads',type=int, default=2, help='GAT of heads')
parser.add_argument('--scheduler', type=str, default='StepLR', help='model of scheduler',choices=['StepLR', 'CosineAnnealingLR'])


args = parser.parse_args()
# for the CGMega and EMOGI models, the train and evaluate functions should make some modifications.
def train(model, optimizer, loss_fn, train_data, train_labels, threshold=0.60):
    model.train()
    optimizer.zero_grad()
    logits = model(data) 
    out = logits[train_data]  
    loss = loss_fn(out, train_labels.unsqueeze(1).float())
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(model, loss_fn, val_data, val_labels):
    model.eval()
    with torch.no_grad():
        out = model(data)
        loss = loss_fn(out[val_data], val_labels.unsqueeze(1).float())
        pred = torch.sigmoid(out[val_data])
        pred_labels = (pred > 0.5).long()
        return pred_labels, loss.item(), pred[:, 0]

def compute_metrics(true_labels, pred_labels, pred_probs):
    accuracy = accuracy_score(true_labels.cpu(), pred_labels.cpu())
    auc = roc_auc_score(true_labels.cpu(), pred_probs.cpu())
    ap = average_precision_score(true_labels.cpu(), pred_probs.cpu())
    f1 = f1_score(true_labels.cpu(), pred_labels.cpu())
    precision = precision_score(true_labels.cpu(), pred_labels.cpu())
    recall = recall_score(true_labels.cpu(), pred_labels.cpu())
    
    precisions, recalls, thresholds = precision_recall_curve(true_labels.cpu(), pred_probs.cpu())

    fixed_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    recall_values = []

    for fixed_thr in fixed_thresholds:
        closest_idx = (abs(thresholds - fixed_thr)).argmin()
        recall_values.append(recalls[closest_idx])

    top_k_values = [10, 20, 30, 40, 50]
    top_k_precisions = []
    top_k_recalls = []

    for k in top_k_values:
        top_k_indices = torch.topk(pred_probs.clone().detach(), k).indices  
        new_pred_labels = torch.zeros_like(pred_labels.clone().detach())
        new_pred_labels[top_k_indices] = 1  

        top_k_precisions.append(precision_score(true_labels.cpu(), new_pred_labels.cpu()))
        top_k_recalls.append(recall_score(true_labels.cpu(), new_pred_labels.cpu()))

    return (
        accuracy, auc, ap, f1, precision, recall,
        recall_values[0], recall_values[1], recall_values[2], recall_values[3], recall_values[4],
        top_k_precisions[0], top_k_precisions[1], top_k_precisions[2], top_k_precisions[3], top_k_precisions[4],
        top_k_recalls[0], top_k_recalls[1], top_k_recalls[2], top_k_recalls[3], top_k_recalls[4]
    )

from model import MDDGCN
# model initialization
def initialize_model(args):
    feature_dims =torch.tensor( [1273, 185, 377, 458, 192] , device=device) # 
    if 'MDDGCN' in args.modeltype:    
        model = MDDGCN(in_channels=data.x.shape[1],hidden_channels=args.hidden_channels, intermediate_channels=args.intermediate_channels, out_channels=1, feature_dims=feature_dims,feature_weights=feature_weights,K=args.ChetK,attention_hidden=args.attention_hidden,perturb_features_p=args.perturb_features_p,dropout_edge_p=args.dropout_edge_p).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs) 
    if 'StepLR' in args.scheduler:
        # every 100 epochs, learning rate decay to 0.9 of original    
        scheduler = StepLR(optimizer, step_size=100, gamma=0.9) 
    elif 'CosineAnnealingLR' in args.scheduler:
        # the T_max  can be adjusted according to need
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # use suitable binary classification loss function
    class_weights = torch.tensor([1.0], dtype=torch.float32).to(device)  
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights) 
    return model, optimizer, loss_fn, scheduler

# MODEL_LIST = ['GCN', 'GAT', 'GraphSAGE','ChebNet','MDDGCN','CGMega','MLP','EMOGI']
MODEL_LIST = ['MDDGCN']

metrics = {model: {'ACC': np.zeros((10, 10)),
                   'AUC': np.zeros((10, 10)),
                   'AUPR': np.zeros((10, 10)),
                   'F1': np.zeros((10, 10)),
                   'Precision': np.zeros((10, 10)),
                   'Recall': np.zeros((10, 10)),
                   'Recall5': np.zeros((10, 10)),
                   'Recall6': np.zeros((10, 10)),
                   'Recall7': np.zeros((10, 10)),
                   'Recall8': np.zeros((10, 10)),
                   'Recall9': np.zeros((10, 10)),
                   'PrecisionTop10': np.zeros((10, 10)),
                   'PrecisionTop20': np.zeros((10, 10)),
                   'PrecisionTop30': np.zeros((10, 10)),
                   'PrecisionTop40': np.zeros((10, 10)),
                   'PrecisionTop50': np.zeros((10, 10)),
                   'RecallTop10': np.zeros((10, 10)),
                   'RecallTop20': np.zeros((10, 10)),
                   'RecallTop30': np.zeros((10, 10)),
                   'RecallTop40': np.zeros((10, 10)),
                   'RecallTop50': np.zeros((10, 10))} for model in MODEL_LIST}
                   
# deepwalk = torch.load('deepwalk_128.pt')
for experiment in range(10):
    print(f'Experiment {experiment + 1}')
    data = torch.load('./data/MDDGCN_PPI_Features_Labels.pt')
    data.to('cpu')

    data,train_data,test_data = build_dataset_Part4_Random(data,experiment)
    data.to(device)
    train_nodes_fold = train_data
    val_nodes_fold =test_data
    train_labels = data.y[train_data]
    val_labels = data.y[test_data] 
    
    for model_name in MODEL_LIST:
        print(f'Training model: {model_name}')

        acc_sum, auc_sum, ap_sum, f1_sum = 0, 0, 0, 0
        precision_sum, recall_sum = 0, 0
        recall5_sum, recall6_sum, recall7_sum, recall8_sum, recall9_sum = 0, 0, 0, 0, 0
        top10_precision_sum, top20_precision_sum, top30_precision_sum, top40_precision_sum, top50_precision_sum = 0, 0, 0, 0, 0
        top10_recall_sum, top20_recall_sum, top30_recall_sum, top40_recall_sum, top50_recall_sum = 0, 0, 0, 0, 0

        if model_name in ['MLP']:
            deepwalk_new = deepwalk.to(device)
            data_new = data.clone().to(device) 
            data_new.x = torch.cat([data_new.x, deepwalk_new], dim=1)
            print(data_new)   

        if model_name in ['GCN', 'GAT', 'GraphSAGE','ChebNet','MDDGCN']:
            data_new = data.clone().to(device) 
            print(data_new)                 

        if model_name in ['MLP','GCN', 'GAT', 'GraphSAGE','ChebNet','MDDGCN']:
            print(f'The training  model is{model_name},data is{data_new}')
            
            for train_iter in tqdm(range(10), desc=f'{model_name} Training Iterations', leave=False):
                
                model, optimizer, loss_fn, scheduler = initialize_model(args,model_name,data_new)
                
                for epoch in range(args.epochs): 
                    loss = train(model, optimizer, loss_fn, train_nodes_fold, train_labels,data_new)
                    scheduler.step() 
                
                pred_labels, val_loss, pred_probs = evaluate(model, loss_fn, val_nodes_fold, val_labels,data_new)
                (accuracy, auc, ap, f1, precision, recall, recall5,recall6,recall7,recall8,recall9, 
                top10_precision, top20_precision, top30_precision, top40_precision, top50_precision, 
                top10_recall, top20_recall, top30_recall, top40_recall, top50_recall) = compute_metrics(val_labels, pred_labels, pred_probs)
                
                print(f'[{model_name}] Epoch {epoch}, Loss: {loss:.4f}, Val_Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f},Top10Precision: {top10_precision:.4f}, Top10Recall: {top10_recall:.4f}')
               
                acc_sum += accuracy
                auc_sum += auc
                ap_sum += ap
                f1_sum += f1
                precision_sum += precision
                recall_sum += recall
                recall5_sum += recall5
                recall6_sum += recall6
                recall7_sum += recall7
                recall8_sum += recall8
                recall9_sum += recall9
                top10_precision_sum += top10_precision
                top20_precision_sum += top20_precision
                top30_precision_sum += top30_precision
                top40_precision_sum += top40_precision
                top50_precision_sum += top50_precision
                top10_recall_sum += top10_recall
                top20_recall_sum += top20_recall
                top30_recall_sum += top30_recall
                top40_recall_sum += top40_recall
                top50_recall_sum += top50_recall
                
        metrics[model_name]['ACC'][experiment] = round(acc_sum / 10, 4)
        metrics[model_name]['AUC'][experiment] = round(auc_sum / 10, 4)
        metrics[model_name]['AUPR'][experiment] = round(ap_sum / 10, 4)
        metrics[model_name]['F1'][experiment] = round(f1_sum / 10, 4)
        metrics[model_name]['Precision'][experiment] = round(precision_sum / 10, 4)
        metrics[model_name]['Recall'][experiment] = round(recall_sum / 10, 4)
        metrics[model_name]['Recall5'][experiment] = round(recall5_sum / 10, 4)
        metrics[model_name]['Recall6'][experiment] = round(recall6_sum / 10, 4)
        metrics[model_name]['Recall7'][experiment] = round(recall7_sum / 10, 4)
        metrics[model_name]['Recall8'][experiment] = round(recall8_sum / 10, 4)
        metrics[model_name]['Recall9'][experiment] = round(recall9_sum / 10, 4)
        metrics[model_name]['PrecisionTop10'][experiment] = round(top10_precision_sum / 10, 4)
        metrics[model_name]['PrecisionTop20'][experiment] = round(top20_precision_sum / 10, 4)
        metrics[model_name]['PrecisionTop30'][experiment] = round(top30_precision_sum / 10, 4)
        metrics[model_name]['PrecisionTop40'][experiment] = round(top40_precision_sum / 10, 4)
        metrics[model_name]['PrecisionTop50'][experiment] = round(top50_precision_sum / 10, 4)
        metrics[model_name]['RecallTop10'][experiment] = round(top10_recall_sum / 10, 4)
        metrics[model_name]['RecallTop20'][experiment] = round(top20_recall_sum / 10, 4)
        metrics[model_name]['RecallTop30'][experiment] = round(top30_recall_sum / 10, 4)
        metrics[model_name]['RecallTop40'][experiment] = round(top40_recall_sum / 10, 4)
        metrics[model_name]['RecallTop50'][experiment] = round(top50_recall_sum / 10, 4)
            
# compute average metrics for each model
for model_name in MODEL_LIST:
    print(f'Final Metrics for {model_name}:')
    avg_acc = np.mean(metrics[model_name]['ACC'])
    avg_auc = np.mean(metrics[model_name]['AUC'])
    avg_aupr = np.mean(metrics[model_name]['AUPR'])
    avg_f1 = np.mean(metrics[model_name]['F1'])
    avg_precision = np.mean(metrics[model_name]['Precision'])
    avg_recall = np.mean(metrics[model_name]['Recall'])
    avg_recall5 = np.mean(metrics[model_name]['Recall5'])
    avg_recall6 = np.mean(metrics[model_name]['Recall6'])
    avg_recall7 = np.mean(metrics[model_name]['Recall7'])
    avg_recall8 = np.mean(metrics[model_name]['Recall8'])
    avg_recall9 = np.mean(metrics[model_name]['Recall9'])
    avg_precision_top10 = np.mean(metrics[model_name]['PrecisionTop10'])
    avg_precision_top20 = np.mean(metrics[model_name]['PrecisionTop20'])
    avg_precision_top30 = np.mean(metrics[model_name]['PrecisionTop30'])
    avg_precision_top40 = np.mean(metrics[model_name]['PrecisionTop40'])
    avg_precision_top50 = np.mean(metrics[model_name]['PrecisionTop50'])
    avg_recall_top10 = np.mean(metrics[model_name]['RecallTop10'])
    avg_recall_top20 = np.mean(metrics[model_name]['RecallTop20'])
    avg_recall_top30 = np.mean(metrics[model_name]['RecallTop30'])
    avg_recall_top40 = np.mean(metrics[model_name]['RecallTop40'])
    avg_recall_top50 = np.mean(metrics[model_name]['RecallTop50'])

    print(f'Accuracy: {avg_acc:.4f}, AUC: {avg_auc:.4f}, AUPR: {avg_aupr:.4f}, F1: {avg_f1:.4f}, '
          f'Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, Recall5: {avg_recall5:.4f}, '
          f'Recall6: {avg_recall6:.4f}, Recall7: {avg_recall7:.4f}, Recall8: {avg_recall8:.4f}, Recall9: {avg_recall9:.4f}, '
          f'PrecisionTop10: {avg_precision_top10:.4f}, PrecisionTop20: {avg_precision_top20:.4f}, '
          f'PrecisionTop30: {avg_precision_top30:.4f}, PrecisionTop40: {avg_precision_top40:.4f}, PrecisionTop50: {avg_precision_top50:.4f}, '
          f'RecallTop10: {avg_recall_top10:.4f}, RecallTop20: {avg_recall_top20:.4f}, '
          f'RecallTop30: {avg_recall_top30:.4f}, RecallTop40: {avg_recall_top40:.4f}, RecallTop50: {avg_recall_top50:.4f}\n')

import os
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

folder_name = "result_part_C/D"

os.makedirs(folder_name, exist_ok=True)

file_name = f"model_metrics3_{timestamp}.csv"

file_path = os.path.join(folder_name, file_name)


with open(file_path, 'w', newline='') as csvfile:

    fieldnames = [
        'Model', 'Accuracy', 'AUC', 'AUPR', 'F1', 'Precision', 'Recall',
        'Recall5', 'Recall6', 'Recall7', 'Recall8', 'Recall9',
        'PrecisionTop10', 'PrecisionTop20', 'PrecisionTop30', 'PrecisionTop40', 'PrecisionTop50',
        'RecallTop10', 'RecallTop20', 'RecallTop30', 'RecallTop40', 'RecallTop50'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for model_name in MODEL_LIST:

        avg_acc = np.mean(metrics[model_name]['ACC'])
        avg_auc = np.mean(metrics[model_name]['AUC'])
        avg_aupr = np.mean(metrics[model_name]['AUPR'])
        avg_f1 = np.mean(metrics[model_name]['F1'])
        avg_precision = np.mean(metrics[model_name]['Precision'])
        avg_recall = np.mean(metrics[model_name]['Recall'])
        avg_recall5 = np.mean(metrics[model_name]['Recall5'])
        avg_recall6 = np.mean(metrics[model_name]['Recall6'])
        avg_recall7 = np.mean(metrics[model_name]['Recall7'])
        avg_recall8 = np.mean(metrics[model_name]['Recall8'])
        avg_recall9 = np.mean(metrics[model_name]['Recall9'])
        avg_precision_top10 = np.mean(metrics[model_name]['PrecisionTop10'])
        avg_precision_top20 = np.mean(metrics[model_name]['PrecisionTop20'])
        avg_precision_top30 = np.mean(metrics[model_name]['PrecisionTop30'])
        avg_precision_top40 = np.mean(metrics[model_name]['PrecisionTop40'])
        avg_precision_top50 = np.mean(metrics[model_name]['PrecisionTop50'])
        avg_recall_top10 = np.mean(metrics[model_name]['RecallTop10'])
        avg_recall_top20 = np.mean(metrics[model_name]['RecallTop20'])
        avg_recall_top30 = np.mean(metrics[model_name]['RecallTop30'])
        avg_recall_top40 = np.mean(metrics[model_name]['RecallTop40'])
        avg_recall_top50 = np.mean(metrics[model_name]['RecallTop50'])

        
        writer.writerow({
            'Model': model_name,
            'Accuracy': round(avg_acc, 4),
            'AUC': round(avg_auc, 4),
            'AUPR': round(avg_aupr, 4),
            'F1': round(avg_f1, 4),
            'Precision': round(avg_precision, 4),
            'Recall': round(avg_recall, 4),
            'Recall5': round(avg_recall5, 4),
            'Recall6': round(avg_recall6, 4),
            'Recall7': round(avg_recall7, 4),
            'Recall8': round(avg_recall8, 4),
            'Recall9': round(avg_recall9, 4),
            'PrecisionTop10': round(avg_precision_top10, 4),
            'PrecisionTop20': round(avg_precision_top20, 4),
            'PrecisionTop30': round(avg_precision_top30, 4),
            'PrecisionTop40': round(avg_precision_top40, 4),
            'PrecisionTop50': round(avg_precision_top50, 4),
            'RecallTop10': round(avg_recall_top10, 4),
            'RecallTop20': round(avg_recall_top20, 4),
            'RecallTop30': round(avg_recall_top30, 4),
            'RecallTop40': round(avg_recall_top40, 4),
            'RecallTop50': round(avg_recall_top50, 4),
        })
    

