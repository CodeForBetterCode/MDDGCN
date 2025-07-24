import torch
import argparse
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split,KFold
from torch_geometric.data import Data
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingLR,StepLR
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score

random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Model Training')

parser.add_argument('--modeltype', type=str, default='MDDGCN', help='model of prdict',choices=['GCN', 'GAT', 'ChebNet', 'GraphSAGE','MDDGCN','CGMega','EMOGI','MDDGCN'])
parser.add_argument('--epochs', type=int, default=2500, help='Number of epochs to train the model')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for the optimizer')
parser.add_argument('--hidden_channels',type=int, default=128, help='Number of hidden channels in the model')
parser.add_argument('--intermediate_channels', type=int, default=128, help='Number of intermediate_channels in the model')
parser.add_argument('--attention_hidden', type=int, default=128, help='Number of attention_hidden in the model')
parser.add_argument('--ChetK',type=int, default=2, help='ChebNet of K')
parser.add_argument('--GATHeads',type=int, default=2, help='GAT of heads')
parser.add_argument('--scheduler', type=str, default='StepLR', help='model of scheduler',choices=['StepLR', 'CosineAnnealingLR'])
parser.add_argument('--feature', type=str, default='Enhanced', choices=['Biological', 'Expression','Enhanced'])
parser.add_argument('--perturb_features_p', type=float, default=0, choices=[0.01,0.02,0.03, 0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20])
parser.add_argument('--dropout_edge_p', type=float, default=0, choices=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

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
    return accuracy, auc, ap, f1

data = torch.load('./data/MDDGCN_PPI_Features_Labels.pt').to('cpu')

# by change the feature,we can test the performance of different feature
if 'Biological' in args.feature: 
    data.x = data.x[:, :-192]
elif 'Expression' in args.feature:
    data.x = data.x[:, -192:]
  
print(data)
ng = np.where(data.y== 1)[0]
print(f'positive sample size：{len(ng)}')
ngg = ng = np.where(data.y== 0)[0]
print(f'negative sample size：{len(ngg)}')

data = data.to(device)

# require the label,0 is negative,1 is positive,-1 is unlabeled
train_mask = (data.y == 0) | (data.y == 1) 
test_mask = (data.y == -1) 

from model import MDDGCN

# model initialization
def initialize_model(args):
    feature_dims =torch.tensor( [1273, 185, 377, 458, 192] , device=device) # 
    if 'MDDGCN' == args.modeltype:
        model = MDDGCN(in_channels=data.x.shape[1],hidden_channels=args.hidden_channels, intermediate_channels=args.intermediate_channels, out_channels=1, feature_dims=feature_dims,attention_hidden=args.attention_hidden,perturb_features_p=args.perturb_features_p,dropout_edge_p=args.dropout_edge_p).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

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

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

ACC = np.zeros(shape=(10, 5))
AUC = np.zeros(shape=(10, 5))
AUPR = np.zeros(shape=(10, 5))
F1 = np.zeros(shape=(10, 5))

best_experiment = -1
best_auc = -1 
best_metrics = {}

worst_experiment = -1
worst_mean_auc = float('inf')
worst_metrics = {}

# performance evaluation in 10 times of 5 fold cross validation
for experiment in range(10): 
    print(f'Experiment {experiment + 1}')
    
    kf = StratifiedKFold(n_splits=5, random_state=random_seed + experiment, shuffle=True)
    
    train_nodes = torch.nonzero(train_mask).squeeze()
    train_labels_all = data.y[train_nodes]


    for fold, (train_idx, val_idx) in enumerate(kf.split(train_nodes.cpu().numpy(), train_labels_all.cpu().numpy())):
        print(f'Fold {fold + 1}')
        
        train_nodes_fold = train_nodes[train_idx]
        val_nodes_fold = train_nodes[val_idx]
        
        train_labels = data.y[train_nodes_fold]
        val_labels = data.y[val_nodes_fold]
        
        model, optimizer, loss_fn, scheduler = initialize_model(args)

        fold_metrics = {'train_Loss':[], 'val_Loss':[], 'accuracy':[], 'auc':[], 'ap':[], 'f1':[], 'positive_ratio':[]}
        
        for epoch in range(args.epochs): 
            loss = train(model, optimizer, loss_fn, train_nodes_fold, train_labels)
            # update learning rate after each epoch
            # scheduler.step()

            # can print learning rate,to observe learning rate change
            # current_lr = scheduler.get_last_lr()[0]
            # print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss:.4f}, Learning Rate: {current_lr:.6e}")
        
        pred_labels, val_loss, pred_probs = evaluate(model, loss_fn, val_nodes_fold, val_labels)
        accuracy, auc, ap, f1 = compute_metrics(val_labels, pred_labels, pred_probs)
            
        print(f'Epoch {epoch}, train_Loss: {loss:.4f}, val_Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}, F1: {f1:.4f}')
        ACC[experiment][fold] = accuracy
        AUC[experiment][fold] = auc
        AUPR[experiment][fold] = ap
        F1[experiment][fold] = f1

    mean_acc = ACC[experiment].mean()
    mean_auc = AUC[experiment].mean()
    mean_aupr = AUPR[experiment].mean()
    mean_f1 = F1[experiment].mean()
    
    print(f'→ Mean Results of Experiment {experiment + 1}: ACC: {mean_acc:.4f}, AUC: {mean_auc:.4f}, AUPR: {mean_aupr:.4f}, F1: {mean_f1:.4f}')

    # write the best one
    if mean_auc > best_auc:
        best_auc = mean_auc
        best_experiment = experiment + 1
        best_metrics = {
            'ACC': mean_acc,
            'AUC': mean_auc,
            'AUPR': mean_aupr,
            'F1': mean_f1
        }
    # write the worst one
    if mean_auc < worst_mean_auc:
        worst_mean_auc = mean_auc
        worst_experiment = experiment + 1
        worst_metrics = {
            'ACC': mean_acc,
            'AUC': mean_auc,
            'AUPR': mean_aupr,
            'F1': mean_f1
        }

mean_ACC = ACC.mean()
mean_AUC = AUC.mean()
mean_AUPR = AUPR.mean()
mean_F1 = F1.mean()

from datetime import datetime
import os

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

file_name = f'result_{current_time}_{args.modeltype}.csv'

result_folder = 'paper_result_part1+2_test'
full_path = os.path.join(result_folder, file_name)

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

with open(full_path, 'w') as file:
    file.write(f'model_type: {args.modeltype}\n')
    file.write(f'epochs: {args.epochs}\n')
    file.write(f'learning_rate: {args.learning_rate}\n')
    file.write(f'hidden_channels: {args.hidden_channels}\n')
    file.write(f'intermediate_channels: {args.intermediate_channels}\n')
    file.write(f'GATHeads: {args.GATHeads}\n')
    file.write(f'ChetK: {args.ChetK}\n')
    file.write(f'feature: {args.feature}\n')
    file.write(f'perturb_features_p: {args.perturb_features_p}\n')
    file.write(f'dropout_edge_p: {args.dropout_edge_p}\n')
    file.write(f'scheduler: {args.scheduler}\n')
    file.write(f'attention_hidden: {args.attention_hidden}\n')
    file.write(f'Mean Accuracy: {mean_ACC}\n')
    file.write(f'Mean AUC: {mean_AUC}\n')
    file.write(f'Mean AUPR: {mean_AUPR}\n')
    file.write(f'Mean F1 Score: {mean_F1}\n')
    file.write(f'Best Mean Metrics: ACC: {best_metrics["ACC"]:.4f}, AUC: {best_metrics["AUC"]:.4f}, AUPR: {best_metrics["AUPR"]:.4f}, F1: {best_metrics["F1"]:.4f}\n')
    file.write(f'Worst Mean Metrics: ACC: {worst_metrics["ACC"]:.4f}, AUC: {worst_metrics["AUC"]:.4f}, AUPR: {worst_metrics["AUPR"]:.4f}, F1: {worst_metrics["F1"]:.4f}\n')
    file.write('\n') 

print("Data saved to  CSV file successfully.")

