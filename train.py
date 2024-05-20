#Import Libraries
import torch_geometric as pygeo
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from sparticles import EventsDataset
from sparticles.transforms import MakeHomogeneous
from sparticles import plot_event_2d
from torch_geometric.nn import global_mean_pool
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm # for nice bar
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch_geometric.transforms import BaseTransform
import numpy as np

from GraphModel.GraphTransformerModel import GraphTransformerModel
from data.datasetClass import CustomEventsDataset

config=dict(
      out_size = 2,
      num_layers=1,
      hidden_size=60,
      input_size=12,
      num_heads= 4,
      learning_rate = 0.0005,
      weight_decay=0.0005,
      batch_size = 1,
      signal=400000,
      singletop=200000,
      ttbar=200000,
      dropout = 0.3,
      normalization = True
)
print(config)

dataset = CustomEventsDataset(
    root='./data',
    url='https://cernbox.cern.ch/s/0nh0g7VubM4ndoh/download',
    delete_raw_archive=False,
    add_edge_index=True,
    event_subsets={'signal': config['signal'], 'singletop': config['singletop'], 'ttbar': config['ttbar']},
    transform=MakeHomogeneous()
)

#split the dataset

# generate indices: instead of the actual data we pass in integers
train_indices, test_indices = train_test_split(
    range(len(dataset)),
    train_size=0.8,
    stratify=[g.y.item() for g in dataset], # to have balanced subsets
    random_state=42
)

dataset_train = Subset(dataset, train_indices)
dataset_test = Subset(dataset, test_indices)

print(f'Train set contains {len(dataset_train)} graphs, Test set contains {len(dataset_test)} graphs')

# Dataloaders
train_loader = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=config['batch_size'], shuffle=False)

#Define the model

#set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#define the model
model = GraphTransformerModel(out_size= config['out_size'],
                              input_size=config['input_size'],
                              hidden_size = config['hidden_size'],
                              num_layers = config['num_layers'],
                              num_heads = config['num_heads'],
                              dropout = config['dropout'],
                              normalization = config['normalization']).to(device)

#define optimizer, criterion and lr schedule
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],weight_decay=config['weight_decay'])
criterion = torch.nn.CrossEntropyLoss()
#lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.5)


#training and test the model
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
train_precisions = []
train_recalls = []
train_f1_scores = []
train_auc_scores = []
test_precisions = []
test_recalls = []
test_f1_scores = []
test_auc_scores = []

train_loss_steps = []
train_acc_steps = []
test_loss_steps = []
test_acc_steps = []

def train_and_evaluate(epochs):
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        # Training loop
        correct_train = 0
        total_train = 0
        predictions_train = []
        targets_train = []

        for data in tqdm(train_loader, leave=False):
            data = data.to(device)
            out = model(data)
            print(out)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

            pred_train = out.argmax(dim=1)
            correct_train += int((pred_train == data.y).sum())
            total_train += len(data.y)
            predictions_train.extend(pred_train.tolist())
            targets_train.extend(data.y.tolist())

            # Record loss and accuracy at each step
            train_loss_steps.append(loss.item())
            train_acc_steps.append(accuracy_score(data.y.cpu().numpy(), pred_train.cpu().numpy()))

        train_losses.append(epoch_loss / len(train_loader))
        train_acc = accuracy_score(targets_train, predictions_train)
        train_precision = precision_score(targets_train, predictions_train)
        train_recall = recall_score(targets_train, predictions_train)
        train_f1 = f1_score(targets_train, predictions_train)
        train_auc = roc_auc_score(targets_train, predictions_train)

        train_accuracies.append(train_acc)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1_scores.append(train_f1)
        train_auc_scores.append(train_auc)

        #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_losses[-1]:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}, Train AUC: {train_auc:.4f}')

        # Testing loop
        model.eval()
        total_loss = 0.0
        correct_test = 0
        total_test = 0
        predictions_test = []
        targets_test = []

        with torch.no_grad():
            for data in tqdm(test_loader, leave=False):
                out = model(data.to(device))
                loss = criterion(out, data.y)
                total_loss += loss.item()

                pred_test = out.argmax(dim=1)
                correct_test += int((pred_test == data.y).sum())
                total_test += len(data.y)
                predictions_test.extend(pred_test.tolist())
                targets_test.extend(data.y.tolist())

                # Record loss and accuracy at each step
                test_loss_steps.append(loss.item())
                test_acc_steps.append(accuracy_score(data.y.cpu().numpy(), pred_test.cpu().numpy()))

        test_losses.append(total_loss / len(test_loader))
        test_acc = accuracy_score(targets_test, predictions_test)
        test_precision = precision_score(targets_test, predictions_test)
        test_recall = recall_score(targets_test, predictions_test)
        test_f1 = f1_score(targets_test, predictions_test)
        test_auc = roc_auc_score(targets_test, predictions_test)

        test_accuracies.append(test_acc)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_f1_scores.append(test_f1)
        test_auc_scores.append(test_auc)


    print(f'Epoch: {epoch:03d}')
    filepath = f'./checkpoint/checkpoint_epoch_{epoch:03d}_2l (2).pt'    
    torch.save(model.state_dict(), filepath)
    #print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f}, Test Loss: {test_losses[-1]:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}')


train_and_evaluate(epochs=10)