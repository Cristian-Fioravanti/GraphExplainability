# Import Libraries
import wandb
# from sparticles.datasetstandard import DEFAULT_EVENT_SUBSETS
# import torch_geometric as pygeo
# import os
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.nn.init as init
# import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
# from sparticles import EventsDataset
from sparticles.transforms import MakeHomogeneous
from sparticles import plot_event_2d
from torch_geometric.nn import global_mean_pool
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
# from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm # for nice bar
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
# from torch_geometric.transforms import BaseTransform
# import numpy as np
from GraphModel.GraphTransformerModel import GraphTransformerModel
from datasetClass import CustomEventsDataset
# import pickle

def train():
    
    wandb.init()

    config = wandb.config

    dataset = CustomEventsDataset(
        root='E:/Cristian/Code/NeuralNetworkTesi/GraphExplainability/data',
        url='https://cernbox.cern.ch/s/0nh0g7VubM4ndoh/download',
        delete_raw_archive=False,
        add_edge_index=True,
        transform=MakeHomogeneous()
    )

    # split the dataset
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
    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)

    # Define the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GraphTransformerModel(out_size=165,
                                  input_size=12,
                                  hidden_size=config.hidden_size,
                                  num_layers=config.num_layers,
                                  num_heads=config.num_heads,
                                  dropout=config.dropout,
                                  normalization=True).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    file_name = './training_data.pkl'

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    train_precisions = []
    train_recalls = []
    train_f1_scores = []
    test_precisions = []
    test_recalls = []
    test_f1_scores = []

    for epoch in range(1, config.epoch):
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
            data.y = data.y.to(device)
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

        train_losses.append(epoch_loss / len(train_loader))
        train_acc = accuracy_score(targets_train, predictions_train)
        train_precision = precision_score(targets_train, predictions_train, average='macro', zero_division=0)
        train_recall = recall_score(targets_train, predictions_train, average='macro', zero_division=0)
        train_f1 = f1_score(targets_train, predictions_train, average='macro', zero_division=0)

        train_accuracies.append(train_acc)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1_scores.append(train_f1)

        # Testing loop
        model.eval()
        total_loss = 0.0
        correct_test = 0
        total_test = 0
        predictions_test = []
        targets_test = []

        with torch.no_grad():
            for data in tqdm(test_loader, leave=False):
                out = model(data)
                data.y = data.y.to(device)
                loss = criterion(out, data.y)
                total_loss += loss.item()

                pred_test = out.argmax(dim=1)
                correct_test += int((pred_test == data.y).sum())
                total_test += len(data.y)
                predictions_test.extend(pred_test.tolist())
                targets_test.extend(data.y.tolist())

        test_losses.append(total_loss / len(test_loader))
        test_acc = accuracy_score(targets_test, predictions_test)
        test_precision = precision_score(targets_test, predictions_test, average='macro', zero_division=0)
        test_recall = recall_score(targets_test, predictions_test, average='macro', zero_division=0)
        test_f1 = f1_score(targets_test, predictions_test, average='macro', zero_division=0)

        test_accuracies.append(test_acc)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_f1_scores.append(test_f1)
        
    
    
    # Log metrics to wandb
    wandb.log({
        "epoch": epoch,
        "train_loss": train_losses[-1],
        "train_accuracy": train_accuracies[-1],
        "train_precision": train_precisions[-1],
        "train_recall": train_recalls[-1],
        "train_f1_score": train_f1_scores[-1],
        "test_loss": test_losses[-1],
        "test_accuracy": test_accuracies[-1],
        "test_precision": test_precisions[-1],
        "test_recall": test_recalls[-1],
        "test_f1_score": test_f1_scores[-1]
    })

    # Save the model if it has the best test accuracy
    wandb.watch(model, log="all")
    # if test_accuracies[-1] == max(test_accuracies):
    #     torch.save(model.state_dict(), f'best_model_epoch_{epoch}.pt')
    #     wandb.save(f'best_model_epoch_{epoch}.pt')

if __name__ == "__main__":
    # Define the configuration parameters for the sweep
    sweep_config = {
        "project": "GraphExplainability",
        "entity": "signorfiss",
        "program": "train.py",
        "method": "bayes",
        "metric": {"name": "test_accuracy", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"min": 0.0001, "max": 0.01},
            "batch_size": {"values": [64, 128, 256, 512]},
            "num_layers": {"values": [6, 8, 10, 12]},
            "hidden_size": {"values": [32, 64, 128, 256]},
            "num_heads": {"values": [2, 4, 8]},
            "dropout": {"min": 0.0, "max": 0.5},
            "weight_decay": {"min": 0.00001, "max": 0.001},
            "epoch" : {"values": [5, 10]}
        }
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config)
    # Start the WandB agent with the sweep ID
    wandb.agent(sweep_id, function=train)
