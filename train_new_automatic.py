from datetime import datetime
import os
import pickle
import random
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from sparticles.transforms import MakeHomogeneous
from sparticles import plot_event_2d
from torch_geometric.nn import global_mean_pool
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm # for nice bar
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils import get_graph_pca
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

from new_datasetClass import CustomEventsDataset
from GraphModel.GraphTransformerModel import GraphTransformerModel

config=dict(
      out_size = 2,
      num_layers=3,
      hidden_size=60,
      input_size=12,
      num_heads= 30,
      learning_rate = 0.0005,
      weight_decay=0.0005,
      batch_size = 512,
      signal=1000,
      singletop=100,
      ttbar=100,
      dropout = 0.3,
      normalization = True
)
print(config)

file_confusion = './confusion_data'
EVENT_SELECTED = {'1000': 918, 'ttbar':6093298}#, '600_300': 857, '165_35': 9321, '900_300': 1039, }# { '800_450': 4753, '550_150': 905, '325_0': 2339}
EVENT_ACCURACY = {}
EVENT_LABELS = {} #'ttbar':0, "singletop": 0
def train_and_evaluate(epochs):
    #set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    num_dati_per_classe=0
    EVENT_SUBSETS = {}
    EVENT_LABELS = {}
    for key_sel,value_sel in EVENT_SELECTED.items():
        EVENT_SUBSETS[key_sel] = value_sel
        EVENT_LABELS[key_sel] = 0 if len(list(EVENT_LABELS.values())) == 0 else max(list(EVENT_LABELS.values())) + 1

    print(f"Inizializzato EVENT_SUBSETS: {EVENT_SUBSETS}")
    print(f"Inizializzato EVENT_LABELS: {EVENT_LABELS}")
    # Lista delle directory da rimuovere
    data_dir = "E:\\Cristian\\Code\\NeuralNetworkTesi\\GraphExplainability\\data\\"
    directories_to_remove = [data_dir+"processed"]#, data_dir+"\\raw\\signal", data_dir+"\\raw\\singletop", data_dir+"\\raw\\ttbar"]
    print(EVENT_LABELS)
    for directory in directories_to_remove:
        if os.path.exists(directory):
            if os.path.isdir(directory):
                shutil.rmtree(directory)  # Usa rmtree se la directory può contenere file
                print(f"Directory '{directory}' rimossa.")
            else:
                print(f"'{directory}' non è una directory.")
        else:
            print(f"Directory '{directory}' non esiste.")

    print(f"Rimosse directory")
    criterion = torch.nn.CrossEntropyLoss()
    #define the model
    model = GraphTransformerModel(out_size= config['out_size'],
                        input_size=config['input_size'],
                        hidden_size = config['hidden_size'],
                        num_layers = config['num_layers'],
                        num_heads = config['num_heads'],
                        dropout = config['dropout'],
                        normalization = config['normalization']).to(device)
    
    dataset = CustomEventsDataset(
    root='E:/Cristian/Code/NeuralNetworkTesi/GraphExplainability/data',
    url='https://cernbox.cern.ch/s/0nh0g7VubM4ndoh/download',
    delete_raw_archive=False,
    add_edge_index=True,
    transform=MakeHomogeneous(),
    enable_pca=False,
    event_subsets = EVENT_SUBSETS,
    event_label = EVENT_LABELS,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],weight_decay=config['weight_decay'])

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
    train_loader = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=config['batch_size'], shuffle=False)

    for epoch in range(1, epochs):
        model.train()
        epoch_loss = 0.0

        # Training loop
        correct_train = 0
        total_train = 0
        predictions_train = []
        targets_train = []

        for data in tqdm(train_loader, leave=False):
            data = data.to(device)
            out = model(data,int(data.x.shape[0] / 7),data.x.shape)
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
            
            train_loss_steps.append(loss.item())
            train_acc_steps.append(accuracy_score(data.y.cpu().numpy(), pred_train.cpu().numpy()))
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
                out = model(data,int(data.x.shape[0] / 7),data.x.shape)
                data.y = data.y.to(device)
                loss = criterion(out, data.y)
                total_loss += loss.item()

                pred_test = out.argmax(dim=1)
                correct_test += int((pred_test == data.y).sum())
                total_test += len(data.y)
                predictions_test.extend(pred_test.tolist())
                targets_test.extend(data.y.tolist())

                test_loss_steps.append(loss.item())
                test_acc_steps.append(accuracy_score(data.y.cpu().numpy(), pred_test.cpu().numpy()))

       
       

        print(f'Epoch: {epoch:03d} ')
        if epoch==100 or  epoch==50 or  epoch==150 or  epoch==300 or  epoch==350:
            filepath = f'./checkpoint/checkpoint_epoch_{epoch:03d}_final_test.pt'    
            torch.save(model.state_dict(), filepath)
            print(test_acc_steps[-1])
            file_name = f'./training_data_{config['out_size']}_{round(test_acc_steps[-1], 2)}_{datetime.now().strftime("%d-%m-%y")}.pkl'
            with open(file_name, 'wb') as file:
                pickle.dump({
                    'train_loss_steps': train_loss_steps,
                    'train_acc_steps': train_acc_steps,
                    'test_loss_steps': test_loss_steps,
                    'test_acc_steps': test_acc_steps
                }, file)    
    
    filepath = f'./checkpoint/checkpoint_epoch_{epoch:03d}_final_test.pt'    
    torch.save(model.state_dict(), filepath)
    print(test_acc_steps[-1])
    file_name = f'./training_data_{config['out_size']}_{round(test_acc_steps[-1], 2)}_{datetime.now().strftime("%d-%m-%y")}.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump({
            'train_loss_steps': train_loss_steps,
            'train_acc_steps': train_acc_steps,
            'test_loss_steps': test_loss_steps,
            'test_acc_steps': test_acc_steps
        }, file)
train_and_evaluate(20)