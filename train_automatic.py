# Import Libraries
from datetime import datetime
import pickle
import random
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
# import pickle
from utils import get_graph_pca
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
config=dict(
      out_size = 6,
      num_layers=2,
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

#set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#define optimizer, criterion and lr schedule
criterion = torch.nn.CrossEntropyLoss()
#lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

file_confusion = './confusion_data'
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

EVENT_SUBSETS = {}
num_dati_per_classe = 0
# Number of events to keep for each event type.
# The total number of events in the dataset is the sum of the values in this dictionary.
# We can use these values to have a more balanced dataset.
DEFAULT_EVENT_SUBSETS = {
#   "singletop": 0,
#   "ttbar": 0,
#   "1000_0": 935,
  "1000_100": 935,
  "1000_150": 963,
  "1000_200": 933,
  "1000_250": 932,
  "1000_300": 959,
#   "1000_350": 994,
  "1000_400": 1003,
  "1000_450": 4901,
  "1000_500": 5050,
  "1000_50": 918,
  "1100_0": 4399,
  "1100_100": 4478,
  "1100_150": 4525,
  "1100_200": 4458,
  "1100_250": 4498,
  "1100_300": 4728,
  "1100_350": 4800,
  "1100_400": 4802,
  "1100_450": 4884,
  "1100_500": 5018,
  "1100_50": 4382,
  "1100_550": 5131,
  "150_0": 13022,
  "152_22": 12538,
  "162_12": 11297,
  "165_35": 9321,
  "175_0": 9543,
  "175_25": 8800,
  "177_47": 6892,
  "187_12": 7013,
  "187_37": 6301,
  "190_60": 5567,
  "200_0": 7054,
  "200_25": 6774,
  "200_50": 6164,
  "202_72": 5371,
  "212_37": 5726,
  "212_62": 4991,
  "225_0": 5509,
  "225_25": 5506,
  "225_50": 5311,
  "225_75": 4534,
  "237_62": 3918,
  "250_0": 3686,
  "250_100": 2735,
  "250_25": 3595,
  "250_50": 3512,
  "250_75": 3209,
  "275_0": 3620,
  "275_25": 3525,
  "275_50": 3465,
  "275_75": 3313,
  "300_0": 2394,
  "300_100": 2167,
  "300_150": 17739,
  "300_25": 2315,
  "300_50": 2402,
  "300_75": 2281,
  "325_0": 2339,
  "325_50": 2271,
  "350_0": 2339,
  "350_100": 2055,
  "350_150": 1853,
  "350_200": 1377,
  "350_25": 2233,
  "350_50": 2282,
#   "350_75": 2227,
  "375_0": 1780,
  "375_50": 1905,
  "400_0": 1995,
  "400_100": 1517,
  "400_150": 1478,
  "400_200": 1287,
  "400_250": 1301,
  "400_25": 1636,
  "400_50": 1356,
  "425_0": 1367,
  "450_0": 1141,
#   "450_100": 1172,
  "450_150": 1060,
  "450_200": 1003,
  "450_250": 1100,
  "450_300": 997,
  "450_50": 1150,
  "500_0": 1187,
  "500_100": 1220,
  "500_150": 1223,
  "500_200": 1136,
  "500_250": 974,
  "500_300": 854,
#   "500_350": 711,
  "500_50": 1216,
#   "535_400": 409,
#   "550_0": 915,
  "550_100": 982,
  "550_150": 905,
  "550_200": 887,
  "550_250": 812,
  "550_300": 3897,
  "550_50": 931,
  "585_450": 2067,
  "600_0": 978,
  "600_100": 915,
  "600_150": 968,
  "600_200": 965,
  "600_250": 960,
  "600_300": 857,
  "600_350": 801,
  "600_400": 664,
  "600_450": 2494,
  "600_50": 929,
  "635_500": 2044,
  "650_0": 964,
  "650_100": 1006,
  "650_150": 934,
  "650_200": 948,
  "650_250": 999,
  "650_300": 895,
  "650_450": 3483,
  "650_500": 2434,
  "650_50": 1041,
  "700_0": 993,
  "700_100": 989,
  "700_150": 960,
  "700_200": 1004,
  "700_250": 1012,
  "700_300": 1016,
  "700_350": 965,
  "700_400": 906,
  "700_450": 4043,
  "700_500": 3483,
  "700_50": 1027,
  "750_0": 1037,
  "750_100": 1002,
  "750_150": 1005,
  "750_200": 1008,
  "750_250": 946,
  "750_300": 1001,
  "750_450": 4403,
  "750_500": 4054,
  "750_50": 1039,
  "800_0": 5062,
  "800_100": 1007,
  "800_150": 1017,
  "800_200": 1036,
  "800_250": 1048,
  "800_300": 1031,
  "800_350": 974,
  "800_400": 956,
  "800_450": 4753,
  "800_500": 4515,
  "800_50": 1038,
  "900_0": 4930,
  "900_100": 884,
  "900_150": 1018,
  "900_200": 988,
  "900_250": 1007,
  "900_300": 1039,
  "900_350": 1005,
  "900_400": 979,
  "900_450": 4943,
  "900_500": 4951,
  "900_50": 988
}
DEFAULT_EVENT_SUBSETS_COPY = DEFAULT_EVENT_SUBSETS.copy()
EVENT_SELECTED = {'1000_0': 935, '535_400': 409, '450_100': 1172, '550_0': 915, '1000_350': 994, 'ttbar':6093298}#  , '350_75': 2227, '500_350': 711, '800_150': 1017, '700_50': 1027, '500_150': 1223, '650_150': 934}
EVENT_ACCURACY = {}
EVENT_LABELS = {  #'ttbar':0, "singletop": 0
}
def train_and_evaluate(epochs):
    global EVENT_SUBSETS, num_dati_per_classe, EVENT_SELECTED, config, EVENT_LABELS, DEFAULT_EVENT_SUBSETS
    EVENT_SUBSETS = {}
    EVENT_LABELS = {}
    for key_sel,value_sel in EVENT_SELECTED.items():
        EVENT_SUBSETS[key_sel] = value_sel
        EVENT_LABELS[key_sel] = 0 if len(list(EVENT_LABELS.values())) == 0 else max(list(EVENT_LABELS.values())) + 1
    # EVENT_LABELS["singletop"] = EVENT_LABELS['ttbar']
    print(f"Inizializzato EVENT_SUBSETS: {EVENT_SUBSETS}")
    print(f"Inizializzato EVENT_LABELS: {EVENT_LABELS}")
    # Lista delle directory da rimuovere
    data_dir = "E:\\Cristian\\Code\\NeuralNetworkTesi\\GraphExplainability\\data\\"
    # directories_to_remove = [data_dir+"processed"]#, data_dir+"\\raw\\signal", data_dir+"\\raw\\singletop", data_dir+"\\raw\\ttbar"]

    # for directory in directories_to_remove:
    #     if os.path.exists(directory):
    #         if os.path.isdir(directory):
    #             shutil.rmtree(directory)  # Usa rmtree se la directory può contenere file
    #             print(f"Directory '{directory}' rimossa.")
    #         else:
    #             print(f"'{directory}' non è una directory.")
    #     else:
    #         print(f"Directory '{directory}' non esiste.")

    # print(f"Rimosse directory")
    num_dati_per_classe = min(EVENT_SUBSETS.values())

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
    event_label = EVENT_LABELS
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
            
            train_loss_steps.append(loss.item())
            train_acc_steps.append(accuracy_score(data.y.cpu().numpy(), pred_train.cpu().numpy()))
        train_losses.append(epoch_loss / len(train_loader))
        train_acc = accuracy_score(targets_train, predictions_train)
        train_precision = precision_score(targets_train, predictions_train, average='macro', zero_division=0)
        train_recall = recall_score(targets_train, predictions_train, average='macro', zero_division=0)
        train_f1 = f1_score(targets_train, predictions_train, average='macro', zero_division=0)
        # train_auc = roc_auc_score(targets_train, predictions_train, average='macro')

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

        # correctly_classified_signal = 0
        # correctly_classified_background = 0
        # misclassified_signal_as_other_signal = 0
        # misclassified_signal_as_background = 0
        # misclassified_background_as_signal = 0
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

                test_loss_steps.append(loss.item())
                test_acc_steps.append(accuracy_score(data.y.cpu().numpy(), pred_test.cpu().numpy()))

        test_losses.append(total_loss / len(test_loader))
        test_acc = accuracy_score(targets_test, predictions_test)
        test_precision = precision_score(targets_test, predictions_test, average='macro', zero_division=0)
        test_recall = recall_score(targets_test, predictions_test, average='macro', zero_division=0)
        test_f1 = f1_score(targets_test, predictions_test, average='macro', zero_division=0)

        test_accuracies.append(test_acc)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_f1_scores.append(test_f1)

        print(f'Epoch: {epoch:03d} ')
        if epoch==100 or  epoch==50:
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
    # filepath = f'./checkpoint/checkpoint_epoch_{epoch:03d}_2l (2).pt'    
    # torch.save(model.state_dict(), filepath)
    #print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f}, Test Loss: {test_losses[-1]:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}')



# -*- coding: utf-8 -*-
"""DatasetStandard.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1OOcf42sKW85Q9tRWG2Kdx-NsWp-d9nyM
"""


def make_tuple(x):
    if isinstance(x, tuple) or isinstance(x, list):
        return x
    else:
        return (x,)

import pickle
import re
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, download_url
import pandas as pd
from torch_geometric.data import Data
from tqdm import tqdm
import os
import shutil
import tarfile
import glob
import numpy as np
from scipy.ndimage import rotate
from sklearn.decomposition import PCA
from copy import deepcopy

# Random state for shuffling the dataset.
RANDOM_STATE = 42


# Names of the directories in the raw directory.
RAW_DIR_NAMES = ['signal', 'singletop', 'ttbar']

# Match between directory and event type.

    

SIGNAL_FILE_NAME = []


# These are the columns we should keep from the raw pandas dataframe.
# The nan columns are just a hack as we need to have the same number of columns for each row.
USEFUL_COLS = [
            # jet 1
            'pTj1',
            'etaj1',
            'phij1',
            'j1_quantile',
            'nan',
            'nan',
            # jet 2
            'pTj2',
            'etaj2',
            'phij2',
            'j2_quantile',
            'nan',
            'nan',
            # jet 3
            'pTj3',
            'etaj3',
            'phij3',
            'j3_quantile',
            'nan',
            'nan',
            # b1
            'pTb1',
            'etab1',
            'phib1',
            'b1_quantile',
            'b1m',
            'nan',
            # b2
            'pTb2',
            'etab2',
            'phib2',
            'b2_quantile',
            'b2m',
            'nan',
            # lepton
            'pTl1',
            'etal1',
            'phil1',
            'nan',
            'nan',
            'nan',
            # energy
            'ETMiss',
            'nan',
            'ETMissPhi',
            'nan',
            'nan',
            'metsig_New',]


# A markdown table to display the structure of a single event.
EVENT_TABLE =   """
                Each event is a graph with 6/7 nodes. Each node is built from the raw file as follows:

                | Particle          | Feature 1 | Feature 2 | Feature 3   | Feature 4     | Feature 5 | Feature 6    |
                |-------------------|-----------|-----------|-------------|---------------|-----------|--------------|
                | jet1              |  'pTj1'   | 'etaj1'   |   'phij1'   | 'j1_quantile' |    nan    |     nan      |
                | jet2              |  'pTj2'   | 'etaj2'   |   'phij2'   | 'j2_quantile' |    nan    |     nan      |
                | jet3 (optional)   |  'pTj3'   | 'etaj3'   |   'phij3'   | 'j3_quantile' |    nan    |     nan      |
                | b1                |  'pTb1'   | 'etab1'   |   'phib1'   | 'b1_quantile' |   'b1m'   |     nan      |
                | b2                |  'pTb2'   | 'etab2'   |   'phib2'   | 'b2_quantile' |   'b2m'   |     nan      |
                | lepton            |  'pTl1'   | 'etal1'   |   'phil1'   |      nan      |    nan    |     nan      |
                | energy            | 'ETMiss'  |   nan     | 'ETMissPhi' |      nan      |    nan    | 'metsig_New' |
                """

signal_file_name_and_event_label = './signal_info.pkl'
class EventsDataset(InMemoryDataset):
    def __init__(
            self,
            root,
            url,
            event_subsets: dict = EVENT_SUBSETS,
            add_edge_index: bool = True,
            delete_raw_archive: bool = False,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            enable_pca=False,
            event_label= EVENT_LABELS):

        self.url = url
        self.delete_raw_archive = delete_raw_archive
        self.event_subsets = event_subsets
        self.event_label = event_label
        self.add_edge_index = add_edge_index
        self.subset_string = '100'
        self.enable_pca = enable_pca
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print(f"Enabled PCA: {self.enable_pca}")

    @property
    def raw_file_names(self):
        return RAW_DIR_NAMES

    @property
    def processed_file_names(self):
        return [f'events_{self.subset_string}.pt']

    @property
    def event_structure(self):
        return EVENT_TABLE

    def download(self):
        print(f'Downloading {self.url} to {self.raw_dir}...')
        print('This may take a while...')
        raw_archive = download_url(self.url, self.raw_dir, filename='events.tar', log=False)
        print(f"self.event_subsets: {self.event_subsets}")

        print('Extracting files...')
        with tarfile.open(raw_archive) as tar:
            members = tar.getmembers()
            for member in members:
                if  'Wh_hbb_fullMix.h5' not in member.name:
                    tar.extract(member, self.raw_dir)
                    # L'espressione regolare
                    pattern = r'[a-zA-Z](\d+)p.*_(\d+)p'
                    name_file_h5 = os.path.basename(os.path.normpath(member.name))
                    # Ricerca della corrispondenza
                    match = re.search(pattern, name_file_h5)
                    
                        
                    # Estrazione dei gruppi
                    if match:
                        primo_numero = match.group(1)
                        secondo_numero = match.group(2)
                        if (primo_numero+"_"+secondo_numero in self.event_subsets.keys()):
                            SIGNAL_FILE_NAME.append(name_file_h5)
                            if (primo_numero+'_'+secondo_numero not in EVENT_LABELS ):
                                EVENT_LABELS[primo_numero+'_'+secondo_numero] = 0 if len(list(EVENT_LABELS.values())) == 0 else max(list(EVENT_LABELS.values())) + 1
                else:
                    continue
        
        
            
        if self.delete_raw_archive:
            os.remove(raw_archive)
        # print(EVENT_LABELS)
        print('Moving files...')
        for dir in self.raw_file_names:
            # print(f'{self.raw_dir}/**/{dir}')
            dirpath = glob.glob(f'{self.raw_dir}/**/{dir}', recursive=True)[0]
            shutil.move(dirpath, self.raw_dir)
            print(f'Moved {dirpath} to {self.raw_dir}')

        print('Cleaning up...')
        for f in os.listdir(self.raw_dir):
            if f not in self.raw_file_names + ['events.tar']:
                try:
                    shutil.rmtree(os.path.join(self.raw_dir, f))
                except NotADirectoryError:
                    os.remove(os.path.join(self.raw_dir, f))

        """
        At this stage, we should have the following directory structure.
        Notice h5 file names can change.

        root
        ├── processed
        └── raw
            ├── signal
            │   └── Wh_hbb_fullMix.h5
            ├── singletop
            │   └── singletop.h5
            └── ttbar
                └── ttbar.h5
        """

    def process(self):
        with open(signal_file_name_and_event_label, 'rb') as file:
            data = pickle.load(file)
            # Accedi agli array caricati
            signal_file_name = data['signal_file_name']
            
            SIGNAL_FILE_NAME = signal_file_name
            # EVENT_LABELS = event_label
        
        print(f"self.event_subsets: {self.event_subsets}")
        h5_files = {}

        for d in self.raw_file_names:
            dir_path = os.path.join(self.raw_dir, d)
            
            if 'signal' in d:
                
                for name_signal_file in SIGNAL_FILE_NAME:
                    pattern = r'[a-zA-Z](\d+)p.*_(\d+)p'
                    match = re.search(pattern, name_signal_file)
                    # Estrazione dei gruppi
                    if match:
                        primo_numero = match.group(1)
                        secondo_numero = match.group(2)
                        if (primo_numero+"_"+secondo_numero in self.event_subsets.keys()):
                            signal_file_path = os.path.join(dir_path, name_signal_file)
                            if os.path.exists(signal_file_path):
                                if (primo_numero+'_'+secondo_numero not in h5_files):
                                    h5_files[primo_numero+'_'+secondo_numero] = [signal_file_path]
                                else:
                                    h5_files[primo_numero+'_'+secondo_numero].append(signal_file_path)
            else:
                if ('ttbar' in dir_path):
                    h5_files[d] = glob.glob(f'{dir_path}/*.h5', recursive=True)[0]
        data_list = []
        conta_dati_per_classe = 0
        before_event_type=''
        
        compute_event_type = True
        # Convertiamo il dizionario in un iteratore
 

        
        for current_event_type, current_h5_file in h5_files.items():
            if (before_event_type!= current_event_type):
                    conta_dati_per_classe = 0 
                    before_event_type = current_event_type
                    compute_event_type = True
         
            if (compute_event_type):
                if (isinstance(current_h5_file, list)):
                    for file in current_h5_file:
                        data_list_ris, conta_dati_per_classe_ris, self.event_subsets = self.process_h5_file(current_event_type, file,self.event_subsets,conta_dati_per_classe,num_dati_per_classe)
                        data_list+= data_list_ris
                        conta_dati_per_classe = conta_dati_per_classe_ris
                        if (conta_dati_per_classe==num_dati_per_classe):
                            compute_event_type = False

                else:
                    data_list_ris, conta_dati_per_classe_ris, self.event_subsets = self.process_h5_file(current_event_type, current_h5_file,self.event_subsets,conta_dati_per_classe,num_dati_per_classe)
                    data_list+= data_list_ris
                    conta_dati_per_classe = conta_dati_per_classe_ris
                    if (conta_dati_per_classe==num_dati_per_classe):
                            compute_event_type = False
                print(f'Classe: {current_event_type} con conta_dati_per_classe:  {conta_dati_per_classe}')
        # print(EVENT_LABELS)
        # print(self.event_subsets)
        if (self.enable_pca):
            data_list = self.compute_pca_coordinates(data_list)
            # print(data_list)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])
    
    def process_h5_file(self,event_type, h5_file, EVENT_SUBSETS, conta_dati_per_classe, num_dati_per_classe):
        with open(signal_file_name_and_event_label, 'rb') as file:
            data = pickle.load(file)
            # Accedi agli array caricati
            signal_file_name = data['signal_file_name']
            event_label = data['event_label']
            SIGNAL_FILE_NAME = signal_file_name

        EVENT_LABELS = self.event_label
        data_list = []
        label = EVENT_LABELS[event_type]
        graphs = pd.read_hdf(h5_file)
        graphs.drop(columns=list(set(graphs.columns) - set(USEFUL_COLS)), inplace=True)
        graphs['nan'] = torch.nan
        graphs = graphs[USEFUL_COLS].reset_index()
        print(EVENT_SUBSETS)
        EVENT_SUBSETS[event_type] += graphs.shape[0]
        # print('EVENT_SUBSETS: ' ,EVENT_SUBSETS[event_type])
        if (event_type not in 'ttbar' and event_type  not in  'singletop'):
            if (graphs.shape[0]>=num_dati_per_classe-conta_dati_per_classe):
                graphs = graphs.sample(n=num_dati_per_classe-conta_dati_per_classe, random_state=RANDOM_STATE)
                conta_dati_per_classe += num_dati_per_classe-conta_dati_per_classe
            else:
                graphs = graphs.sample(n=graphs.shape[0], random_state=RANDOM_STATE)
                conta_dati_per_classe += graphs.shape[0]
        else:
            graphs = graphs.sample(n=num_dati_per_classe, random_state=RANDOM_STATE) # da fixare
        if (event_type not in 'ttbar' and event_type  not in  'singletop'):
            
            graphs = graphs.sample(n=graphs.shape[0], random_state=RANDOM_STATE)
        else:
            graphs = graphs.sample(n=120, random_state=RANDOM_STATE)
        
        for row in tqdm(graphs.values, total=graphs.shape[0], desc=f'Processing events in {h5_file}'):
            event_id = int(row[0])
            graph_features = row[1:]
            if (event_type not in 'ttbar' and event_type  not in  'singletop'):
                values_array = []

                # Iteriamo su ogni elemento dell'array
                for element in row[1:]:
                    # Se l'elemento è un dizionario, aggiungiamo tutti i valori al nuovo array
                    if isinstance(element, dict):
                        values_array.extend(element.values())
                    # Se l'elemento è un NaN, aggiungiamo semplicemente NaN al nuovo array
                    elif np.isnan(element):
                        values_array.append(np.nan)

                # Convertiamo l'array di valori in un array numpy
                graph_features = np.array(values_array)
                
            x = torch.from_numpy(graph_features).reshape(7, -1)
            
            x = x[x[:,0]>0]

            edge_index = None
            if self.add_edge_index:
                directed_edge_index = torch.combinations(torch.arange(x.shape[0]), 2)
                edge_index = torch.cat([directed_edge_index, directed_edge_index.flip(1)], dim=0).T
            
            data_list.append(Data(
                x=x,
                event_id=f'{event_type}_{event_id}',
                y=label,
                edge_index=edge_index,
            ))
            
            data_list.append(Data(
                x=x*1.1,
                event_id=f'{event_type}_{event_id}',
                y=label,
                edge_index=edge_index,
            ))
            for i in range(0,5):
                data_list.append(Data(
                    x=x+random.uniform(0.1, 0.5),
                    event_id=f'{event_type}_{event_id}',
                    y=label,
                    edge_index=edge_index,
                ))
            
        return data_list,conta_dati_per_classe, EVENT_SUBSETS
    
    def compute_pca_coordinates(self,data_list):
        all_graph = []
        for data in data_list:
            data_copy = deepcopy(data)
            
            data_trans = self.transform(data_copy)
            value_append = np.array(data_trans.x.view(-1))
            if (data_trans.x.shape[0]== 6):
                # print(f"1 {len(value_append)}")
                value_append= np.insert(value_append,24,[torch.nan,torch.nan,torch.nan,torch.nan,torch.nan,torch.nan,torch.nan,torch.nan,torch.nan,torch.nan,torch.nan,torch.nan])
                # print(f"2 {len(value_append)}")
            all_graph.append(value_append) 
        # 1. Estrarre i dati per il PCA
        all_x = np.vstack([graph for graph in all_graph])
        # print(all_x.shape)
        # print(np.array(all_graph).shape)
        all_x_without_nan = np.nan_to_num(all_x, nan=0)
        # 2. Applicare il PCA
        n_components = 2  # Numero di componenti principali desiderate
        pca = PCA(n_components=n_components)
        all_x_pca = pca.fit_transform(all_x_without_nan)

        # 3. Aggiungere le coordinate PCA a ciascun elemento in data_list
        # start_idx = 0
        for index,data in enumerate(data_list):
            # num_samples = data.x.shape[0]
            data.pca = all_x_pca[index]
            # start_idx += num_samples
        return data_list

class CustomEventsDataset(EventsDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mean_values, self.std_values = self.calculate_mean_std() #calculate mean and std

    def custom_transform(self, data):
        # Standardize the features using mean and standard deviation
        standardized_x = (data.x - self.mean_values) / (self.std_values + 1e-6)
        return standardized_x

    def calculate_mean_std(self):
        # Initialize lists to store all features of all graphs
        all_features = []

        # Iterate through all graphs in the dataset
        for idx in range(len(self)):
            data = super().__getitem__(idx)
            all_features.append(data.x.numpy())

        # Concatenate features from all graphs
        all_features = np.concatenate(all_features, axis=0)

        # Calculate mean and standard deviation
        mean_values = np.mean(all_features, axis=0)
        std_values = np.std(all_features, axis=0)

        return mean_values, std_values

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        # Apply the custom transformation to obtain the normalized data
        normalized_data = self.custom_transform(data)

        # Add the 'data_norm' attribute to the data
        data.data_norm = normalized_data

        return data
# def get_confusion_matrix(targets,predictions):
#             # Calculate confusion matrix
#         conf_matrix = confusion_matrix(targets, predictions)

#         # Plot confusion matrix with correct labels
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['background', 'signal'], yticklabels=['background', 'signal'])
#         plt.xlabel('Predicted Label')
#         plt.ylabel('True Label')
#         plt.title('Confusion Matrix')
#         plt.show()

train_and_evaluate(epochs=200)