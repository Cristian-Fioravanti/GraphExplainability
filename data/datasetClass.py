#This Custom dataset compute normalization of the node features
#and add the normalized features in 'data.data_norm'.
import numpy as np
from sparticles.dataset import EventsDataset

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