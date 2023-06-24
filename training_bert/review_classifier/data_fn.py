#data processing utilities
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


def optimization_read_split_data(path, test_size, label_col_name):
    df = pd.read_csv(path)
    #y = df[label_col_name].astype(str)
    #df = df.drop(columns=[label_col_name])

    #X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=test_size, stratify=y)
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, stratify=y_train)
    #return X_train.values.tolist(), X_val.values.tolist(), X_test.values.tolist(), y_train, y_val, y_test
    train, test = train_test_split(df, test_size=test_size, stratify=df[label_col_name])
    train, validate = train_test_split(train, test_size=test_size, stratify=train[label_col_name])
    labels = set(df[label_col_name])
    return train, validate, test, labels

def training_read_split_data(path, test_size, label_col_name):
    df = pd.read_csv(path)
    y = df[label_col_name].astype(str)
    df = df.drop(labels=label_col_name)

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=test_size, stratify=y)
    return X_train.values.tolist(), X_test.values.tolist(), y_train, y_test

def dataloader(X, y, batch_size):
    dataloader =CustomTextDataset(X, y)
    dataloader = DataLoader(dataloader, batch_size=batch_size, shuffle=True)
    return dataloader


class CustomTextDataset(Dataset):
    def __init__(self, txt, labels):
        self.labels = labels
        self.text = txt
    def __len__(self):
            return len(self.labels)
    def __getitem__(self, idx):
            label = self.labels[idx]
            text = self.text[idx]
            sample = {"Text": text, "Class": label}
            return sample

    

from sklearn.model_selection import train_test_split

def get_data_splits(X, y, train_size=0.7):
    """Generate balanced data splits."""
    X_train, X_, y_train, y_ = train_test_split(
        X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_, y_, train_size=0.5, stratify=y_)
    return X_train, X_val, X_test, y_train, y_val, y_test