import ast

import torch
from torch.utils.data import DataLoader, Dataset

import pandas as pd

import numpy as np

from sklearn.utils import shuffle

from sklearn.preprocessing import QuantileTransformer


class DatasetController:
    def __init__(
        self, file_path, lm, mask_prob, train_size, valid_size, test_size, train_dir, valid_dir, test_dir, rewrite=False
    ):
        self.rewrite = rewrite
        self.lm = lm
        self.mask_prob = mask_prob

        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir

        dataset = pd.read_csv(file_path, sep=";")[["customer_id", "history"]]

        self.train_dataset, self.valid_dataset, self.test_dataset = None, None, None

        if rewrite:
            dataset = shuffle(dataset)
            self.split_and_save_sets(dataset, train_size, valid_size, test_size, train_dir, valid_dir, test_dir)

        self.train_dataset, self.valid_dataset, self.test_dataset = self.read_sets()
        del dataset

    def get_sets(self, train=True, valid=True, test=True):
        datasets = []
        if train:
            datasets.append(self.train_dataset)
        if valid:
            datasets.append(self.valid_dataset)
        if test:
            datasets.append(self.test_dataset)

        return datasets

    def read_sets(self, train=True, valid=True, test=True):
        datasets = []
        if train:
            train_dataset = TimeSeriesBERTDataset(
                self.train_dir + "train.csv", lm=self.lm, mask_prob=self.mask_prob, static=False
            )
            self.quantile_transformer = train_dataset.quantile_transformer
            datasets.append(train_dataset)
        if valid:
            valid_dataset = TimeSeriesBERTDataset(
                self.valid_dir + "valid.csv",
                lm=self.lm,
                mask_prob=self.mask_prob,
                quantile_transformer=self.quantile_transformer,
                static=True,
            )
            datasets.append(valid_dataset)
        if test:
            test_dataset = TimeSeriesBERTDataset(
                self.test_dir + "test.csv",
                lm=self.lm,
                mask_prob=self.mask_prob,
                quantile_transformer=self.quantile_transformer,
                static=True,
            )
            datasets.append(test_dataset)

        return datasets

    @staticmethod
    def split_and_save_sets(dataset, train_size, valid_size, test_size, train_dir, valid_dir, test_dir):
        dataset_len = dataset.shape[0]

        train_end_index = int(dataset_len * train_size)
        valid_end_index = train_end_index + int(dataset_len * valid_size)
        test_end_index = valid_end_index + int(dataset_len * test_size)

        train_dataset = dataset[:train_end_index]
        valid_dataset = dataset[train_end_index:valid_end_index]
        test_dataset = dataset[valid_end_index:test_end_index]

        train_dataset.to_csv(train_dir + "train.csv", sep=";", index=False)
        valid_dataset.to_csv(valid_dir + "valid.csv", sep=";", index=False)
        test_dataset.to_csv(test_dir + "test.csv", sep=";", index=False)

    @staticmethod
    def calculate_mean_and_std(column):
        column = list(column.values)
        column = str(column)
        column = column.replace("'", "")
        column = ast.literal_eval(column)
        column = np.array(column)
        return column.mean(), column.std()


class TimeSeriesBERTDataset(Dataset):
    def __init__(self, time_series_dataset_path, lm, mask_prob, quantile_transformer=None, static=True):
        self.time_series_dataset_path = time_series_dataset_path

        self.lm = lm
        self.mask_prob = mask_prob

        self.quantile_transformer = quantile_transformer

        self.dataset = pd.read_csv(time_series_dataset_path, sep=";")

        self.static = static

        if self.quantile_transformer is None:  # Train: =None; Val/Test: !=None
            self.quantile_transformer = self.fit_quantile_transformer(self.dataset["history"])

        if self.static:
            self.make_instances()

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        row = self.dataset.iloc[index]

        target_series = None
        masked_series = None
        mask = None

        if not self.static:
            target_series = row["history"]

            if type(target_series) == str:
                target_series = np.array(ast.literal_eval(target_series))

            target_series = self.quantile_transformer.transform(target_series.reshape(1, -1)).reshape(-1)

            masked_series, mask = self.get_masked_series(series=target_series)

        else:
            target_series = row["target_series"]
            masked_series = row["masked_series"]
            mask = row["mask"]

        item = {
            "input_series": torch.Tensor(masked_series),
            "target_series": torch.Tensor(target_series),
            "mask": torch.BoolTensor(mask),
        }

        return item

    def make_instances(self, save=True):
        all_target_series = []
        all_masked_series = []
        all_masks = []
        for i, row in self.dataset.iterrows():
            target_series = np.array(ast.literal_eval(self.dataset.iloc[i]["history"]))
            target_series = self.quantile_transformer.transform(target_series.reshape(1, -1)).reshape(-1)
            masked_series, mask = self.get_masked_series(series=target_series)
            all_target_series.append(list(target_series))
            all_masked_series.append(masked_series)
            all_masks.append(list(np.array(mask)))
        self.dataset["target_series"] = all_target_series
        self.dataset["masked_series"] = all_masked_series
        self.dataset["mask"] = all_masks

        if save:
            self.dataset.to_csv(self.time_series_dataset_path, sep=";", index=False)

    @staticmethod
    def fit_quantile_transformer(column):
        quantile_transformer = QuantileTransformer()
        column = list(column.values)
        column = str(column)
        column = column.replace("'", "")
        column = ast.literal_eval(column)
        column = np.array(column)
        quantile_transformer = quantile_transformer.fit(column)

        return quantile_transformer

    def get_masked_series(self, series):
        output_series = series.copy()

        keep_mask = np.ones(len(output_series), dtype=bool)
        p_m = 1 / self.lm  # probability of each masking sequence stopping. parameter of geometric distribution.
        p_u = (
            p_m * self.mask_prob / (1 - self.mask_prob)
        )  # probability of each unmasked sequence stopping. parameter of geometric distribution.
        p = [p_m, p_u]

        # Start in state 0 with masking_ratio probability
        state = int(np.random.rand() > self.mask_prob)  # state 0 means masking, 1 means not masking

        for i in range(len(output_series)):
            keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
            if np.random.rand() < p[state]:
                state = 1 - state
            if (keep_mask == 0).sum() >= len(series) * self.mask_prob:
                break

        if (keep_mask == 0).sum() == 0:
            random_id = np.random.randint(keep_mask.shape[0])
            keep_mask[random_id] = 0
            output_series[random_id] = -10

            return list(output_series), list(~torch.BoolTensor(keep_mask))

        for i in range(keep_mask.shape[0]):
            if keep_mask[i] == 0:
                output_series[i] = -10

        return list(output_series), list(~torch.BoolTensor(keep_mask))


def get_dataloaders(batch_size, train_dataset=None, valid_dataset=None, test_dataset=None):
    dataloaders = []
    if train_dataset:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        dataloaders.append(train_dataloader)
    if valid_dataset:
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
        dataloaders.append(valid_dataloader)
    if test_dataset:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        dataloaders.append(test_dataloader)
    return dataloaders


def main():
    dataset = pd.read_csv("dataset/customers_histories.csv", sep=";")[["customer_id", "history"]]
    delete_indexes = []
    for i, row in dataset.iterrows():
        target_series = ast.literal_eval(dataset.iloc[i]["history"])
        if target_series.count(0) > 20:
            delete_indexes.append(i)
    dataset = dataset.drop(delete_indexes)
    dataset.to_csv("dataset/customers_histories.csv", sep=";", index=False)
