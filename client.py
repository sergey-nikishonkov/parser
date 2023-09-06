import torch
from torch import nn, optim

from tsb_data import DatasetController, get_dataloaders
from tsb_model import TimeSeriesBERTModel, TimeSeriesBERTModelForTraining

dataset_parameters = {
    "file_path": "dataset/customers_histories.csv",
    "lm": 3,
    "mask_prob": 0.15,
    "train_size": 0.8,
    "valid_size": 0.1,
    "test_size": 0.1,
    "train_dir": "dataset/train/",
    "valid_dir": "dataset/valid/",
    "test_dir": "dataset/test/",
    "rewrite": True,
}

model_parameters = {
        "time_series_size": 71,
        "hidden_size": 128,
        "encoder_layers_count": 4,
        "heads_count": 6,
        "dropout_prob": 0.1,
    }

trainer_parameters = {
        "batch_size": 16,
        "learning_rate": 0.0001,
        "epoch_num": 20,
        "k": 10,
    }

class Client:
    def __init__(self, id: str, global_weights_save_folder_path: str = None, local_weights_save_folder_path: str = None):
        tsb_model = TimeSeriesBERTModel(**model_parameters)
        model = TimeSeriesBERTModelForTraining(tsb_model)
        super().__init__(id, model)

        dataset_controller = DatasetController(**dataset_parameters)

        train_dataset, valid_dataset, test_dataset = dataset_controller.get_sets()
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = get_dataloaders(
            batch_size=trainer_parameters["batch_size"],
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
        )

        self.model.init_weights(lambda _x: nn.init.xavier_uniform_(_x))

    def get_weights(self, get_weights_ins: GetWeightsInstructions):
        return self.model.get_weights()

    def set_weights(self, set_weights_ins: SetWeightsInstructions):
        self.model.set_weights(set_weights_ins.weights)

    def train(self, train_ins: TrainInstructions):
        epochs_num = trainer_parameters['epochs_num']

        optimizer = optim.AdamW(self.model.parameters(), lr=trainer_parameters["learning_rate"])
        lossf = TimeSeriesBERTLoss(k=trainer_parameters["k"])

        for epoch in range(epochs_num):
            train_loss = 0.0

            self.model.train()

            for batch_id, data in enumerate(self.train_dataloader):
                data = {key: value.to(self.device) for key, value in data.items()}
                batch_size = data["input_series"].shape[0]
                time_series_size = data["input_series"].shape[1]

                optimizer.zero_grad()

                pred_series = self.model(data["input_series"])

                loss, masked_pred, masked_true = lossf(
                    pred_series,
                    data["target_series"].reshape(batch_size, time_series_size, 1),
                    data["mask"].reshape(batch_size, time_series_size, 1),
                    epoch + 1,
                )

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(self.train_dataloader)

            return train_loss

    def test(self, eval_ins: EvaluateInstructions) -> EvaluateResult:
        test_loss = 0.0

        self.model.eval()
        lossf = TimeSeriesBERTLoss()

        with torch.no_grad():
            for batch_id, data in enumerate(self.test_dataloader):
                data = {key: value.to(self.device) for key, value in data.items()}
                batch_size = data["input_series"].shape[0]
                time_series_size = data["input_series"].shape[1]

                pred_series = self.model(data["input_series"])

                loss, masked_pred, masked_true = lossf(
                    pred_series,
                    data["target_series"].reshape(batch_size, time_series_size, 1),
                    data["mask"].reshape(batch_size, time_series_size, 1),
                )

                test_loss += loss.item()

            test_loss /= len(self.test_dataloader)

            return test_loss

    def get_prediction(self):
        test_loss = 0.0

        self.model.eval()
        lossf = TimeSeriesBERTLoss()
        predictions = []

        with torch.no_grad():
            for batch_id, data in enumerate(self.test_dataloader):
                data = {key: value.to(self.device) for key, value in data.items()}
                batch_size = data["input_series"].shape[0]
                time_series_size = data["input_series"].shape[1]

                pred_series = self.model(data["input_series"])
                predictions += list(pred_series)

            return predictions

class TimeSeriesBERTLoss(nn.Module):
    """
    TimeSeriesBERTLoss
    ---------------

    Loss = (k / epoch_num) * loss_reconstruct + masked_ts_loss
    loss_reconstruct = MSE(predicted series, true series)
    masked_ts_loss = RMSE(predicted masked values, true masked values)

    In Train:
    returns Loss
    In Valid/Test:
    returns masked_ts_loss
    """

    def __init__(self, k=None):
        super().__init__()

        self.k = k
        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor, epoch=None):
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        masked_ts_loss = torch.sqrt(self.mse_loss(masked_pred, masked_true))

        if epoch is not None:
            loss_reconstruct = self.mse_loss(y_pred, y_true)

            loss = masked_ts_loss + (self.k / epoch) * loss_reconstruct  # k = 10

            return loss, masked_pred, masked_true

        return masked_ts_loss, masked_pred, masked_true
