from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from dataset import MNISTDataset, train_test_dataset_split
from net import load_model, SimpleModel



INPUT_SHAPE = 784
CLASSES_NUM = 10


class Client:
    def __init__(self, id: str, model: SimpleModel, global_weights_save_folder_path: str, local_weights_save_folder_path: str, train_set, test_set):
        super().__init__(id, model, global_weights_save_folder_path, local_weights_save_folder_path)
        self.train_set = train_set
        self.test_set = test_set

    def get_weights(self, get_weights_ins):
        weights = self.model.get_weights()
        return weights

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def train(self, train_ins):
        if train_ins.weights:
            weights = train_ins.weights
            self.model.set_weights(weights)
        config = train_ins.config
        batch_size = config['batch_size']
        epochs = config['epochs']

        lossf = nn.CrossEntropyLoss()
        dataloader = DataLoader(self.train_set, batch_size=batch_size)
        optimizer = SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        self.model.train()

        for epoch in range(epochs):
            train_loss = 0.0
            train_correct = 0.0

            for batch_id, data in enumerate(dataloader):
                images = data['image']
                labels = data['label'].reshape(-1)

                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                output = self.model(images)

                loss = lossf(output, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_correct += (output.max(dim=1).indices == labels).sum()

            train_loss /= len(dataloader)
            train_accuracy = (train_correct * 100) / len(self.train_set)

        weights = self.model.get_weights()
        return train_loss

    def test(self, eval_ins):
        weights = eval_ins.weights
        self.model.set_weights(weights)

        dataloader = DataLoader(self.test_set)
        lossf = nn.CrossEntropyLoss()
        self.model.eval()
        test_loss = 0.0
        test_correct = 0.0

        for batch_id, data in enumerate(dataloader):
            images = data['image']
            labels = data['label'].reshape(-1)

            images = images.to(self.device)
            labels = labels.to(self.device)

            output = self.model(images)

            loss = lossf(output, labels)

            test_loss += loss.item()
            test_correct += (output.max(dim=1).indices == labels).sum().item()

        test_loss /= len(dataloader)
        test_accuracy = test_correct * 100 / len(self.test_set)

        metrics = {'test_loss': test_loss, 'test_accuracy': test_accuracy}

        return metrics

    def get_prediction(self, weights):
        self.model.set_weights(weights)

        dataloader = DataLoader(self.test_set)
        lossf = nn.CrossEntropyLoss()
        self.model.eval()
        test_loss = 0.0
        test_correct = 0.0
        predictions = []

        for batch_id, data in enumerate(dataloader):
            images = data['image']
            labels = data['label'].reshape(-1)

            images = images.to(self.device)
            labels = labels.to(self.device)

            output = self.model(images)

            predictions.append(list(output))

        return predictions


def create_client(id, dataset_path, global_weights_save_folder_path: str, local_weights_save_folder_path: str) -> Client:
    dataset = MNISTDataset(dataset_path=dataset_path)
    train_set, test_set = train_test_dataset_split(dataset, test_size=0.1)

    model = load_model(input_shape=INPUT_SHAPE, classes=CLASSES_NUM)

    client = Client(id=id, model=model, global_weights_save_folder_path=global_weights_save_folder_path, local_weights_save_folder_path=local_weights_save_folder_path, train_set=train_set, test_set=test_set)

    return client