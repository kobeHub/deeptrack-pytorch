import pathlib
import config
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from absl import app, flags
import pytorch_model_summary as pms

from visual import plot_losses
from data_utils import SensorData
from model import DeepTrackingRNN
from criterion import WeightedBCELoss

FLAGS = flags.FLAGS
flags.DEFINE_string('data_file', './data/data.t7', 'Path to the data file')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(argv):
    del argv
    param = config.PARAM
    print(f'Parameters: {param}')
    batch_size = 2
    epochs = 50
    seq_len = 100
    dataset = SensorData(FLAGS.data_file, param)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(
        f'Length of dataset: {len(dataset)}, loader batches: {len(data_loader)}')

    model_dir = pathlib.Path("results")
    img_dir = model_dir / "imgs"
    img_dir.mkdir(exist_ok=True, parents=True)

    model = DeepTrackingRNN(width=50, height=50, num_units=seq_len)
    # pms.summary(model, torch.zeros(batch_size, seq_len, 2, 50, 50),
    #             batch_size=batch_size,
    #             show_input=True,
    #             show_hierarchical=True,
    #             print_summary=True)
    model.to(DEVICE)
    criterion = WeightedBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    traing_loss = []
    for epoch in range(epochs):
        for batch_idx, data in enumerate(data_loader):
            inputs, targets = data
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            traing_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(
                    f'Epoch: {epoch}/{epochs}, Batch: {batch_idx}, Loss: {loss.item()}')

    # Save the model
    np.save(model_dir / "loss.npy", traing_loss)
    model.save(model_dir / f"model-{epochs}.pth")
    plot_losses(traing_loss, img_dir / f"loss-{epochs}.png")
    print(f'Model saved at {model_dir / f"model-{epochs}.pth"}')


if __name__ == "__main__":
    app.run(main)
