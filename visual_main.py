from absl import app, flags
from torch.utils.data import DataLoader
import torch

from visual import visual_multiple_samples, animate_images
from data_utils import SensorData


FLAGS = flags.FLAGS
flags.DEFINE_string('data_file', './data/data.t7', 'Path to the data file')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_string('option', 'img', 'Visualize the input data')
flags.DEFINE_integer('n_seq', 100, 'Length of sequence data')


def main(argv):
    del argv
    param = {'grid_minX': -25,  # occupancy grid bounds [m]
             'grid_maxX': 25,  # occupancy grid bounds [m]
             'grid_minY': -45,  # occupancy grid bounds [m]
             'grid_maxY': 5,  # occupancy grid bounds [m]
             'grid_step': 1,  # resolution of the occupancy grid [m]
             'sensor_start': -180,  # first depth measurement [degrees]
             'sensor_step': 0.5}  # resolution of depth measurements [degrees]
    print(f'Parameters: {param}')
    dataset = SensorData(file=FLAGS.data_file, params=param,
                         sequence_length=FLAGS.n_seq)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    print(f'Length of dataset: {len(dataset)}, loader: {len(data_loader)}')
    for batch_idx, data in enumerate(data_loader):
        inputs, targets = data
        print(f'Batch {batch_idx}: {inputs.shape}, targets: {targets.shape}')
        value_counts = torch.unique(inputs, return_counts=True)
        print(f'Value counts: {value_counts}')
        # Visualize the first image in the batch
        if FLAGS.option == 'img':
            for j in range(0, 100, 20):
                visual_multiple_samples(inputs[0, j:j+10, :, :, :])
        elif FLAGS.option == 'anim':
            animate_images(targets[0], 'results/imgs/sensor_data.mp4')
        break


if __name__ == "__main__":
    app.run(main)
