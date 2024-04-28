import torch
from torch.utils.data import DataLoader
from absl import app, flags

from model import DeepTrackingRNN
from data_utils import SensorData
from visual import visual_multiple_samples, animate_images
import config


FLAGS = flags.FLAGS
flags.DEFINE_string('model_path', './results/epoch-5/model-5.pth',
                    'Path to the model file')
flags.DEFINE_string('data_path', './data/data.t7', 'Path to the data file')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_string('option', 'img', 'Visualize the predicted data')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def infer(argv):
    del argv
    model_path = FLAGS.model_path
    data_path = FLAGS.data_path

    # Load the model
    model = DeepTrackingRNN.from_pretrained(model_path, map_location=DEVICE)
    model.to(DEVICE)
    model.eval()
    print(f'Model loaded from {model_path}')

    # Load the data
    param = config.PARAM
    batch_size = FLAGS.batch_size
    dataset = SensorData(data_path, param)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(
        f'Length of dataset: {len(dataset)}, loader batches: {len(data_loader)}')

    # Perform inference
    for batch_idx, data in enumerate(data_loader):
        inputs, _ = data
        inputs = inputs.to(DEVICE)

        with torch.no_grad():
            outputs = model(inputs)
        print(f'Batch {batch_idx}: {outputs.shape}')

        if FLAGS.option == 'img':
            # Visualize the predicted data
            for j in range(0, 100, 20):
                visual_multiple_samples(
                    outputs[0, j:j+10, :, :, :], title="Seeing beyond at", dim=0, img_path='results/imgs/predicted_data_e5.png')
        elif FLAGS.option == 'anim':
            animate_images(
                imgs=outputs[0], anim_path='results/imgs/predicted_data_e5.mp4', title="Seeing beyond at", dim=0)
        break


if __name__ == '__main__':
    app.run(infer)
