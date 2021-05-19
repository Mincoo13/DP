import os
import sys
import json
import datetime
import skimage.draw
import numpy as np
import config as cfg
from mrcnn.config import Config
from mrcnn import model as modellib, utils

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

args = cfg.SEGMENT_TRAIN_ARGS


if args['current_folder']:
    os.chdir(f'{os.path.realpath(__file__)}\\..\\')

WEIGHTS_PATH = args['model_pretrained']

DATASET_PATH = args['dataset_path']

class LemnaConfig(Config):

    NAME = os.path.basename(args['logs'])

    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 1

    STEPS_PER_EPOCH = 1

    DETECTION_MIN_CONFIDENCE = 0.9

    BACKBONE = "resnet50"

    WEIGHT_DECAY = 0.01

class LemnaDataset(utils.Dataset):

    def load_lemna(self, dataset_dir, subset):

        self.add_class("lemna_minor", 1, "lemna_minor")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations_file = args['annot_path'].format(dataset_path=DATASET_PATH,
                                                     subset=subset)
        annotations = json.load(open(annotations_file))
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]

        for a in annotations:
            if type(a['regions']) is dict:
                points = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                points = [r['shape_attributes'] for r in a['regions']]

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "lemna_minor",
                image_id=a['filename'],
                path=image_path,
                width=width, height=height,
                points=points)

    def load_mask(self, image_id):

        image_info = self.image_info[image_id]
        if image_info["source"] != "lemna_minor":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["points"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["points"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "lemna_minor":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    dataset_train = LemnaDataset()
    dataset_train.load_lemna(DATASET_PATH, "train")
    dataset_train.prepare()

    dataset_val = LemnaDataset()
    dataset_val.load_lemna(DATASET_PATH, "val")
    dataset_val.prepare()

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=args['epochs'],
                layers='all')


################################################################################
if __name__ == '__main__':
    print("Weights: ", args['model_pretrained'])
    print("Dataset: ", args['dataset_path'])
    print("Logs: ", args['logs'])

    if not os.path.exists(args['logs']):
        os.makedirs(args['logs'])

    config = LemnaConfig()

    config.display()

    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=args['logs'])

    if args['model_pretrained'].lower() == "last":
        weights_path = model.find_last()
    else:
        weights_path = WEIGHTS_PATH

    model.load_weights(weights_path, by_name=True)
    train(model)

    folder_name = os.path.basename(args['logs'])
    graphs_path = f'{args["graphs_path"]}/{folder_name}_train'
    if not os.path.exists(graphs_path):
        os.makedirs(graphs_path)
    plt.figure(figsize=(8, 8))
    hstr = [model.keras_model.history.history['loss'],
            model.keras_model.history.history['val_loss']]
    plt.plot(hstr[0], label=f'Chybovosť trénovania')
    plt.plot(hstr[1], label=f'Chybovosť validácie')
    ax = plt.gca()
    plt.xlim(0, len(hstr[0]))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#666666', linestyle='-', alpha=0.2)
    plt.title('Chybovosť')
    plt.xlabel('Epochy')
    plt.ylabel('Chybovosť')
    plt.legend()
    plt.savefig(f'{graphs_path}/loss.png')
