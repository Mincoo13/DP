import os
import sys
import json
import shutil
import argparse
import datetime
import skimage.draw
import numpy as np
import pandas as pd
import config as cfg
from glob import glob
from os import listdir
from PIL import Image
from skimage import io
from sklearn import metrics

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mrcnn import model as modellib, visualize
from mrcnn.config import Config

from utils import annotation_to_mask, generate_segment_graph
from utils import mask_to_rgb, color_splash, write_result

args = cfg.SEGMENT_PRED_ARGS

if args['current_folder']:
    os.chdir(f'{os.path.realpath(__file__)}\\..\\')

if sys.platform == 'linux2' and 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')

class TrainConfig(Config):

    NAME = "68_20_resnet50_all_layers"

    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 1

    STEPS_PER_EPOCH = 68

    BACKBONE = 'resnet50'

    WEIGHT_DECAY = 0.01

    DETECTION_MIN_CONFIDENCE = 0.9

class InferenceConfig(TrainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def load_image(im_path):
    image = io.imread(im_path, plugin='matplotlib')

    if len(image.shape) > 2 and image.shape[2] > 3:
        image = image[:, :, :3]
    elif len(image.shape) == 2:
        image = np.stack((image, image, image), axis=2)

    return image

def inference():
    configuration = InferenceConfig()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model_name = os.path.basename(args['model_path']).split('.')[0]
    model = modellib.MaskRCNN(mode="inference", config=configuration,
                            model_dir=args['model_path'])
    model.load_weights(args['model_path'], by_name=True)
    df = pd.DataFrame(columns=['id', 'actual_count', 'predicted_count',
                    'actual_area', 'predicted_area'])
    df_actual = pd.read_csv(args['csv_path'], sep=",")
    df_actual = df_actual.set_index('id')
    actual_masks = annotation_to_mask(args)
    imgs_path = f"{args['imgs_path']}/*.jpg"
    output_path = f"{args['output_path']}/{model_name}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if args['intent'] == 'mask':
        with open(os.path.join(output_path, 'leafCounts.csv'), 'a') as count_file:
            count_file.write("Image, Count\n")
            for im_path in glob(imgs_path):
                out_path = os.path.join(output_path, '{suffix}')
                image = load_image(im_path)
                results = model.detect([image], verbose=True)
                masks = results[0]['masks']
                rgb_mask = mask_to_rgb(masks, predicted=True, 
                                       color=args['mask_color'])
                
                img_save_name = os.path.basename(im_path)
                io.imsave(out_path.format(suffix=img_save_name), rgb_mask)
                
                img_id = img_save_name.split('.')[0]
                
                if 'after' in img_id:
                    img_prefix = f"{img_id.split('after')[0]}after"
                else:
                    img_prefix = img_id.split('_')[0]
                if img_prefix in actual_masks.keys():
                    actual_area = np.count_nonzero(np.array(actual_masks[img_prefix]))
                else:
                    actual_area = None
                count_file.write(img_id + ", " + str(masks.shape[2]) + "\n")
                print(df_actual.loc[img_id])
                actual_count = int(df_actual.loc[img_id]['count'])
                predicted_area = np.count_nonzero(masks)
                df = df.append({'id': img_id, 'actual_count': actual_count,
                                'predicted_count': masks.shape[2],
                                'actual_area': actual_area,
                                'predicted_area': predicted_area},
                            ignore_index=True)

            graph_folder = os.path.basename(args['model_path']).split('.')[0]
            graph_folder = f'{args["graphs_path"]}\\{graph_folder}_segmentation'
            if not os.path.exists(graph_folder):
                os.makedirs(graph_folder)

            prediction_path = f'{args["predictions_path"]}/{model_name}.csv'
            df.to_csv(prediction_path, index=False, sep=';')
            df = df.sort_values(by=['actual_count'])
            y_test = df.actual_count.to_numpy()
            y_pred = df.predicted_count.to_numpy()
            count_abs_error = metrics.mean_absolute_error(y_test, y_pred)
            count_mean_sq_error = metrics.mean_squared_error(y_test, y_pred)

            generate_segment_graph(graph_folder, y_pred, y_test, intent='leaf_count')

            df = df.dropna()
            df = df.sort_values(by=['actual_area'])
            y_test = df.actual_area.to_numpy()
            y_pred = df.predicted_area.to_numpy()
            area_abs_error = metrics.mean_absolute_error(y_test, y_pred)
            area_mean_sq_error = metrics.mean_squared_error(y_test, y_pred)
            generate_segment_graph(graph_folder, y_pred, y_test, intent='area')

            fieldnames = ['model', 'count_mae', 'count_mse', 'area_mae', 'area_mse']
            result_row = {'model': model_name,
                        'count_mae': count_abs_error,
                        'count_mse': count_mean_sq_error,
                        'area_mae': area_abs_error,
                        'area_mse': area_mean_sq_error}
            write_result(args['results_path'], fieldnames, result_row)
    elif args['intent'] == 'splash':
        splash_folder = f"{output_path}_splash_{args['splash_color']}"
        if not os.path.exists(splash_folder):
            os.makedirs(splash_folder)
        for im_path in glob(imgs_path):
            out_path = os.path.join(splash_folder, '{suffix}')
            print("Saving prediction for", im_path, "at", out_path)
            image = load_image(im_path)
            results = model.detect([image], verbose=True)
            splash = color_splash(image, results[0]['masks'],
                                color=args['splash_color'])
            img_save_name = os.path.basename(im_path)
            io.imsave(out_path.format(suffix=img_save_name), splash)
    else:
        print(f'Zamer "{args["intent"]}"" neexistuje. Vyberte "splash" alebo "mask".')

if __name__ == '__main__':
    inference()
