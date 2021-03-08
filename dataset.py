import os
import csv
from PIL import Image, ImageOps
import pandas as pd
import config as cfg
import sys

# pd.options.mode.chained_assignment = None
args = cfg.DATASET_ARGS

if args['current_folder']:
    os.chdir(f'{os.path.realpath(__file__)}\\..\\')

print(os.getcwd())

def rotate_save(path, df_orig, flipped, save_path):
    df = pd.DataFrame()
    image = Image.open(path)

    if flipped:
        flip_text = '_flipped'
        image = ImageOps.flip(image)
    else:
        flip_text = ''

    name = path.split('\\')[-1].split('.')[0]
    image.save(f'{save_path}\\{name}{flip_text}.jpg')
    df = df.append(df_orig, ignore_index=True)

    def rotate_img(degree, image):
        image = image.rotate(degree)
        image_id = f'{name}{flip_text}_{str(degree)}'
        image.save(f'{save_path}\\{image_id}.jpg')
        df_orig.loc[:, ('id')] = image_id
        print(df_orig)
        return df_orig

    for degree in [90, 180, 270]:
        df = df.append(rotate_img(degree, image), ignore_index=True)

    return df

df = pd.read_csv(args['old_csv_path'], sep=args['csv_sep'])
df_new = pd.DataFrame()

for path, dirs, files in os.walk(args['imgs_path']):
    for filename in files:
        name = path + '\\' + filename
        df_find = df[df['id'] == filename.split('.')[0]]
        df_new = df_new.append(rotate_save(name, df_find, False, args['results_path']),
                               ignore_index=True)
        df_new = df_new.append(rotate_save(name, df_find, True, args['results_path']),
                               ignore_index=True)
        # if filename.split('.')[-1] == 'png':
        #     # print(filename)
        #     to_jpg = image.convert('RGB')
        #     to_jpg.save(path + '\\' + filename.split('.')[0] + '.jpg')
        #     os.remove(path + '\\' + filename)
        #     print(path + '\\' + filename.split('.')[0] + '.jpg')

print(df_new)
df_new.to_csv(args['new_csv_path'])
