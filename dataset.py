import os
import csv
from PIL import Image, ImageOps
import pandas as pd
import config as cfg
import sys

args = cfg.DATASET_ARGS

if args['current_folder']:
    os.chdir(f'{os.path.realpath(__file__)}\\..\\')

def rotate_save(im_path, df_orig, flipped, save_path):
    df_rotated = pd.DataFrame()
    image = Image.open(im_path)

    if flipped:
        flip_text = '_flipped'
        image = ImageOps.flip(image)
    else:
        flip_text = ''

    im_base = im_path.split('\\')[-1].split('.')[0]
    im_name = f"{im_base}{flip_text}"
    image.save(f'{save_path}\\{im_name}.jpg')
    df_orig['id'] = im_name
    df_rotated = df_rotated.append(df_orig, ignore_index=True)

    def rotate_img(degree, image):
        image = image.rotate(degree)
        image_id = f'{im_name}_{str(degree)}'
        image.save(f'{save_path}\\{image_id}.jpg')
        df_orig.loc[:, ('id')] = image_id
        return df_orig

    for degree in [90, 180, 270]:
        df_rotated = df_rotated.append(rotate_img(degree, image),
                                       ignore_index=True)

    print(df_rotated)

    return df_rotated

df = pd.read_csv(args['old_csv_path'], sep=args['csv_sep'])
df_new = pd.DataFrame()

for path, dirs, files in os.walk(args['imgs_path']):
    for filename in files:
        name = path + '\\' + filename
        df_find = df[df['id'] == filename.split('.')[0]]
        df_new = df_new.append(rotate_save(name, df_find, False, args['results_path']),
                               ignore_index=True)
        df_find = df[df['id'] == filename.split('.')[0]]

        df_new = df_new.append(rotate_save(name, df_find, True, args['results_path']),
                               ignore_index=True)

print(df_new)
df_new.to_csv(args['new_csv_path'])
