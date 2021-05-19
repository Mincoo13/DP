import config as cfg
import numpy as np
import os
from glob import glob
from skimage import io
from utils import annotation_to_mask, mask_to_rgb
args = cfg.VIS_ANNOT_DATA

masks = annotation_to_mask(args)
imgs_path = f"{args['imgs_path']}/*.jpg"
out_path = os.path.join(args['output_path'], '{suffix}')
if not os.path.exists(args['output_path']):
    os.makedirs(args['output_path'])
for im_path in glob(imgs_path):
    img_save_name = os.path.basename(im_path)
    img_id = img_save_name.split('.')[0]
    if img_id in masks.keys():
        rgb_mask = mask_to_rgb(np.array(masks[img_id]), predicted=False,
                               color=args['mask_color'])
        io.imsave(out_path.format(suffix=img_save_name), rgb_mask)
