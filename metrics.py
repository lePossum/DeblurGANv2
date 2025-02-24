from sklearn.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from os import listdir
from matplotlib.pyplot import imread
import copy
import numpy as np
from skimage import img_as_float
import os
import random

def prepare_img1(img):
    image = copy.copy(img)
    if len(image.shape) == 2:
        image = np.stack((image, image, image))
        image = np.transpose(image, axes=(1,2, 0))
    if (image.shape[2] == 4):
        image = image[:,:,:3]
    return img_as_float(image)

def calc_metrics(predicted_dir, target_dir, list_len = None):
    pred_files = listdir(predicted_dir)
    targ_files = listdir(target_dir)
    if (not list_len is None):
        pred_files = pred_files[:list_len]
    l = len(pred_files)
    metrics = dict()
    metrics['PSNR'] = 0
    metrics['SSIM'] = 0
    for idx in range(len(pred_files)):
        p = prepare_img1(imread(os.path.join(predicted_dir, pred_files[idx])))

        y_size = int(0.8 * (p.shape[0]))
        x_size = int(0.8 * (p.shape[1]))
        y = random.randrange(p.shape[0] - y_size)
        x = random.randrange(p.shape[1] - x_size)
        
        p = p[y : y + y_size, x : x + x_size]
        if (pred_files[idx] in targ_files):
            t = prepare_img1(imread(os.path.join(target_dir, pred_files[idx])))[y : y + y_size, x : x + x_size]
        # try:
        metrics['PSNR'] += PSNR(t,p)
        metrics['SSIM'] += SSIM(t,p, multichannel=True)
        # except:
        #     l -= 1
        #     print('fovno')

    metrics['PSNR'] /= l
    metrics['SSIM'] /= l
    return metrics

def mm(img):
    print ("Image min : " ,img.min() , ", max : " , img.max())

def norm(img):
    im = img-img.min()
    im /= im.max()
    return im
