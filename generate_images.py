from os import listdir
from os.path import join
import matplotlib.pyplot as plt
import argparse
from matplotlib.pyplot import imread
from skimage.util import random_noise
from scipy.ndimage import gaussian_filter
import random
from scipy.signal import convolve2d
from multiprocessing import Pool
import threading
import queue

from lib import *
from metrics import prepare_img1


def convolve_img(image, args):
    # ker_len = args.len
    # ker_len = random.randint(3, 19)
    # ker = make_ker(ker_len, 0)
    ker_len = args.len
    ker_angle = args.angle
    ker = make_ker(ker_len, ker_angle)
    img = image
    pad = ker_len // 2
    if len(img.shape) == 2:
        img = np.stack((image, image, image))
        img = np.transpose(image, axes=(1,2, 0))
    r = convolve2d(np.pad(img[:,:,0], pad, 'edge'), ker, mode='valid'); #r /= r.max()
    r = gaussian_filter(r, sigma=0.7)
    g = convolve2d(np.pad(img[:,:,1], pad, 'edge'), ker, mode='valid'); #g /= g.max()
    g = gaussian_filter(g, sigma=0.7)
    b = convolve2d(np.pad(img[:,:,2], pad, 'edge'), ker, mode='valid'); #b /= b.max()
    b = gaussian_filter(b, sigma=0.7)
    image = np.stack((r, g, b))
    image = np.clip(image, 0., 1.)
    image = np.transpose(image, axes=(1,2, 0))
    return image

def process_image(path, idx_to_save, args, amount_on_picture):
    # try:
        cur_img = prepare_img1(imread(path))
        blurred = convolve_img(cur_img, args)
        noised = random_noise(blurred, var=0.0001)
        h, w = cur_img.shape[:2]

        if (h > IMG_SIZE and w > IMG_SIZE):
            for idx in range(amount_on_picture):
                y = random.randrange(h - IMG_SIZE)
                x = random.randrange(w - IMG_SIZE)
                plt.imsave(join(args.to_save, 'blurred/') + 'img_' + str(idx_to_save) + '.png', noised[y : y + IMG_SIZE, x : x + IMG_SIZE])
                plt.imsave(join(args.to_save,   'sharp/') + 'img_' + str(idx_to_save) + '.png', cur_img[y : y + IMG_SIZE, x : x + IMG_SIZE])
        return

    # except Exception as e: # work on python 2.x
    #     print('caught ex', str(e))

def wrapper_targetFunc(f, q, dir_to_save, amount_on_picture):
    while True:
        try:
            work = q.get(timeout=1)  # or whatever
        except queue.Empty:
            return
        f(work[0], work[1], dir_to_save, amount_on_picture)
        q.task_done()


def generate_pics(paths, args, amount_on_picture = 1):
    dir_to_save = args.to_save
    make_directory(dir_to_save)
    make_directory(join(dir_to_save, 'blurred'))
    make_directory(join(dir_to_save, 'sharp'))
    # for p in paths:
    #     process_image(p, dir_to_save, amount_on_picture, break_on)

    q = queue.Queue()
    for ptf in paths:
        q.put_nowait(ptf)
    for _ in range(8):
        threading.Thread(target=wrapper_targetFunc,
                        args=(process_image, q, args, amount_on_picture)).start()
    q.join()

def generate_pics_one_thread(paths, args, amount_on_picture = 1):
    dir_to_save = args.to_save
    make_directory(dir_to_save)
    make_directory(join(dir_to_save, 'blurred'))
    make_directory(join(dir_to_save, 'sharp'))
    for idx, p in enumerate(paths):
        process_image(p, idx, args, amount_on_picture)


def get_paths(directory):
    fnames = listdir(directory)
    fnames.sort()
    return list([os.path.join(directory, item) for (idx, item) in enumerate(fnames)])[:3000]
    # return np.random.permutation(list([(directory + item, idx) for (idx, item) in enumerate(fnames)])[-20:])


def add_parser():
    parser = argparse.ArgumentParser(description='Generate images for neural network calculations')

    parser.add_argument('dir_to_use', type=str, nargs='?', help='directory from to generate')
    parser.add_argument('to_save',  type=str, nargs='?', help='where to save')
    parser.add_argument('len',  type=int, default=9, nargs='?', help='length to blur')
    parser.add_argument('angle',  type=int, default=0, nargs='?', help='angle to blur')
    parser.add_argument('no_mt', default=False, action='store_true', help='use multithreading')
    # parser.set_defaults(mt=False)

    return parser.parse_args()

if __name__ == '__main__':
    args = add_parser()
    
    ### global variables block  ###
    IMG_SIZE = 1000
    random.seed(1337)
    ###                         ###

    if (args.no_mt):
        print('no mt')
        generate_pics_one_thread(get_paths(args.dir_to_use), args, 1)
    else:
        print('mt')
        generate_pics(get_paths(args.dir_to_use), args, 1)
        

