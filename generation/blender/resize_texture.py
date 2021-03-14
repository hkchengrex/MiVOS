import sys
import os
import cv2

from multiprocessing import Pool
from progressbar import progressbar

input_dir = sys.argv[1]
output_dir = sys.argv[2]

min_size = 512

def process_fun(sub_dir):
    this_in_dir = os.path.join(input_dir, sub_dir)
    this_out_dir = os.path.join(output_dir, sub_dir)
    os.makedirs(this_out_dir, exist_ok=True)

    for f in os.listdir(this_in_dir):
        img = cv2.imread(os.path.join(this_in_dir, f))
        if img is None:
            continue
        h, w, _ = img.shape

        scale = min(h, w) / min_size

        img = cv2.resize(img, (int(w/scale), int(h/scale)), interpolation=cv2.INTER_AREA)
        if len(img.shape) == 3:
            img = img[0:min_size, 0:min_size, :]
        else:
            img = img[0:min_size, 0:min_size]
        cv2.imwrite(os.path.join(this_out_dir, os.path.basename(f)), img)

if __name__ == '__main__':
    pool = Pool() 
    chunksize = 1

    os.makedirs(output_dir, exist_ok=True)
    for _ in progressbar(pool.map(process_fun, os.listdir(input_dir)), chunksize):
        pass

    print('All done.')