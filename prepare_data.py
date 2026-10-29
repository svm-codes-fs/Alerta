"""Builds data/train, data/valid, data/test for model.py from the
Driver Drowsiness Dataset (DDD), using the project's own Haar eye cascades.

Each DDD face image is cropped to its left/right eye exactly like
drowsinessdetection.py does (gray -> detectMultiScale -> first box -> 24x24).
Eye crops from 'Drowsy' faces go to Closed/, 'Non Drowsy' faces go to Open/.

The split is done BY SUBJECT (filename prefix, e.g. A0001 / a0001 = subject 'a'),
so test eyes always belong to people the model never saw during training.
"""
import glob, io, os, random, re, shutil
from multiprocessing import Pool

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

RAW = 'data/raw'          # DDD parquet shards (HF mirror: akahana/Driver-Drowsiness-Dataset)
OUT = 'data'
CLASS = {0: 'Closed', 1: 'Open'}   # DDD label 0 = Drowsy, 1 = Non Drowsy
N_TEST, N_VALID = 5, 4             # subjects held out


def init_worker():
    global leye, reye
    leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')


def crop_eyes(args):
    name, data = args
    gray = cv2.cvtColor(np.array(Image.open(io.BytesIO(data)).convert('RGB')), cv2.COLOR_RGB2GRAY)
    eyes = []
    for side, casc in (('L', leye), ('R', reye)):
        boxes = casc.detectMultiScale(gray)
        if len(boxes):
            x, y, w, h = boxes[0]
            eyes.append((side, cv2.resize(gray[y:y + h, x:x + w], (24, 24))))
    return name, eyes


def batches():
    # stream the shards in small batches (whole dataset is ~2.7 GB)
    for f in sorted(glob.glob(os.path.join(RAW, '*.parquet'))):
        for b in pq.ParquetFile(f).iter_batches(batch_size=500):
            d = b.to_pydict()
            yield [(im['path'], im['bytes']) for im in d['image']], d['label']


def subject(p):
    return re.match(r'[A-Za-z]+', p).group().lower()


def main():
    label = {}
    for f in sorted(glob.glob(os.path.join(RAW, '*.parquet'))):
        for b in pq.ParquetFile(f).iter_batches(batch_size=2000, columns=['label', 'image']):
            d = b.to_pydict()
            label.update({im['path']: lab for im, lab in zip(d['image'], d['label'])})
    subjects = sorted({subject(p) for p in label})
    random.Random(42).shuffle(subjects)
    split_of = {s: 'test' for s in subjects[:N_TEST]}
    split_of.update({s: 'valid' for s in subjects[N_TEST:N_TEST + N_VALID]})
    print('test subjects :', sorted(subjects[:N_TEST]))
    print('valid subjects:', sorted(subjects[N_TEST:N_TEST + N_VALID]))

    for s in ('train', 'valid', 'test'):
        shutil.rmtree(os.path.join(OUT, s), ignore_errors=True)
        for c in CLASS.values():
            os.makedirs(os.path.join(OUT, s, c))

    counts = {}
    with Pool(4, initializer=init_worker) as pool:
        for imgs, _ in batches():
            for name, eyes in pool.imap_unordered(crop_eyes, imgs, chunksize=16):
                split, cls = split_of.get(subject(name), 'train'), CLASS[label[name]]
                for side, eye in eyes:
                    cv2.imwrite(os.path.join(OUT, split, cls, f'{name[:-4]}_{side}.png'), eye)
                    counts[(split, cls)] = counts.get((split, cls), 0) + 1
    for k in sorted(counts):
        print(k, counts[k])


if __name__ == '__main__':
    main()
