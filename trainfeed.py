from config import *
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from pathlib import Path

def datagen():
    lf = open(label_path, encoding='utf-8')
    ixarr = []
    imarr = []
    valarr = []
    for line in lf:
        line = line.strip()
        linebits = line.split()
        imageName = linebits[0]
        labelindices = linebits[1:11]
        imf = os.path.join(image_dir,imageName)
        if not Path(imf).exists():
            continue
        im = Image.open(imf)
        ima = np.array(im.convert('L'))
        imarr.append(np.expand_dims(ima,axis=-1))
        ixarr += [[len(imarr)-1,i] for i in range(10)]
        valarr += [v for v in labelindices]
        if len(imarr) == batch_size:
            spar_lb = tf.SparseTensorValue(ixarr, valarr, [batch_size, 10])
            yield np.array(imarr), spar_lb
            imarr = []
            ixarr = []
            valarr = []










