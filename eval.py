import tensorflow as tf
from config import *
from PIL import Image
import numpy as np
from util import *

ix2char = loadindex2char()

tf.train.import_meta_graph(model_dir+'.meta')
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess,model_dir)

imarr = []
im = Image.open(r'C:\迅雷下载\Synthetic Chinese String Dataset\images\20436312_1683447152.jpg')
ima = np.array(im.convert('L'))
imarr.append(np.expand_dims(ima,axis=-1))

image_holder = tf.get_default_graph().get_tensor_by_name("image_holder:0")
best_path = tf.get_default_graph().get_tensor_by_name("ctc_best_path:0")
_preds = sess.run(best_path, feed_dict={image_holder: np.array(imarr)})
strarr = []
for line in _preds:
    strarr.append(''.join([ix2char[char_index] if char_index!=-1 else '' for char_index in line]))
print(strarr)



