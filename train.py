from config import *
from model import *
import tensorflow as tf
from trainfeed import *
from util import *

ix2char = loadindex2char()

saver = tf.train.Saver()
tf.train.export_meta_graph(filename=model_dir+".meta")
ops = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess,model_dir)

for i in range(epochs):
    print('--------epoch:'+str(i))
    for j,(imgs,spar_lb) in enumerate(datagen()):
        ctc_err,_log,_logits = sess.run([loss,logging,logits], feed_dict={images: imgs, sparse_label: spar_lb})
        writer.add_summary(_log)
        print(ctc_err)
        if(j%1)==0:
            _preds = sess.run(dense_decodes,feed_dict={logits:_logits,seq_len:np.tile([35],[batch_size])})#top_n*[[batch_size, max_decoded_length]]
            #densematrix = tf.sparse_to_dense(_preds[0].indices,_preds[0].dense_shape,_preds[0].values,default_value=-1)
            strarr = []
            for line in _preds:
                strarr.append(''.join([ix2char[char_index] if char_index!=-1 else '' for char_index in line]))
            print(strarr)
            #saver.save(sess,model_dir,write_meta_graph=False)



