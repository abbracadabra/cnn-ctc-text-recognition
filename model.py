import tensorflow as tf
from config import *

images = tf.placeholder(dtype=tf.float32,shape=[None,None,None,1],name='image_holder')
sparse_label = tf.sparse.placeholder(tf.int32)
_ = images
_ = tf.layers.conv2d(inputs=_,filters=64,kernel_size=3,strides=(2, 2),padding="SAME")
_ = tf.layers.batch_normalization(_)
_ = tf.nn.leaky_relu(_)
_ = tf.layers.conv2d(inputs=_,filters=128,kernel_size=3,strides=(2, 2),padding="SAME")
_ = tf.layers.batch_normalization(_)
_ = tf.nn.leaky_relu(_)
_ = tf.layers.conv2d(inputs=_,filters=128,kernel_size=3,strides=(2, 2),padding="SAME")
_ = tf.layers.batch_normalization(_)
_ = tf.nn.leaky_relu(_)
_ = tf.layers.conv2d(inputs=_,filters=128,kernel_size=3,strides=(2, 1),padding="SAME")
_ = tf.layers.batch_normalization(_)
_ = tf.nn.leaky_relu(_)
_ = tf.layers.conv2d(inputs=_,filters=128,kernel_size=3,strides=(2, 1),padding="SAME")
_ = tf.layers.batch_normalization(_)
_ = tf.nn.leaky_relu(_)
_ = tf.layers.conv2d(inputs=_,filters=128,kernel_size=3,strides=(1, 1),padding="SAME")#[None,1,35,128]
_ = tf.layers.batch_normalization(_)
_ = tf.squeeze(_,axis=1)#[None,35,128]
time_step = tf.shape(_)[1]#steps
seq_len = tf.tile(tf.expand_dims(time_step,axis=0),[tf.shape(_)[0]])#[batchsize]

# lstmcell = tf.nn.rnn_cell.LSTMCell(num_units=64)
# initial_state = lstmcell.zero_state(batch_size, dtype=tf.float32)
# outputs, _ = tf.nn.dynamic_rnn(cell=lstmcell,inputs=_,sequence_length=seq_len,initial_state=initial_state)#[None,35,64]
logits = tf.layers.dense(_, units = num_chars)#[None,35,num_chars]
logits_timemajor = tf.transpose(logits,[1,0,2])
prob = tf.nn.softmax(_,axis=-1)#[None,35,num_chars]
loss = tf.reduce_mean(tf.nn.ctc_loss(sparse_label, inputs=logits_timemajor, sequence_length=seq_len))

decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=logits_timemajor,sequence_length=seq_len,merge_repeated=False,beam_width=1)#top_n*[[batch_size, max_decoded_length]]
dense_decodes = tf.sparse_to_dense(decodes[0].indices,decodes[0].dense_shape,decodes[0].values,default_value=-1,name='ctc_best_path')

tf.summary.scalar("ctcloss",loss)
logging = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_dir)

