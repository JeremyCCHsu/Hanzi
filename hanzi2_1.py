# My thoughts:
#   1. Two conv layers => too few layers (hard to interpret)
#       First layer learn corners, but the second layer is almost noise
#       (or at least, un-interpretable)
#   2. Downsampling is too soon (100 -> 20 -> 5)
#   3. conv's out is too large?
#   4. Need better augmentation? Or regularization? (data is too few)
# 


# Hints
#   1. Hex2dec: d = int(h, 16)  # h is a string
#      Dec2hex: h = hex(d)

# How to read png

# Using Scipy
# http://www.scipy-lectures.org/advanced/image_processing/

from __future__ import division
from __future__ import division
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from os import path
import tensorflow as tf

# from __future__ import print_function
import re
from scipy.stats import truncnorm

NUM_CORES = 8
session_conf = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True,
    inter_op_parallelism_threads=NUM_CORES,
    intra_op_parallelism_threads=NUM_CORES)



def read_imgs(iDir):
    ini = 19968
    fin = 40883 
    imgs   = list()
    labels = list()
    for i in range(ini, fin+1):
        filename = path.join(iDir, 'U%d.png' % i)
        if path.isfile(filename):
            im = misc.imread(filename)
            if np.sum(im[:,:,0]) > 1:   # Valid (Not an empty img)
                im = im[:,:,0] / 255.0
                # im = np.reshape(im, ())
                imgs.append(im)
                labels.append(i)
    return imgs, labels


# def findBusou(i, top10):
#     '''
#     0: not in the top 10 busous
#     '''
#     k = -1;
#     for j in range(len(top10)-1):
#         if i > top10[j] and i < top10[j+1]:
#             k = j
#             break
#     if k > -1 and k % 2 == 0:
#         return k/2 +1
#     else:
#         return 0

def findBusou(i, top10):
    '''
    0: not in the top 10 busous
    '''
    k = -1;
    for j in range(len(top10)-1):
        if i > top10[j] and i < top10[j+1]:
            k = j
            break
    if k > -1 and k % 2 == 0:
        return k/2
    else:
        return 10


def toBushou(labels):
    '''
    top10 = [
        [1,  '8278', '864C'], 
        [2,  '6C34', '706A'], 
        [3,  '6728', '6B1F'],
        [4,  '53E3', '56D6'],
        [5,  '5FC3', '6207'],
        [6,  '91D1', '9576'],
        [7,  '5973', '5B4F'],
        [8,  '706B', '7229'],
        [9,  '4EC5', '513E'],
        [10, '7CF8', '7F35']]

    top10 = [
        [9,  '4EC5', '513E'],
        [4,  '53E3', '56D6'],
        [5,  '5FC3', '6207'],
        [7,  '5973', '5B4F'],       
        [3,  '6728', '6B1F'],
        [2,  '6C34', '706A'], 
        [8,  '706B', '7229'],
        [10, '7CF8', '7F35'],
        [1,  '8278', '864C'], 
        [6,  '91D1', '9576']]
    '''
    bushouLabel = list()
    top10 = [
        '4EC5', '513E',
        '53E3', '56D6',
        '5973', '5B4F',       
        '5FC3', '6207',
        '6728', '6B1F',
        '6C34', '706A', 
        '706B', '7229',
        '7CF8', '7F35',
        '8278', '864C', 
        '91D1', '9576']
    top10 = map(lambda h: int(h, 16), top10)
    for i in labels:
        kth = findBusou(i, top10)
        bushouLabel.append(kth)
    return bushouLabel


def onehot(labels, K):
    lab = np.zeros((labels.shape[0], K))
    # lab[:, labels] = 1.0
    for i in range(lab.shape[0]):
        j = labels[i] -1    
        lab[i, j] = 1.0
    return lab



def generate_batch(imgs, laby):
    N = imgs.shape[0]



# iDir = 'ChineseGlyphs'
iDir = 'TWKai98_50x50'

imgs, labels = read_imgs(iDir)
labs = toBushou(labels)

# [TODO] Input check must be earlier

szImg = imgs[0].shape[0]
# [TODO] Including Label-0  might be unwise (imbalanced dataset)
assert(imgs[0].shape[0] == imgs[0].shape[1])    # check square input

# np.concat > shuffle
# imgs = np.asarray(imgs)
# labels = np.asarray(labels)

# idx = all(labs > 0)
labs = np.array(labs)
idx = labs < 10

# imgs = imgs[idx]

img1 = list()
for i in range(len(imgs)):
    if idx[i]:
        img1.append(imgs[i])

imgs = img1
labs = labs[idx]

szTrn = 7000   # Total: 7459
# szImg = imgs[0].shape[0]

# Class 0: 632
# Class 1: 754
# Class 2: 475
# Class 3: 579
# Class 4: 1014
# Class 5: 1077
# Class 6: 445
# Class 7: 572
# Class 8: 979
# Class 9: 932


x = zip(imgs, labs)
np.random.shuffle(x)
imgs = [i for i, l in x]    # np.asarray
labs = [l for i, l in x]

img_trn = imgs[:szTrn]
lab_trn = labs[:szTrn]

img_vld = imgs[szTrn:]
lab_vld = labs[szTrn:]

imgs = None
labs = None



x = None
img_trn = np.reshape(img_trn, (-1, szImg, szImg, 1))
lab_trn = np.asarray(lab_trn)

img_vld = np.reshape(img_vld, (-1, szImg, szImg, 1))
lab_vld = np.asarray(lab_vld)


# laby = onehot(labs, 11)     # [Makeshift]
# lab_y_trn = onehot(lab_trn, 10)     # [Makeshift]
# lab_y_vld = onehot(lab_vld, 10)     # [Makeshift]
lab_y_trn = lab_trn.astype(np.int64)
lab_y_vld = lab_vld.astype(np.int64)

# batch generator

# random start filters
# relu
# max-pooling (tf.nn.max_pool)
# 3 layers
# szImg = 100
szBatch = 64

szPatch1 = 5
szPatch2 = 5
szPatch3 = 3

szStride1 = 2
szStride2 = 1     # szPatch2/2
szStride3 = 1
# szDownSmp = 2
ksize = 3
kstrd = 2
nChannel = 1
nFilter1 = 64
nFilter2 = 128
nHidden1 = 394
nHidden2 = 128
nClass = 10


nFilter3 = 256




TOWER_NAME = 'tower'
def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _images_summary(images, tag):
    sz = images.get_shape().as_list()[1]
    # ch = images.get_shape().as_list()[3]
    # for i in range(10):
    #     img = tf.slice(
    #         images, 
    #         begin=[0, 0, 0, i],
    #         size=[szBatch, sz, sz, 1])
    #     tf.image_summary(tag, img)
    img = tf.slice(
        images,
        begin=[0,0,0,0],
        size=[szBatch, sz, sz, 1])
    tf.image_summary(tag, img)


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  var = tf.Variable(
    tf.truncated_normal(
        shape=shape, 
        stddev=stddev), 
    name=name)
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

# a2 = 1e-1   # L1=2.6821e-01, L2=6.2035e-03, Sim=2.0267e+00
a2 = 1e-6
stddev = 1e-4
graph = tf.Graph()
with graph.as_default():
    lr = tf.placeholder(tf.float32)
    Xi = tf.placeholder(
        tf.float32, 
        shape=[None, szImg, szImg, nChannel])
    Yi = tf.placeholder(tf.int64, shape=[None])
    # Conv 1
    _images_summary(Xi, 'original')
    with tf.variable_scope('conv1') as scope:
        W1 = tf.Variable(
            tf.truncated_normal(
                shape=[szPatch1, szPatch1, nChannel, nFilter1],
                stddev=1.0/(szPatch1*szPatch1)))
        # W1 = _variable_with_weight_decay(
        #     'W1', 
        #     shape=[szPatch1, szPatch1, nChannel, nFilter1], 
        #     stddev=stddev, 
        #     wd=a2)
        b1 = tf.Variable(tf.zeros(shape=[nFilter1]))
        c1 = tf.nn.conv2d(Xi, W1, [1, szStride1, szStride1, 1], 
            padding='SAME')
        h1 = tf.nn.bias_add(c1, b1)
        f1 = tf.nn.relu(h1, name=scope.name)
        _activation_summary(f1)
        _images_summary(f1, 'conv1')
    norm1 = tf.nn.lrn(f1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')
    p1 = tf.nn.max_pool(norm1,
        ksize=[1, ksize, ksize, 1],
        strides=[1, kstrd, kstrd, 1],
        padding='VALID')
    L2_on_w1 = tf.reduce_sum(tf.reduce_sum(tf.square(W1), [0, 1]))
    L1_on_w1 = tf.reduce_sum(tf.reduce_sum(tf.abs(W1), [0, 1]))
    # Conv 2
    with tf.variable_scope('conv2') as scope:
        W2 = tf.Variable(
            tf.truncated_normal(
                shape=[szPatch2, szPatch2, nFilter1, nFilter2],
                stddev=1.0/(szPatch2*szPatch2)))
        # W2 = _variable_with_weight_decay(
        #     'W2', 
        #     shape=[szPatch2, szPatch2, nFilter1, nFilter2], 
        #     stddev=stddev, 
        #     wd=a2)
        b2 = tf.Variable(tf.zeros(shape=[nFilter2]))
        c2 = tf.nn.conv2d(p1, W2, [1, szStride2, szStride2, 1], 
            padding='SAME')
        h2 = tf.nn.bias_add(c2, b2)
        f2 = tf.nn.relu(h2, 'conv2')
        _activation_summary(f2)
        _images_summary(f2, 'conv2')
    norm2 = tf.nn.lrn(f2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
    p2 = tf.nn.max_pool(norm2,
        ksize=[1, ksize, ksize, 1],
        strides=[1, kstrd, kstrd, 1],
        padding='VALID')
    L2_on_w2 = tf.reduce_sum(tf.reduce_sum(tf.square(W2), [0, 1]))
    L1_on_w2 = tf.reduce_sum(tf.reduce_sum(tf.abs(W2), [0, 1]))
    # Conv 3
    with tf.variable_scope('conv3') as scope:
        W3 = tf.Variable(
            tf.truncated_normal(
                shape=[szPatch3, szPatch3, nFilter2, nFilter3],
                stddev=1.0/(szPatch3*szPatch3)))
        # weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        # W3 = _variable_with_weight_decay(
        #     'W3', 
        #     shape=[szPatch3, szPatch3, nFilter2, nFilter3], 
        #     stddev=stddev, 
        #     wd=a2)
        b3 = tf.Variable(tf.zeros(shape=[nFilter3]))
        c3 = tf.nn.conv2d(p2, W3, [1, szStride3, szStride3, 1], 
            padding='SAME')
        h3 = tf.nn.bias_add(c3, b3)
        f3 = tf.nn.relu(h3, 'conv2')
        _activation_summary(f3)
        _images_summary(f3, 'conv3')
    norm3 = tf.nn.lrn(f3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm3')
    p3 = tf.nn.max_pool(norm3,
        ksize=[1, ksize, ksize, 1],
        strides=[1, kstrd, kstrd, 1],
        padding='VALID')
    L2_on_w3 = tf.reduce_sum(tf.reduce_sum(tf.square(W3), [0, 1]))
    L1_on_w3 = tf.reduce_sum(tf.reduce_sum(tf.abs(W3), [0, 1]))
    # Locals
    dim = 1
    for d in p3.get_shape()[1:].as_list():
        dim *= d
    Xf = tf.reshape(p3, [-1, dim])
    Wf1 = tf.Variable(
        tf.truncated_normal(
            shape=[dim, nHidden1],
            stddev=1.0/dim))
    bf1 = tf.Variable(tf.zeros([nHidden1]))
    hf1 = tf.nn.relu(tf.matmul(Xf, Wf1) + bf1)
    L2_on_wf1 = tf.reduce_sum(tf.reduce_sum(tf.square(Wf1), 0))
    L1_on_wf1 = tf.reduce_sum(tf.reduce_sum(tf.abs(Wf1), 0))
    Wf2 = tf.Variable(
        tf.truncated_normal(
            shape=[nHidden1, nHidden2],
            stddev=1.0/nHidden1))
    bf2 = tf.Variable(tf.zeros([nHidden2]))
    hf2 = tf.nn.relu(tf.matmul(hf1, Wf2) + bf2)
    Wf3 = tf.Variable(
        tf.truncated_normal(
            shape=[nHidden2, nClass],
            stddev=1.0/nHidden2))
    bf3 = tf.Variable(tf.zeros([nClass]))
    L2_on_wf2 = tf.reduce_sum(tf.reduce_sum(tf.square(Wf2), 0))
    L1_on_wf2 = tf.reduce_sum(tf.reduce_sum(tf.abs(Wf2), 0))
    logits = tf.matmul(hf2, Wf3) + bf3
    Yo = tf.nn.softmax(logits)
    # simi_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits, Yi)
    simi_loss = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, Yi),
        name='cross_entropy_mean')
    # tf.add_to_collection('losses', simi_loss)
    L1_loss = a2*(L1_on_w1 + L1_on_w2 + L1_on_w3 + L1_on_wf1 + L1_on_wf2)
    L2_loss = a2*(L2_on_w1 + L2_on_w2 + L2_on_w3 + L2_on_wf1 + L2_on_wf2)
    loss = simi_loss + L1_loss + L2_loss
    # train = tf.train.AdagradOptimizer(lr).minimize(loss)
    train = tf.train.MomentumOptimizer(lr, 0.95).minimize(loss)
    # tf.add_to_collection('losses', L1_loss)
    # tf.add_to_collection('losses', L2_loss)
    tf.scalar_summary(simi_loss.op.name, simi_loss)
    tf.scalar_summary(L1_loss.op.name, L1_loss)
    tf.scalar_summary(L2_loss.op.name, L2_loss)
    top_k_op = tf.nn.in_top_k(logits, Yi, 1)
    saver = tf.train.Saver(tf.all_variables())
    summary_op = tf.merge_all_summaries()
    # for l in losses:
    #     tf.scalar_summary(l.op.name, l)
    # yhat = tf.argmax(Yo)
    # acc = 


nEpoch = 20000
nSmp = img_trn.shape[0]
nBatch = nSmp // szBatch
nSmpVld = img_vld.shape[0]
nBatchVld = nSmpVld // szBatch

# nEpoch = 10000
# with tf.Session(graph=graph, config=session_conf) as session:
session = tf.InteractiveSession(graph=graph)    # , config=session_conf
# session = tf.Session(graph=graph, config=session_conf)    # , config=session_conf
init = tf.initialize_all_variables()
session.run(init)
# summary_writer = tf.train.SummaryWriter('tmp', session.graph)
summary_writer = tf.train.SummaryWriter(logdir='tmp')
err_log = list()
cursor = 0
for epoch in range(nEpoch):
    # err = 0.0
    # errSim = 0.0
    err, errSim, errl1, errl2, tokens = (0.0, 0.0, 0.0, 0.0, 0.0)
    # cifar10's batch is from a generator (without explicit loops)
    for i in range(nBatch-1):
        if cursor + szBatch >= nSmp:
            cursor = (cursor + szBatch) % nSmp
        index = range(cursor, cursor+szBatch)
        noise = truncnorm.rvs(
            a= 0.0, 
            b=0.6, 
            loc=0, 
            scale=0.2, 
            size=(szBatch, szImg, szImg, 1))
        batch_x = img_trn[index] + noise
        # batch_x = img_trn[index]
        batch_y = lab_y_trn[index]
        # 1e-2: explode, 1e-3: snail, 1e-4: frozen 1e-5: frozen
        feed_dict = {Xi: batch_x, Yi: batch_y, lr: 1e-4}
        _, sl, l1, l2, ls, tpk = session.run(
            [train, simi_loss, L1_loss, L2_loss, loss, top_k_op], 
            feed_dict=feed_dict)
        err += ls
        errl1 += l1
        errl2 += l2
        errSim += np.sum(sl)
        tokens += np.sum(tpk)
    err /= nBatch
    errSim /= nBatch
    errl1 /= nBatch
    errl2 /= nBatch
    err_log.append(err)
    tokens /= nSmp
    # feed_dict = {Xi: img_vld, Yi: lab_y_vld, lr: 1e-2}
    # err_vld, yhat = session.run(
    #     [simi_loss, Yo], 
    #     feed_dict=feed_dict)
    print 'Epoch %4d, err trn = %.4e (L1=%.4e, L2=%.4e, Sim=%.4e) Acc=%.4f' % (
        epoch, err, errl1, errl2, errSim, tokens)
    # print '            err vld = %.4e' % (err_vld / (lab_y_vld.shape[0] // szBatch))
    if epoch % 10 == 0:
        checkpoint_path = 'model.ckpt'
        saver.save(session, checkpoint_path, global_step=epoch)
        # why do I need to feed this?
        summary_str = session.run(
            summary_op,
            feed_dict=feed_dict)   
        summary_writer.add_summary(summary_str, epoch)
        total_hit = 0
        # for i in range(nBatchVld-1):
        #     # if (i+1)*szBatch > nSmpVld:
        #     #     i = (i+1)*szBatch % nSmpVld
        #     # if cursor + szBatch >= nSmp:
        #     #     cursor = (cursor + szBatch) % nSmp
        #     index = range(i, i+szBatch)
        #     batch_x = img_vld[index]
        #     batch_y = lab_y_vld[index]
        #     feed_dict = {Xi: batch_x, Yi: batch_y}
        #     predict_hit = session.run(
        #         [top_k_op], 
        #         feed_dict=feed_dict)            
        #     total_hit += np.sum(predict_hit)
        feed_dict = {Xi: img_vld[index], Yi: lab_y_vld[index]}
        predict_hit = session.run(
            [top_k_op], 
            feed_dict=feed_dict)            
        # total_hit += np.sum(predict_hit)
        # prec = float(total_hit) / ((nBatchVld-1)*szBatch)
        prec = np.sum(predict_hit) /nSmpVld
        print 'Validation precision: %.4f' % prec



# # ==== Test ====
# fir1, fir2 = session.run([W1, W2])

# plt.figure()
# for i in range(16):
#     plt.subplot(4, 4, i+1)
#     plt.imshow(fir1[:,:,0,i])

# plt.savefig('W1.png')

# plt.figure()
# for i in range(16):
#     plt.subplot(4, 4, i+1)
#     plt.imshow(fir2[:,:,0,i])

# plt.savefig('W2.png')

# map1, map2 = session.run([h1, h2], feed_dict=feed_dict)

# plt.figure()
# for i in range(16):
#     plt.subplot(4, 4, i+1)
#     plt.imshow(map1[0,:,:,i])

# plt.savefig('M1.png')
# # plt.show()

# plt.figure()
# for i in range(16):
#     plt.subplot(4, 4, i+1)
#     plt.imshow(map2[0,:,:,i])

# plt.savefig('M2.png')
# # plt.show()


# plt.figure()
# plt.plot(err_log[:200])
# plt.show()



# # ============
# n = 5000
# plt.figure()
# plt.imshow(imgs[n])
# plt.title(labels[n])
# plt.savefig('test.png')
# print unichr(labels[n])
# # ============

# Using Tensorflow
# http://stackoverflow.com/questions/33648322/tensorflow-image-reading-display

# Using PIL
# http://www.geeks3d.com/20100930/tutorial-first-steps-with-pil-python-imaging-library/#p02
