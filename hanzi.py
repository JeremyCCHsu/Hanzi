# Hints
#   1. Hex2dec: d = int(h, 16)  # h is a string
#      Dec2hex: h = hex(d)

# How to read png

# Using Scipy
# http://www.scipy-lectures.org/advanced/image_processing/

from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from os import path
import tensorflow as tf


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
        return k/2 +1
    else:
        return 0

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

iDir = 'ChineseGlyphs'

imgs, labels = read_imgs(iDir)
labs = toBushou(labels)

# [TODO] Input check must be earlier

szImg = imgs[0].shape[0]
# [TODO] Including Label-0  might be unwise (imbalanced dataset)
assert(imgs[0].shape[0] == imgs[0].shape[1])    # check square input

# np.concat > shuffle
# imgs = np.asarray(imgs)
# labels = np.asarray(labels)
x = zip(imgs, labs)
np.random.shuffle(x)
imgs = [i for i, l in x]    # np.asarray
labs = [l for i, l in x]

x = None
imgs = np.reshape(imgs, (-1, szImg, szImg, 1))
labs = np.asarray(labs)


# batch generator

# random start filters
# relu
# max-pooling (tf.nn.max_pool)
# 3 layers
# szImg = 100
szBatch = 16
szPatch1 = 11
szPatch2 = 5
szStride1 = szPatch1/2
szStride2 = szPatch2/2
# szDownSmp = 2
ksize = 3
kstrd = 2
nChannel = 1
nFilter1 = 16
nFilter2 = 192
nHidden1 = 394
nHidden2 = 192
nClass = 11



graph = tf.Graph()
with graph.as_default():
    lr = tf.placeholder(tf.float32)
    Xi = tf.placeholder(
        tf.float32, 
        shape=[szBatch, szImg, szImg, nChannel])
    Yi = tf.placeholder(tf.int64, shape=[szBatch])
    # Conv 1
    W1 = tf.Variable(
        tf.truncated_normal(
            shape=[szPatch1, szPatch1, nChannel, nFilter1],
            stddev=1.0/(szPatch1*szPatch1)))
    b1 = tf.Variable(tf.zeros(shape=[nFilter1]))
    c1 = tf.nn.conv2d(Xi, W1, [1, szStride1, szStride1, 1], 
        padding='SAME')
    h1 = tf.nn.bias_add(c1, b1)
    p1 = tf.nn.max_pool(h1,
        ksize=[1, ksize, ksize, 1],
        strides=[1, kstrd, kstrd, 1],
        padding='VALID')
    # Conv 2
    W2 = tf.Variable(
        tf.truncated_normal(
            shape=[szPatch2, szPatch2, nFilter1, nFilter2],
            stddev=1.0/(szPatch2*szPatch2)))
    b2 = tf.Variable(tf.zeros(shape=[nFilter2]))
    c2 = tf.nn.conv2d(p1, W2, [1, szStride2, szStride2, 1], 
        padding='SAME')
    h2 = tf.nn.bias_add(c2, b2)
    p2 = tf.nn.max_pool(h2,
        ksize=[1, ksize, ksize, 1],
        strides=[1, kstrd, kstrd, 1],
        padding='VALID')
    dim = 1
    for d in p2.get_shape()[1:].as_list():
        dim *= d
    Xf = tf.reshape(p2, [szBatch, dim])
    Wf1 = tf.Variable(
        tf.truncated_normal(
            shape=[dim, nHidden1],
            stddev=1.0/dim))
    bf1 = tf.Variable(tf.zeros([nHidden1]))
    hf1 = tf.nn.relu(tf.matmul(Xf, Wf1) + bf1)
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
    logits = tf.matmul(hf2, Wf3) + bf3
    Yo = tf.nn.softmax(logits)
    simi_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, Yi)
    loss = tf.reduce_mean(simi_loss)
    train = tf.train.AdagradOptimizer(lr).minimize(loss)


# ==== Test ====
batch_x = imgs[:szBatch]
batch_y = labs[:szBatch]

session = tf.InteractiveSession(graph=graph, config=session_conf)
init = tf.initialize_all_variables()
session.run(init)

feed_dict = {Xi: batch_x, Yi: batch_y, lr: 1e-3}
_, ls = session.run([train, loss], feed_dict=feed_dict)


# ==== Test ====


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