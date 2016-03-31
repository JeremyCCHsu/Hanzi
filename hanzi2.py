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


def onehot(labels, K):
    lab = np.zeros((labels.shape[0], K))
    # lab[:, labels] = 1.0
    for i in range(lab.shape[0]):
        j = labels[i] -1    
        lab[i, j] = 1.0
    return lab



def generate_batch(imgs, laby):
    N = imgs.shape[0]



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

# idx = all(labs > 0)
labs = np.array(labs)
idx = labs > 0

# imgs = imgs[idx]

img1 = list()
for i in range(len(imgs)):
    if idx[i]:
        img1.append(imgs[i])

imgs = img1
labs = labs[idx]

szTrn = 4000

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
lab_y_trn = onehot(lab_trn, 10)     # [Makeshift]
lab_y_vld = onehot(lab_vld, 10)     # [Makeshift]



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
nClass = 10



graph = tf.Graph()
with graph.as_default():
    a2 = 1e-3
    lr = tf.placeholder(tf.float32)
    Xi = tf.placeholder(
        tf.float32, 
        shape=[None, szImg, szImg, nChannel])
    # Yi = tf.placeholder(tf.int64, shape=[None])
    Yi = tf.placeholder(tf.float32, shape=[None, nClass])
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
    L2_on_w1 = tf.reduce_mean(tf.reduce_sum(tf.square(W1), [0, 1]))
    L1_on_w1 = tf.reduce_mean(tf.reduce_sum(tf.abs(W1), [0, 1]))
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
    L2_on_w2 = tf.reduce_mean(tf.reduce_sum(tf.square(W2), [0, 1]))
    L1_on_w2 = tf.reduce_mean(tf.reduce_sum(tf.abs(W2), [0, 1]))
    dim = 1
    for d in p2.get_shape()[1:].as_list():
        dim *= d
    Xf = tf.reshape(p2, [-1, dim])
    Wf1 = tf.Variable(
        tf.truncated_normal(
            shape=[dim, nHidden1],
            stddev=1.0/dim))
    bf1 = tf.Variable(tf.zeros([nHidden1]))
    hf1 = tf.nn.relu(tf.matmul(Xf, Wf1) + bf1)
    L2_on_wf1 = tf.reduce_mean(tf.reduce_sum(tf.square(Wf1), 0))
    L1_on_wf1 = tf.reduce_mean(tf.reduce_sum(tf.abs(Wf1), 0))
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
    L2_on_wf2 = tf.reduce_mean(tf.reduce_sum(tf.square(Wf2), 0))
    L1_on_wf2 = tf.reduce_mean(tf.reduce_sum(tf.abs(Wf2), 0))
    logits = tf.matmul(hf2, Wf3) + bf3
    Yo = tf.nn.softmax(logits)
    # simi_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits, Yi)
    simi_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, Yi))
    L1_loss = a2*(L1_on_w1 + L1_on_w2 + L1_on_wf1 + L1_on_wf2)
    L2_loss = a2*(L2_on_w1 + L2_on_w2 + L2_on_wf1 + L2_on_wf2)
    loss = simi_loss + L1_loss + L2_loss
    train = tf.train.AdagradOptimizer(lr).minimize(loss)
    # yhat = tf.argmax(Yo)
    # acc = 

# ==== Test ====
# session = tf.InteractiveSession(graph=graph, config=session_conf)

# with tf.Session(graph=graph) as session:
#     session = tf.InteractiveSession()
#     init = tf.initialize_all_variables()
#     session.run(init)

# session = tf.Session(graph=graph)
#session = tf.InteractiveSession(graph=graph)
#init = tf.initialize_all_variables().run()
# session.run(init)

# batch_x = imgs[:szBatch]
# batch_y = laby[:szBatch]

# feed_dict = {Xi: batch_x, Yi: batch_y, lr: 1e-3}
# _, ls = session.run([train, loss], feed_dict=feed_dict)


from scipy.stats import truncnorm

nEpoch = 2000
nSmp = img_trn.shape[0]
nBatch = nSmp // szBatch

# with tf.Session(graph=graph, config=session_conf) as session:
#     tf.initialize_all_variables().run()
#     err_log = list()
#     for epoch in range(nEpoch):
#         # err = 0.0
#         # errSim = 0.0
#         err, errSim, errl1, errl2 = (0.0, 0.0, 0.0, 0.0)
#         for i in range(nBatch):
#             if (i+1)*szBatch > nSmp:
#                 i = (i+1)*szBatch % nSmp
#             index = range(i, i+szBatch)
#             noise = truncnorm.rvs(
#                 a= 0.0, 
#                 b=0.5, 
#                 loc=0, 
#                 scale=0.1, 
#                 size=(szBatch, szImg, szImg, 1))
#             batch_x = img_trn[index] + noise
#             batch_y = lab_y_trn[index]
#             feed_dict = {Xi: batch_x, Yi: batch_y, lr: 1e-2}
#             _, sl, l1, l2, ls = session.run(
#                 [train, simi_loss, L1_loss, L2_loss, loss], 
#                 feed_dict=feed_dict)
#             err += ls
#             errl1 += l1
#             errl2 += l2
#             errSim += np.sum(sl)
#         err /= nBatch
#         errSim /= nBatch
#         errl1 /= nBatch
#         errl2 /= nBatch
#         err_log.append(err)
#         feed_dict = {Xi: img_vld, Yi: lab_y_vld, lr: 1e-2}
#         err_vld, yhat = session.run(
#             [simi_loss, Yo], 
#             feed_dict=feed_dict)
#         print 'Epoch %4d, err trn = %.4e (L1=%.4e, L2=%.4e, Sim=%.4e)' % (
#             epoch, err, errl1, errl2, errSim)
#         print '            err vld = %.4e' % (err_vld / (lab_y_vld.shape[0] // szBatch))

nEpoch = 10
# with tf.Session(graph=graph, config=session_conf) as session:
session = tf.InteractiveSession(graph=graph)    # , config=session_conf
init = tf.initialize_all_variables()
session.run(init)
err_log = list()
for epoch in range(nEpoch):
    # err = 0.0
    # errSim = 0.0
    err, errSim, errl1, errl2 = (0.0, 0.0, 0.0, 0.0)
    for i in range(nBatch):
        if (i+1)*szBatch > nSmp:
            i = (i+1)*szBatch % nSmp
        index = range(i, i+szBatch)
        noise = truncnorm.rvs(
            a= 0.0, 
            b=0.5, 
            loc=0, 
            scale=0.1, 
            size=(szBatch, szImg, szImg, 1))
        batch_x = img_trn[index] + noise
        batch_y = lab_y_trn[index]
        feed_dict = {Xi: batch_x, Yi: batch_y, lr: 1e-2}
        _, sl, l1, l2, ls = session.run(
            [train, simi_loss, L1_loss, L2_loss, loss], 
            feed_dict=feed_dict)
        err += ls
        errl1 += l1
        errl2 += l2
        errSim += np.sum(sl)
    err /= nBatch
    errSim /= nBatch
    errl1 /= nBatch
    errl2 /= nBatch
    err_log.append(err)
    feed_dict = {Xi: img_vld, Yi: lab_y_vld, lr: 1e-2}
    err_vld, yhat = session.run(
        [simi_loss, Yo], 
        feed_dict=feed_dict)
    print 'Epoch %4d, err trn = %.4e (L1=%.4e, L2=%.4e, Sim=%.4e)' % (
        epoch, err, errl1, errl2, errSim)
    print '            err vld = %.4e' % (err_vld / (lab_y_vld.shape[0] // szBatch))


# ==== Test ====

fir1, fir2 = session.run([W1, W2])

plt.figure()
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(fir1[:,:,0,i])

plt.savefig('W1.png')

plt.figure()
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(fir2[:,:,0,i])

plt.savefig('W2.png')

map1, map2 = session.run([h1, h2], feed_dict=feed_dict)

plt.figure()
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(map1[0,:,:,i])

plt.savefig('M1.png')
# plt.show()

plt.figure()
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(map2[0,:,:,i])

plt.savefig('M2.png')
# plt.show()


plt.figure()
plt.plot(err_log[:200])
plt.show()

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
