'''
Theory:
  Auto-Encoding Variational Bayes 
  (Kingma, 2014)
Implementation:
  https://jmetzen.github.io/2015-11-27/vae.html 
  (Jan Hendrik Metzen, 2015)
Note:
  1. No MCMC or other complex sampling methods required.


As for initialization, please refer to these 2 threads:
  https://www.reddit.com/r/MachineLearning/comments/29ctf7/how_to_initialize_rectifier_linear_units_relu
  http://deepdish.io/2015/02/24/network-initialization/

Fast conclusion:
  1. ReLU doesn't suffer from initialization, but the subsequent layers do!
  2. The reference of Link 2 is crucial!!!

Issues for Hanzi:
  1. Reconstruction is poor (though Bushous are well-distributed 
     in the latent space)
  2. Bushous 'mouth' and 'girl' are ruined.
  3. While reonstruction, bushous are correct, but the main part
     are all distorted.
'''

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
# %matplotlib inline
from scipy import misc
from os import path

np.random.seed(0)
tf.set_random_seed(0)

# import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# n_samples = mnist.train.num_examples

# from hanzi2_1 import read_imgs


class IBatchGenerator(object):
    def __init__(self):
        raise ValueError('Abstract class is not initializable')
    def __iter__(self):
        pass
    def next(self):
        pass

class XBatchGenerator(IBatchGenerator):
    def __init__(self, X, szBatch):
        '''
        '''
        self.X = X
        self.szTrn = X.shape[0]
        self.szBatch = szBatch
        self.nBatches = self.szTrn // self.szBatch
        self.cursor = 0
    def __iter__(self):
        '''
        [TODO] Here, we should add a random shuffle
        [TODO] In current implementation, the number of batches is unstable
        '''
        self.cursor = (self.cursor + self.szBatch) % self.szTrn
        return self
    def next(self):
        index = self._batch_index()
        xi = self.X[index, :]
        return xi
    def _batch_index(self):
        if self.cursor + self.szBatch > self.szTrn:
            # cursor = (cursor + szBatch) % szTrn
            raise StopIteration
        else:
            index = np.array(
                range(self.cursor, self.cursor+self.szBatch))
            self.cursor += self.szBatch
        return index
    def get_num_of_batches(self):
        return self.nBatches


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



def read(iDir):
    ''' Ad-hoc (not randomized) '''
    imgs, labels = read_imgs(iDir)
    labs = toBushou(labels)
    # [TODO] Input check must be earlier
    szImg = imgs[0].shape[0]
    # [TODO] Including Label-0  might be unwise (imbalanced dataset)
    assert(imgs[0].shape[0] == imgs[0].shape[1])    # check square input
    labs = np.array(labs)
    idx = labs < 10
    # imgs = imgs[idx]
    img1 = list()
    for i in range(len(imgs)):
        if idx[i]:
            img1.append(imgs[i])
    imgs = img1
    labs = labs[idx]
    imgs = np.asarray(imgs)
    imgs = np.reshape(imgs, (-1, imgs.shape[1]*imgs.shape[2]))
    return imgs


def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

class VariationalAutoencoder(object):
    """ 
    Variation Autoencoder (VAE) with an sklearn-like interface 
    implemented using TensorFlow.
    
    This implementation uses probabilistic encoders and decoders 
    using Gaussian distributions and  realized by multi-layer perceptrons. 
    The VAE can be learned end-to-end.
    
    See "Auto-Encoding Variational Bayes" by Kingma and Welling 
    for more details.
    """
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, 
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # tf Graph input
        self.x = tf.placeholder(
            tf.float32, 
            shape=[None, network_architecture["n_input"]])
        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer()
        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()
        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)
        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"], 
                                      network_weights["biases_recog"])
        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, 
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, 
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])
    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, 
                            n_hidden_gener_1,  n_hidden_gener_2, 
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights      
    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)
    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), 
                                 biases['out_mean']))
        return x_reconstr_mean
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluatio of log(0.0)
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                           + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                           1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        # J: please refer to Equation 10 and Appendix B in the paper. 
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                           - tf.square(self.z_mean) 
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
    def partial_fit(self, X):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost), 
                                  feed_dict={self.x: X})
        return cost
    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.z: z_mu})
    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.x: X})




# 
iDir = 'TWKai98_50x50'
imgs = read(iDir)   # Top 10 Bushou
# imgs, labels = read_imgs(iDir)
# imgs = np.reshape(imgs, (-1, imgs.shape[1]*imgs.shape[2]))

def train(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=5):
    vae = VariationalAutoencoder(network_architecture, 
                                 learning_rate=learning_rate, 
                                 batch_size=batch_size)
    # Training cycle
    n_samples = imgs.shape[0]
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        # for i in range(total_batch):
        #     batch_xs, _ = mnist.train.next_batch(batch_size)
        np.random.shuffle(imgs)
        batches = XBatchGenerator(imgs, batch_size)
        for batch_xs in batches:
            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), \
                  "cost=", "{:.9f}".format(avg_cost)
    return vae



# # Test 1
# network_architecture = \
#     dict(n_hidden_recog_1=500, # 1st layer encoder neurons
#          n_hidden_recog_2=500, # 2nd layer encoder neurons
#          n_hidden_gener_1=500, # 1st layer decoder neurons
#          n_hidden_gener_2=500, # 2nd layer decoder neurons
#          n_input=784, # MNIST data input (img shape: 28*28)
#          n_z=20)  # dimensionality of latent space

# vae = train(network_architecture, training_epochs=75)


# x_sample = mnist.test.next_batch(100)[0]
# x_reconstruct = vae.reconstruct(x_sample)

# plt.figure(figsize=(8, 12))
# for i in range(5):

#     plt.subplot(5, 2, 2*i + 1)
#     plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1)
#     plt.title("Test input")
#     plt.colorbar()
#     plt.subplot(5, 2, 2*i + 2)
#     plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1)
#     plt.title("Reconstruction")
#     plt.colorbar()
# plt.tight_layout()



# Test 2
network_architecture = \
    dict(n_hidden_recog_1=500, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_input=2500, # MNIST data input (img shape: 28*28)
         n_z=2)  # dimensionality of latent space

vae_2d = train(
    network_architecture, 
    batch_size=128,
    learning_rate=1e-4,
    training_epochs=76)


# x_sample, y_sample = mnist.test.next_batch(5000)
# z_mu = vae_2d.transform(x_sample)
# plt.figure(figsize=(8, 6)) 
# plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
# plt.colorbar()


# 
nx = ny = 25
x_values = np.linspace(-7, 7, nx)
y_values = np.linspace(-7, 7, ny)
szImg = 50

canvas = np.empty((szImg*ny, szImg*nx))
for i, yi in enumerate(x_values):
    for j, xi in enumerate(y_values):
        z_mu = np.array([[xi, yi]])
        x_mean = vae_2d.generate(z_mu)
        canvas[(nx-i-1)*szImg:(nx-i)*szImg, j*szImg:(j+1)*szImg] = x_mean[0].reshape(szImg, szImg)

plt.figure(figsize=(32, 40))        
Xi, Yi = np.meshgrid(x_values, y_values)
plt.imshow(canvas, origin="upper")
plt.tight_layout()
plt.savefig('vae.png')
plt.close()




idx1 = las > 20200
idx2 = las < (20200+128+1)
idxa = idx1 * idx2
test = np.asarray(ims)[idxa]

n = 12
# x_reconstruct = vae_2d.reconstruct(imgs[:128, :])
x_reconstruct = vae_2d.reconstruct(test.reshape(128, 2500))
x_reconstruct = np.reshape(x_reconstruct, (128, szImg, szImg))
canvas = np.empty((szImg*12, szImg*12))
for i in range(12):
    for j in range(12):
        canvas[(n-i-1)*szImg:(n-i)*szImg, j*szImg:(j+1)*szImg] = \
            x_reconstruct[i*8 +j, :, :]

plt.figure(figsize=(32, 40))        
plt.imshow(canvas, origin="upper")
plt.tight_layout()
plt.savefig('vae_reconst.png')
plt.close()




