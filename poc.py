import tensorflow as tf
import numpy as np
import scipy
from matplotlib import pyplot as plt
from tensorflow.python import debug as tf_debug


def cartesian_product(arrays):
    """
    Generalized N-dimensional products
    :param arrays: a list of arrays 
    :return: an array of arrays representing the cartesian product of the input
    """
    la = len(arrays)
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


class Model(object):
    """
    Model:
    µ ← N(0,1)
    p ← B(1,1)
    ∀ i ∈ 0…n,
      xᵢ ← N(µ,0.5) with probability p,
      xᵢ ← N(-µ,0.5) with probability (1-p)
    """

    def __init__(self):
        # Initialize distribution objects used to describe
        # hyper-parameter priors.

        # N(0,1) for mu
        self.norm_µ = tf.distributions.Normal(loc=0.0, scale=1.0)
        # for the xᵢ
        self.norm_x = tf.distributions.Normal(loc=0.0, scale=0.5)
        # for p
        self.beta = tf.distributions.Beta(0.8, 0.8)

    # Sample the hyper parameters and n values of xᵢ.
    def sample(self, n):
        µ = self.norm_µ.sample()
        p = self.beta.sample()
        c = tf.cast(tf.distributions.Bernoulli(p).sample(n), tf.float32)
        x = c * µ + (1.0 - c) * (-µ) + self.norm_x.sample(n)
        return x

    def log_prob(self, z, x):
        """
        Log probability of a batch of hyperparameters z = [µ, p] *and*
        a batch of vectors x. Note: differentiable with respect to z.

        :param z: batch array of length 2 array containing µ and p
        :param x: sample
        :return:
        """

        # Start with the log prior.
        res = self.norm_µ.log_prob(z[0])
        res += self.beta.log_prob(z[1])

        # Log probability if drawn from the gaussian centered on µ.
        log_prob_a = self.norm_x.log_prob(
            tf.expand_dims(x, 1) - tf.expand_dims(z[0], 0)
        ) + tf.log(z[1])
        # Log probability if drawn from the gaussian centered on -µ.
        log_prob_b = self.norm_x.log_prob(
            tf.expand_dims(x, 1) + tf.expand_dims(z[0], 0)
        ) + tf.log(1.0 - z[1])

        # Probability of mixture.
        log_prob = tf.reduce_logsumexp([log_prob_a, log_prob_b], axis=0)

        res += tf.reduce_sum(log_prob, axis=0)
        return res

    def posterior(self, x, resolution=100):
        """
        Compute the posterior distribution of [µ, p] given x by
        straight Riemann integration. Used only for making pretty
        images.
        :param resolution: number of pixels in the x and y direction
        :return: an image representing the posterior of the latent
        given the sample
        """
        image = np.zeros((resolution, resolution))
        µ = np.linspace(-3, 3, resolution + 2, dtype=np.float32)[1:-1]
        p = np.linspace(0, 1, resolution + 2, dtype=np.float32)[1:-1]
        z = np.transpose(cartesian_product([µ, p]))

        ll = self.log_prob(z, x).eval(session=tf.Session())

        for i, a in enumerate(ll):
            image[i // resolution,
                  i % resolution] = -np.inf if np.isnan(a) else a

        # Integrate and normalize
        lZ = scipy.misc.logsumexp(image)
        return np.exp(image - lZ)


class Network(object):
    """ A nijective neural network """

    def __init__(self, dim=2, n_layers=7,
                 activation='sinh', relu_slope=0.9):
        """
        :param dim: Dimension of the problem, every layer has dimension dim
        :param n_layers: number of layers
        :param activation: activation function, can be 'sinh' or 'relu'
        :param relu_slope: slope of the leaky relu, if activation='relu' is used
        """
        self.dim = dim
        self.n_layers = n_layers

        # Randonly initialize layers.
        self.layers = [
            {
                # For lack of a better technique, initialize matrices close to the identity.
                'w': tf.Variable(
                    tf.random_normal([dim, dim]) * 0.01 + tf.eye(dim) * 0.99,
                    name='w%d' % i),
                'b': tf.Variable(
                    tf.random_normal([dim, 1]) * 0.1, name='b%d' % i)
            } for i in range(0, n_layers)]
        self.activation = activation
        self.relu_slope = relu_slope

        # The scaling matrix is used to map a [-1,1]^dim hypercube to the
        # dimensions expected in the problem.
        self.scaling_matrix = np.eye(self.dim, dtype=np.float32)
        # µ is only sampled between -4 and 4
        self.scaling_matrix[0, 0] = 4.0
        self.scaling_matrix[1, 1] = 0.5

        # The offset applied to the [-1,1]^dim output, after scaling
        # is applied
        self.offset_vector = np.zeros((self.dim, 1), dtype=np.float32)
        self.offset_vector[1, 0] = 0.5
        self.inv_scaling_matrix = np.linalg.inv(self.scaling_matrix)
        self.error_unknown_activation_function = Exception("unknown activation function")

    def rand_rot(self):
        """
        Generate a random rotation matrix, possible initialization method.
        """
        # The approach is to start with an identity matrix, and then to randomly
        # rotate around n(n-1)/2 axes.
        m = np.eye(self.dim)
        for i in range(0, self.dim):
            for j in range(i + 1, self.dim):
                u, v = m[i, :], m[j, :]
                theta = np.random.rand() * 2.0 * np.pi
                m[i, :] = np.cos(theta) * u - np.sin(theta) * v
                m[j, :] = np.sin(theta) * u + np.cos(theta) * v
        return m.astype(np.float32)

    def to_model(self, x):
        """
        Converts an input of the neural network to a value
        that corresponds to the latent parameters of the model by applying the scaling
        matrix and the offset vector.
        :param y: output of the neural network
        :return: output as expected in the statistical model
        """
        return tf.matmul(self.scaling_matrix, x) + self.offset_vector

    def of_model(self, z):
        """
        Converts latent parameters from the model into a corresponding input for the neural network.
        :param o:
        :return:
        """
        return tf.matmul(self.inv_scaling_matrix, z - self.offset_vector)

    def f(self, z):
        """
        Computes a forward pass in the network
        :param z: latent parameters of the model
        :return: a mapping in [-1,1]^dim
        """

        # Initialize the log Jacobian at 0. It is updated by repeatedly adding to it.
        log_Jacobian = tf.zeros(z.shape[1])

        input = self.of_model(z)

        # Initial expansion
        log_Jacobian += tf.reduce_sum(-tf.log(1.0 - input ** 2), axis=0)
        y = tf.atanh(input)

        # Hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            log_Jacobian += tf.linalg.logdet(layer['w'])
            y = tf.matmul(layer['w'], y) + layer['b']

            if self.activation == 'sinh':
                # Alternate sinh and asinh for gradient stability.
                if i % 2 == 0:
                    y = tf.asinh(y)
                    log_Jacobian += tf.reduce_sum(0.5 * tf.log(1.0 + y ** 2), axis=0)
                else:
                    log_Jacobian += tf.reduce_sum(-0.5 * tf.log(1.0 + y ** 2), axis=0)
                    y = tf.sinh(y)

            elif self.activation == 'relu':
                log_Jacobian += tf.reduce_sum(
                    tf.log(0.5 * (self.relu_slope + 1.0) + 0.5 * (1.0 - self.relu_slope) * tf.sign(y)), axis=0)
                y = tf.nn.leaky_relu(y, alpha=self.relu_slope)
            else:
                raise self.error_unknown_activation_function

        # Final layer, contract
        log_Jacobian += tf.linalg.logdet(self.layers[-1]['w'])
        y = tf.tanh(tf.matmul(self.layers[-1]['w'], y) + self.layers[-1]['b'])
        log_Jacobian += tf.reduce_sum(tf.log(1 - y ** 2), axis=0)

        return y, log_Jacobian

    def inverse_f(self, output):
        """
        Backwards pass through the neural network
        :param output: a point in the [-1, 1]^dim hypercube
        :return: the corresponding latent parameters
        """
        log_Jacobian = tf.zeros(output.shape[1])

        u = output
        # undo contraction
        log_Jacobian -= tf.reduce_sum(tf.log(1.0 - u ** 2), axis=0)
        u = tf.atanh(u)

        log_Jacobian -= tf.linalg.logdet(self.layers[-1]['w'])

        u = tf.linalg.solve(self.layers[-1]['w'], u - self.layers[-1]['b'])

        # hidden layers
        for i, layer in enumerate(self.layers[:-1][-1::-1]):

            if self.activation == 'sinh':
                if i % 2 == 0:
                    log_Jacobian -= tf.reduce_sum(0.5 * tf.log(1.0 + u ** 2), axis=0)
                    u = tf.sinh(u)
                else:
                    u = tf.asinh(u)
                    log_Jacobian -= tf.reduce_sum(-0.5 * tf.log(1.0 + u ** 2), axis=0)

            elif self.activation == 'relu':

                u = -tf.nn.leaky_relu(- u / self.relu_slope, alpha=self.relu_slope)
                log_Jacobian -= tf.reduce_sum(
                    tf.log(0.5 * (self.relu_slope + 1.0) + 0.5 * (1.0 - self.relu_slope) * tf.sign(u)), axis=0)

            else:
                raise self.error_unknown_activation_function

            log_Jacobian -= tf.linalg.logdet(layer['w'])
            u = tf.linalg.solve(layer['w'], u - layer['b'])

        # First layer contraction
        u = tf.tanh(u)
        log_Jacobian -= tf.reduce_sum(-tf.log(1.0 - u ** 2), axis=0)

        z = self.to_model(u)
        return z, log_Jacobian

    def posterior(self, resolution=100, W=25):
        """
        Draws an image of the posterior of the distribution being modeled
        :param resolution: number of pixels in width and height
        :param W: number of Monte-Carlo samples to take to marginalize extraneous dimensions
        :return: image of the distribution
        """
        outputs = []
        # Stack up all the points we need to evaluate in one big batch.
        for i, µ in enumerate(np.linspace(-3.0, 3.0, resolution + 2, dtype=np.float32)[1:-1]):
            for j, p in enumerate(np.linspace(0.0, 1.0, resolution + 2, dtype=np.float32)[1:-1]):
                for w in range(0, W):
                    out = 2.0 * np.random.rand(self.dim) - 1.0
                    out[0], out[1] = µ, p
                    out = out.astype(np.float32)
                    outputs.append(out)

        _, log_Jacobian = self.f(tf.transpose(tf.stack(outputs)))
        image = tf.reshape(log_Jacobian, (resolution, resolution, W))

        # Remove nans, as the numerical stability of this code leaves a lot to be desired.

        # Replace all nans with -infinity
        image = tf.where(
            tf.is_nan(image),
            np.zeros((resolution, resolution, W), dtype=np.float32) - np.inf,
            image)

        # squash in the y direction
        image = tf.reduce_logsumexp(image, axis=2) - np.log(float(W))

        # Replace nans with -infinity one more time
        image = tf.where(
            tf.is_nan(image),
            np.zeros((resolution, resolution), dtype=np.float32) - np.inf,
            image)

        # For good measure, normalize, though this ought not to be necessary.
        Z = tf.reduce_logsumexp(image)
        image = tf.exp(image - Z)
        return image

    def sample(self, n):
        """
        Sample n draws  the distribution implied by the netwo
        :param n: number of draws
        :return: a sample
        """

        # Sample from network
        u = tf.distributions.Uniform(low=-1.0, high=1.0).sample((self.dim, n))
        z, log_Jacobian = self.inverse_f(u)
        return z, - log_Jacobian

class VB(object):
    """
    Variational Bayes optimization using a bijective network
    """

    def __init__(self, model, net):
        self.model = model
        self.net = net

    def stochastic_score(self, x, n):
        """
        Stochastic estimate of the KL divergence given sample x, averaged over n random input
        :param x:
        :param n:
        :return:
        """
        z, log_probability = self.net.sample(n)

        # log Q(z ) / P(z, x)
        dl = log_probability - self.model.log_prob(z, x)

        # remove nans
        dl = tf.where(tf.is_nan(dl), -np.inf + tf.zeros(n), dl)

        return tf.reduce_sum(dl)


def print_mat(w, name="w"):
    """
    Debug function to print a matrix in mathematica ormat
    :param w: matrix entries
    :param name: name of the matrix
    """
    print('%s={%s};' % (name, ','.join(map(lambda row: '{%s}' % ','.join(map(lambda el: "%lf" % el, row)), w))))


if __name__ == '__main__':

    model = Model()
    net = Network(dim=4, n_layers=9)
    vb = VB(model, net)
    x = model.sample(23).eval(session=tf.Session())

    # Set up matplotlib
    plt.ion()
    plt.show()
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(model.posterior(x))
    plt.draw()
    plt.pause(0.1)


    score_op = vb.stochastic_score(x, 20)

    optimizer = tf.train.MomentumOptimizer(learning_rate=1e-5, momentum=0.1)
    grads_and_vars = optimizer.compute_gradients(score_op, gate_gradients=tf.train.Optimizer.GATE_GRAPH)
    # Remove nans from gradinets
    modified_gradients = [
                (tf.where(tf.is_nan(gv[0]), tf.zeros(gv[0]._shape_tuple()), gv[0]), gv[1]) for gv in
                          grads_and_vars]
    minimize = optimizer.apply_gradients(modified_gradients)

    print("Compiling posterior operation. This can take a *long* time...")
    posterior = net.posterior(W=25)
    print("done")
    # How often to display a new image of the posterior
    print_every = 1

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        score_ema = None
        i = 0

        while True:
            try:
                score = sess.run(score_op)
                sess.run(minimize)
            except:
                for i, w in enumerate(sess.run([l['w'] for l in net.layers])):
                    print_mat(w, 'w%d' % i)
                raise Exception("womp womp")

            if not score_ema:
                score_ema = score
            else:
                if not np.isnan(score):
                    score_ema = 0.9 * score_ema + 0.1 * score

            i += 1
            print(i)

            if (i + 1) % print_every == 0:
                print(score, score_ema)
                post = sess.run(posterior)
                ax2.imshow(post)
                plt.draw()
                plt.pause(0.1)
