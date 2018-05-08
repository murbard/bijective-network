import tensorflow as tf
import numpy as np
import scipy
from matplotlib import pyplot as plt
from tensorflow.python import debug as tf_debug

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
        self.snorm = tf.distributions.Normal(loc=0.0, scale=1.0)
        # for the xᵢ
        self.gnorm = tf.distributions.Normal(loc=0.0, scale=0.5)
        # for p
        self.beta = tf.distributions.Beta(0.8, 0.8)

    # sample the hyper parameters and n values of xᵢ
    def sample(self, n):
        µ = self.snorm.sample()
        p = self.beta.sample()
        c = tf.cast(tf.distributions.Bernoulli(p).sample(n), tf.float32)
        x = c * µ + (1.0 - c) * (-µ) + self.gnorm.sample(n)
        return x

    # log probability of z = [µ, p] *and* a vector of xᵢ
    def log_prob(self, z, x):
        res   = self.snorm.log_prob(z[0])
        print("a", res.shape)
        res  += self.beta.log_prob(z[1])
        print("b", res.shape)

        aa = self.gnorm.log_prob(tf.expand_dims(x,1) - tf.expand_dims(z[0],0)) + tf.log(z[1])
        print("c", aa.shape)
        bb = self.gnorm.log_prob(tf.expand_dims(x,1) + tf.expand_dims(z[0],0)) + tf.log(1.0 - z[1])
        print("d", bb.shape)
        cc =  tf.reduce_logsumexp([ aa , bb ], axis=0)
        print("e", cc.shape)

        res  += tf.reduce_sum(
            # use reduce_logsumexp to avoid numerical instability
           cc, axis=0)
        print("f",res.shape)
        return res



    # compute the posterior distribution of [µ, p] given x by
    # straight Riemann integration
    def posterior(self, s):
        post = np.zeros((100,100))
        Z = 0.0
        with tf.Session() as sess:
            µtf = tf.Variable(0.0,trainable=False)
            ptf = tf.Variable(0.0,trainable=False)
            llop = self.log_prob(tf.transpose(tf.stack([µtf,ptf])), s)

            for i,µ in enumerate(np.linspace(-3, 3, 101, endpoint=False, dtype=np.float32)[1:]):
                for j,p in enumerate(np.linspace(0, 1, 101, endpoint=False, dtype=np.float32)[1:]):
                    ll  = sess.run(llop, feed_dict={µtf:µ, ptf:p})
                    if np.isnan(ll):
                        ll = -np.inf
                    post[i,j] = ll

            # Integrate and normalize
            lZ = scipy.misc.logsumexp(post)
            post = np.exp(post - lZ)
        return post

class Network(object):

    def __init__(self, dim = 2, n_layers = 7, activation = 'sinh' , relu_slope=0.9):
        self.dim = dim
        self.n_layers = n_layers
        self.layers = [
            {
#                'w': tf.Variable(self.rand_rot()) ,
                'w':tf.Variable(tf.random_normal([dim, dim]) * 0.01 + tf.eye(dim) * 1.0, name='w%d' % i) ,
                'b': tf.Variable(tf.random_normal([dim, 1]) * 0.1, name='b%d' % i)
            } for i in range(0, n_layers)
        ]
        self.activation = activation
        self.relu_slope = relu_slope
        self.magic_matrix = np.eye(self.dim, dtype=np.float32)
        self.magic_matrix[0, 0] = 3.0
        self.magic_matrix[1, 1] = 0.5
        self.magic_vector = np.zeros((self.dim, 1), dtype=np.float32)
        self.magic_vector[1, 0] = 0.5
        self.inv_magic_matrix = np.linalg.inv(self.magic_matrix)


    def rand_rot(self):
        m = np.eye(self.dim)
        for i in range(0,self.dim):
            for j in range(i+1,self.dim):
                u = m[i,:]
                v = m[j,:]
                theta = np.random.rand() * 2.0 * np.pi
                m[i,:] = np.cos(theta) * u - np.sin(theta) * v
                m[j,:] = np.sin(theta) * u + np.cos(theta) * v
        return m.astype(np.float32)


    def to_output(self, y):
        return tf.matmul(self.magic_matrix, y) + self.magic_vector

    def of_output(self, o):
        return tf.matmul(self.inv_magic_matrix, o - self.magic_vector)



    def f(self, input):
        log_jacobian = tf.zeros(input.shape[1])

        # Initial expension
        log_jacobian += tf.reduce_sum(-tf.log(1.0 - input**2), axis=0)
        y = tf.atanh(input)

        # Hidden layers
        for i,layer in enumerate(self.layers[:-1]):
            log_jacobian += tf.linalg.logdet(layer['w'])
            y = tf.matmul(layer['w'], y) + layer['b']

            if self.activation == 'sinh':
                if i % 2 == 0:
                    y = tf.sinh(y)
                    log_jacobian += tf.reduce_sum(0.5 * tf.log(1.0 + y**2), axis=0)
                else:
                    log_jacobian += tf.reduce_sum(-0.5 * tf.log(1.0 + y**2), axis=0)
                    y = tf.asinh(y)

            elif self.activation == 'relu':
                log_jacobian += tf.reduce_sum(tf.log(0.5*(self.relu_slope + 1.0) + 0.5*(1.0-self.relu_slope) * tf.sign(y)), axis=0)
                y = tf.nn.leaky_relu(y, alpha=self.relu_slope)
            else:
                raise "unknown activation function"

        # Final layer, contract
        log_jacobian += tf.linalg.logdet(self.layers[-1]['w'])
        y = tf.tanh(tf.matmul(self.layers[-1]['w'], y) + self.layers[-1]['b'])
        log_jacobian += tf.reduce_sum(tf.log(1-y**2), axis=0)

        # Adapt y to match the scale of the hyperparameters
        y = self.to_output(y)

        return (y, log_jacobian)

    def inverse_f(self, output, everything=False):
        log_jacobian = tf.zeros(output.shape[1])

        # Normalize
        x = self.of_output(output)

        if (everything):
            everything = [x]

        # undo contraction
        log_jacobian += tf.reduce_sum(tf.log(1.0 - x**2), axis=0)
        x = tf.atanh(x)
        if (everything):
            everything.append(x)

        log_jacobian += tf.linalg.logdet(self.layers[-1]['w'])


        x = tf.linalg.solve(self.layers[-1]['w'], x - self.layers[-1]['b'])
#        x = tf.matmul(tf.linalg.inv(self.layers[-1]['w']), x - self.layers[-1]['b'])
        if (everything):
            everything.append(x)

        # hidden layers
        for i,layer in enumerate(self.layers[:-1][-1::-1]):

            if self.activation == 'sinh':
                if i % 2 == 0:
                    log_jacobian += tf.reduce_sum(0.5 * tf.log(1.0 + x**2), axis=0)
                    x = tf.asinh(x)
                    if (everything):
                        everything.append(x)
                else:
                    x = tf.sinh(x)
                    log_jacobian += tf.reduce_sum(-0.5 * tf.log(1.0 + x**2), axis=0)
                    if (everything):
                        everything.append(x)


            elif self.activation == 'relu':

                x = -tf.nn.leaky_relu(- x / self.relu_slope, alpha = self.relu_slope)
                if (everything):
                    everything.append(x)
                log_jacobian += tf.reduce_sum(tf.log(0.5*(self.relu_slope + 1.0) + 0.5*(1.0-self.relu_slope) * tf.sign(x)), axis=0)

            else:
                raise "unknown activation"

            log_jacobian += tf.linalg.logdet(layer['w'])
            x = tf.linalg.solve(layer['w'], x - layer['b'])
#           x = tf.matmul(tf.linalg.inv(layer['w']), x - layer['b'])
            if (everything):
                everything.append(x)

        # First layer contraction
        x = tf.tanh(x)
        if (everything):
            everything.append(x)
        log_jacobian += tf.reduce_sum(-tf.log(1.0 - x**2), axis=0)

        if everything:
            return everything
        else:
            return (x, -log_jacobian)

    def log_det(self):
        return [ tf.linalg.det(tf.linalg.inv(l['w'])) for l in self.layers ]

    def posterior(self):

        outputs = []
        W = 25
        for i,µ in enumerate(np.linspace(-3.0, 3.0, 101, endpoint=False, dtype=np.float32)[1:]):
            for j,p in enumerate(np.linspace(0.0, 1.0, 101, endpoint=False, dtype=np.float32)[1:]):
                for w in range(0, W):
                    out = 2.0 * np.random.rand(self.dim) - 1.0
                    out[0] = µ
                    out[1] = p
                    out = out.astype(np.float32)
                    outputs.append(out)

        _, log_jacobian = self.inverse_f(tf.transpose(tf.stack(outputs)))

                    # # Checks
                    # #

                    # try:
                    #     pass
                    # #assert( np.abs(µ - check_y[0]) < 1e-3)
                    # #assert( np.abs(p - check_y[1]) < 1e-3)
                    # #assert( np.abs(log_jacobian + check_log_jacobian) < 1e-3)
                    # except Exception as e:
                    #     print(np.array([µ, p, log_jacobian]), np.array([check_y[0], check_y[1], -check_log_jacobian]))
                    #     print(x)
                    #     weights = sess.run([l['w'] for l in net.layers])
                    #     biases = sess.run([l['b'] for l in net.layers])
                    #     for i, w in enumerate(weights):
                    #         print(('w%d' % i) + ' = {%s};' % ','.join(map(lambda row: '{%s}' % ','.join(map(lambda el: "%lf" % el, row)), w)))
                    #         for i, b in enumerate(biases):
                    #             print(('b%d' % i) + ' = {%s};' % ','.join(map(lambda el: '{%lf}' % el, b)))
                    #             print("Everything:")
                    #             ev = sess.run(self.inverse_f(np.array([µ,p]),everything=True))
                    #             for x in ev:
                    #                 print(x)

                    #                 raise e
                    # remove nan


        log_jacobian = tf.reshape(log_jacobian, (100, 100, W))
        log_jacobian = tf.where(tf.is_nan(log_jacobian), np.zeros((100, 100, W), dtype=np.float32) - np.inf, log_jacobian)

        posterior = tf.reduce_logsumexp(log_jacobian, axis=2) - np.log(W + 0.0)
        posterior = tf.where(tf.is_nan(posterior), np.zeros((100,100), dtype=np.float32) - np.inf, posterior)

        Z = tf.reduce_logsumexp(posterior)
        posterior = tf.exp(posterior - Z)
        return posterior

    # def check(self, which):
    #     total_j = 0.0
    #     total_j2 = 0.0
    #     for i in range(0,100):
    #         y = tf.distributions.Uniform(low=-1.0, high=1.0).sample((self.dim,1))

    #         y = self.to_output(y)

    #         x, log_jacobian = self.inverse_f(y)
    #         j = tf.exp(log_jacobian)
    #         total_j += j
    #         total_j2 += j*j

    #     total_j /= 100.0
    #     total_j2 /= 100.0
    #     std = tf.sqrt((total_j2 - total_j * total_j) / 100.0)
    #     return (total_j - 2.0 * std, total_j, total_j + 2.0 * std)



    def sample(self, n):

        p = 0.8
        n1, n2 = round(p * n), n - round(p * n)
        p = (n1 + 0.0) / n

        # sample from network
        x = tf.distributions.Uniform(low=-1.0, high=1.0).sample((self.dim, n1))
        y_Q, log_jacobian_Q = self.f(x)

        # sample uniformly
        y_U = tf.distributions.Uniform(low=-1.0, high=1.0).sample((self.dim, n2))
        y_U = self.to_output(y_U)
        _, log_jacobian_U = self.inverse_f(y_U)

        log_prob = tf.concat([- log_jacobian_Q, log_jacobian_U], axis=0)
        y = tf.concat([y_Q, y_U], axis=1)

#        weight = tf.reduce_logsumexp(tf.stack([np.log(p) + log_prob, np.log(1.001-p) * tf.ones(n)]), axis=0)
        weight = log_prob - tf.reduce_logsumexp(tf.stack([np.log(p) + log_prob, np.log(1.0-p) * tf.ones(n) ]), axis=0)
        return (y, log_prob, weight)

class VB(object):
    def __init__(self, model, net):
        self.model = model
        self.net = net

    def stochastic_score(self, x, n):
        z, log_probability, log_weight= self.net.sample(n)

        dl = (self.model.log_prob(z,x) - log_probability)


        dl = tf.where(tf.is_nan(dl), -np.inf * tf.ones(n), dl)
        log_weight = tf.where(tf.is_nan(log_weight), -np.inf * tf.ones(n), log_weight)

        #factor = tf.exp(dl + log_weight)
#        return tf.reduce_sum(factor * dl)
        weighted = -dl * tf.exp(log_weight)
        return tf.reduce_sum(tf.where(tf.is_nan(weighted), tf.zeros(n), weighted)) #+ n * net.log_det2()


def print_mat(w, name):
    print('%s={%s};' % (name, ','.join(map(lambda row: '{%s}' % ','.join(map(lambda el: "%lf" % el, row)), w))))


if __name__ == '__main__':

    model = Model()
    net = Network(dim=3, n_layers=9)
    vb = VB(model, net)


    print('{%s};' % ','.join(map(lambda row: '{%s}' % ','.join(map(lambda el: "%lf" % el, row)), net.rand_rot())))
    x = model.sample(23).eval(session=tf.Session())

    plt.ion()
    plt.show()
    plt.imshow(model.posterior(x))
    plt.draw()
    plt.pause(0.1)

    score_op = vb.stochastic_score(x, 20)
    optimizer = tf.train.MomentumOptimizer(learning_rate=1e-7, momentum=0.1)
    grads_and_vars = optimizer.compute_gradients(score_op, gate_gradients=tf.train.Optimizer.GATE_GRAPH)

    modified_gradients = [(tf.where(tf.is_nan(gv[0]), tf.zeros(gv[0]._shape_tuple()), gv[0]), gv[1]) for gv in grads_and_vars]
    minimize = optimizer.apply_gradients(modified_gradients)

    posterior = net.posterior()
#    log_det = net.log_det()

#    g = tf.norm(tf.gradients(score_op, [l['w'] for l in net.layers]), axis=[1,2])

    with tf.Session() as sess:
#        sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        sess.run(tf.global_variables_initializer())
        score_ema = None

        print_every = 100
        i = 0

#        print(sess.run(log_det))
        while True:
           # print(sess.run(log_det))
           # print(sess.run(g))


#            for g,v in grads_and_vars:
 #               print_mat(sess.run(g),g.name)

            try:
                score = sess.run(score_op)
                sess.run(minimize)
            except:
                for i,w in enumerate(sess.run([l['w'] for l in net.layers])):
                    print_mat(w, 'w%d' % i)
                raise "womp"

            if not score_ema:
                score_ema = score
            else:
                if not np.isnan(score):
                    score_ema = 0.9 * score_ema + 0.1 * score

            i += 1
            print(i)



            if (i + 1) % print_every == 0:
#                check = sess.run(check_op)
#                print("check",check)

                print(score, score_ema)
#                print(sess.run([l['w'] for l in net.layers]))
                post = sess.run(posterior)
#                print(post)
                plt.imshow(post)
                plt.draw()
                plt.pause(0.1)
