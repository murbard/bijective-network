from scipy.stats import norm
import scipy
import numpy as np
#from mnist import MNIST
#from matplotlib import pyplot as plt


class Layer :
    def __init__(self, dim, relu=None):
        self.relu = relu
        self.dim = dim
        self.L = np.eye(dim) # np.tril(np.random.randn(dim, dim) * 0.1,-1) + np.eye(dim)
        self.D = np.ones(dim) # np.random.randn(dim) * 0.01 + 1.0
        self.U = np.eye(dim) # np.triu(np.random.randn(dim, dim) * 0.1 ,1) + np.eye(dim)
        self.b = np.zeros(dim) # np.random.randn(dim) * 0.1

    def apply_gradient(self, epsilon):
        self.L += epsilon * self.dJ_L
        self.D += epsilon * self.dJ_D
        self.U += epsilon * self.dJ_U
        self.b += epsilon * self.dJ_b
        if self.relu:
            self.relu += epsilon * self.dJ_relu

    def grad_mag(self):
        return (
            (self.dJ_relu**2 if self.relu else 0.0) +
            (self.dJ_L**2).sum() +
            (self.dJ_D**2).sum() +
            (self.dJ_U**2).sum() +
            (self.dJ_b**2).sum() +
            0
            )


def forward(image):

    J = 0

    h = image.copy()

    # expand
#    h = norm.ppf(h)
#    J = J + 0.5 * (h**2).sum()

    # transform
    for layer in layers:
        layer.input_L = h.copy()
        h = h.dot(layer.L)

        layer.input_D = h.copy()
        h = h * layer.D

        J += np.log(np.abs(layer.D)).sum() * h.shape[0]

        layer.input_U = h.copy()
        h = h.dot(layer.U)

        layer.input_b = h.copy()
        h += layer.b

        if layer.relu:
            # leaky relu
            layer.input_relu = h.copy()
            J += np.log(np.abs(layer.relu)) * (np.abs(h) < 1.0).sum()
            h = np.where(np.abs(h) < 1.0, layer.relu * h, h - (1.0-layer.relu) * np.sign(h))


    # contract
    J = J - 0.5 * (h**2).sum()
    # h = norm.cdf(h)

    dJdh = -h.copy()

    for layer in layers[-1::-1]:
        if layer.relu:
            layer.dJ_relu = (
                (np.abs(layer.input_relu) < 1.0).sum() / layer.relu +
                (np.where(np.abs(layer.input_relu) < 1.0, layer.input_relu, 0.0) * dJdh).sum() +
                ((layer.input_relu > 1.0) * dJdh).sum() -
                ((layer.input_relu < -1.0) * dJdh).sum()
            )
            dJdh *= np.where(np.abs(layer.input_relu) < 1.0, layer.relu, 1.0)

        layer.dJ_b = dJdh.sum(axis=0)

#        layer.dJ_U = - np.triu(dJdh.T.dot(layer.input_U), 1)
        layer.dJ_U =  np.triu(layer.input_U.T.dot(dJdh), 1)

#        dJdh = dJdh.dot(layer.U.T)
        dJdh = dJdh.dot(layer.U.T)

#        print("foo", layer.D, "\n\n", np.sum(layer.input_D, axis=0), "bar")
        layer.dJ_D = h.shape[0] / layer.D + (layer.input_D * dJdh).sum(axis=0)

        dJdh = dJdh * layer.D

        layer.dJ_L = np.tril(layer.input_L.T.dot(dJdh),-1)

        dJdh = dJdh.dot(layer.L.T)

    return (h, J)


def apply_gradients(layers, epsilon):
    for layer in layers:
        layer.apply_gradient(epsilon)



def invert(output):
    h = output.copy()
    for layer in layers[-1::-1]:
        if layer.relu:
            h =  np.where(np.abs(h) < layer.relu, h / layer.relu, h + np.sign(h) * (1.0 - layer.relu))
        h = h - layer.b
        h = scipy.linalg.solve_triangular(
            layer.U.T, h.T,
            lower=True, unit_diagonal=True, overwrite_b=True).T
        h = h / layer.D
        h = scipy.linalg.solve_triangular(
            layer.L.T, h.T,
            lower=False, unit_diagonal=True, overwrite_b=True).T
    return h

def sample():
    y = invert(np.random.randn(dim))
    im = (norm.cdf(y[:-1]) * 257.0 - 1.0).reshape((28,28))
#    plt.imshow(im)
    label = norm.cdf(im[-1]) * 11.0 - 1.0
    return (label, im)




#mnist = MNIST()
#data = mnist.load_training()
#labels = np.array(data[1])
#images = norm.ppf((1.0 + np.array(data[0]))/257.0)
#labels = norm.ppf((1.0 + labels)/11.0)
#data = np.concatenate([images, np.array([labels]).T],axis=1)

data = np.load('mnist.npy')

dim = len(data[1])
batch_size = 20

epsilon = 1e-4


layers = [ Layer(dim, relu=1.0), Layer(dim, relu=1.0),   Layer(dim, relu=1.0), Layer(dim, relu=1.0), Layer(dim, relu=1.0), Layer(dim) ]
#layers = [Layer(dim)]

if layers:
    layers[-1].relu = 1.0
    pass
else:
    layers = [Layer(dim, relu=1.0), Layer(dim, relu=1.0)]
layers.append(Layer(dim))

#layers = [ Layer(dim, relu=1.0) ]
aJ = 0
for w in range(0,10000000):
    eta = epsilon * 10000.0 / ( 10000.0 + w) / batch_size
    batch = data[np.random.choice(data[0].size, batch_size)]
    _, J = forward(batch)
#    g2 = sum(layer.grad_mag() for layer in layers)
    apply_gradients(layers, eta)
#    J2 = forward(batch)
#    print (J2-J, epsilon * g2)
    aJ = aJ * 0.99 + (J / batch_size) * 0.01
    if w % 100 == 0:
        print(w, aJ)
    if w % 1000 == 0:
        label, im = sample()
        np.save('im_%d_%d' % (w, int(label[0])), im)
        print([layer.relu for layer in layers])
