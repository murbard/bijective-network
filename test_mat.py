import numpy as np
from scipy.stats import norm
import scipy

dim = 6
n = 10000
mat = np.random.randn(dim, dim) * 0.05 + np.diag(np.sqrt(np.random.randint(20,size=dim)))
X = np.random.randn(n, dim).dot(mat)


def compute(x, M) :
    h = x.dot(M)
    J = np.log(np.linalg.det(M)) - 0.5 * h.dot(h.transpose()) #n np.sum(np.log(norm.pdf(h)))
    dJ = np.linalg.inv(M).transpose() - x.transpose().dot(h)
    return (J, dJ)


order = []
for i in range(0,dim):
    for j in range(i+1, dim):
        order.append((i,j))


def compute2(x, ms) :
    x = x.transpose()
    h = x.copy()
    dim = x.shape[0]
    dJ = np.zeros(ms.shape)
    io = np.zeros((dim,dim,2))
    J = 0
    for (i,j) in order:
        a, b, c, d = ms[i,j,0,0], ms[i,j,0,1], ms[i,j,1,0], ms[i,j,1,1]
        io[i,j,0], io[i,j,1] = h[i,0],h[j,0]
        h[i,0], h[j,0] = a * h[i,0] + b * h[j,0], c* h[i,0] + d * h[j,0]
        det = a * d - b * c
        J += np.log(np.abs(det))
        ia, ib, ic, id = d / det, -c / det, -b / det,  a / det
        dJ[i,j] = np.array([[d, -c],[-b,a]]) / det

    J -= 0.5 * h.transpose().dot(h)

    dh = h.copy()
    for (i,j) in order[-1::-1]:

        dJ[i,j] -= np.array([[dh[i,0] * io[i,j,0], dh[i,0] * io[i,j,1]],[dh[j,0] * io[i,j,0], dh[j,0] * io[i,j,1]]])
        a, b, c, d = ms[i,j,0,0], ms[i,j,0,1], ms[i,j,1,0], ms[i,j,1,1]
        dh[i,0], dh[j,0] = a * dh[i,0] + c * dh[j,0], b * dh[i,0] + d * dh[j,0] # transposed

    return (J, dJ)

def compute3(x, m):
    l, d, u = m[0], m[1], m[2]
    x = x.transpose()
    h1 = l.dot(x)
    h2 = (h1.transpose() * d).transpose()
    y  = u.dot(h2)

    dy = u.transpose().dot(y)
    dh2 = (dy.transpose() * d).transpose()
    dJ = [- np.tril(dh2.dot(x.transpose()),-1) , 1.0 / d - (dy * h1)[:,0] , - np.triu(y.dot(h2.transpose()),1)]
    return dJ


eigs = np.array(scipy.linalg.eig(X.transpose().dot(X)/n)[0],dtype=float)


def test():
    M = np.eye(dim) + np.random.randn(dim, dim) * 0.1
    aJ = 0
    epsilon = 1e-4
    for e in range(0,1000000):
        k = np.random.randint(n)
        x = X[k:k+1,:]


        J, dJ = compute(x, M)
        aJ = aJ * 0.99 + J * 0.01
        eta =  epsilon * 1000.0 / (1000.0 + e)
        for (i,j) in order[-1::-1]:
            M += dJ * eta

        if e % 1000 == 0:
            print(aJ, J, (sum(np.linalg.inv(M)[:,:]**2)),(eigs))


def test2():
    ms = np.zeros((dim,dim,2,2))
    for (i,j) in order:
        ms[i,j] = np.eye(2)

    aJ = 0
    epsilon = 1e-3
    for e in range(0,1000000):
        k = np.random.randint(n)
        x = X[k:k+1,:]
        J, dJ = compute2(x, ms)
        aJ = aJ * 0.99 + J * 0.01
        eta =  epsilon * 1000.0 / (1000.0 + e)
        for (i,j) in order[-1::-1]:
            ms[i,j] += eta * dJ[i,j]

        if e % 1000 == 0:
            M = np.eye(dim, dim)
            for (i,j) in order:
                M[i, :], M[j, :] = ms[i,j,0,0] * M[i, :] + ms[i,j,0,1] * M[j, :], ms[i,j,1,0] * M[i, :] + ms[i,j,1,1] * M[j, :]
            print(aJ, J, (sum(np.linalg.inv(M)[:,:]**2)),(eigs))

def test3():
    m = [np.eye(dim, dim), np.ones(dim), np.eye(dim,dim)]
    epsilon = 1e-4
    for e in range(0,1000000):
        k = np.random.randint(n)
        x = X[k:k+1,:]
        dJ = compute3(x, m)
        eta =  epsilon * 1000.0 / (1000.0 + e)
        m[0] += eta * dJ[0]
        m[1] += eta * dJ[1]
        m[2] += eta * dJ[2]
        if e % 1000 == 0:
            M = m[2].dot(np.diag(m[1])).dot(m[0])
            print(sorted(sum(np.linalg.inv(M)[:,:]**2)),sorted(eigs))
