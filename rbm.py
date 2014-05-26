# Copyright (c) 2011 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''An implementation of several types of deep belief networks.

This code is largely based on the Matlab generously provided by Taylor, Hinton
and Roweis, and described in their 2006 NIPS paper, "Modeling Human Motion Using
Binary Hidden Variables". Their code and results are available online at
http://www.cs.nyu.edu/~gwtaylor/publications/nips2006mhmublv/.

There are more RBM implementations in this module, though. The basic
(non-Temporal) RBM is based on the Taylor, Hinton, and Roweis code, but stripped
of the dynamic bias terms and significantly refactored into an object-based
approach.

The convolutional RBM code is based on the 2009 ICML paper by Lee, Grosse,
Ranganath and Ng, "Convolutional Deep Belief Networks for Scalable Unsupervised
Learning of Hierarchical Representations".

All implementations incorporate an option to train hidden unit biases using a
sparsity criterion, as described in Lee, Ekanadham and Ng, "Sparse Deep Belief
Net Model for Visual Area V2" (NIPS 2008).

All RBM implementations also provide an option to treat visible units as either
binary or gaussian. Training networks with gaussian visible units is a tricky
dance of parameter-twiddling, but binary units seem quite stable in their
learning and convergence properties. In that light, it would be interesting to
incorporate a hessian-free second-order learning framework like Martens, "Deep
Learning via Hessian-free Optimization" (ICML 2010).

Also, it would be fun (and probably not too much work ?) to move this code onto
a GPU with pycuda. For the time being it runs quickly enough though.
'''

import numpy
import logging
import scipy.signal
from math import ceil
import numpy.random as rng
from itertools import product, izip, imap, cycle
from multiprocessing import Pool
from operator import itemgetter
from itertools import product

def identity(eta):
    return eta
    
def gaussian(eta, mean, std):
    return eta + std*numpy.random.randn(*eta.shape) + mean

def mask_item(x):
    ranges = [range(l) for l in x.shape]
    for mask in product(*ranges):
        yield mask, x[mask]

def bernoulli(p):
    if p.dtype==object:
        result = numpy.empty(p.shape, dtype=object)
        for mask, item in mask_item(p):
            result[mask] = rng.rand(*p[mask].shape) < p[mask]
        return result
    else:
        return rng.rand(*p.shape) < p

def mean(x):
    if x.dtype==object:
        result = numpy.zeros(x.shape)
        for mask, item in mask_item(x):
            result[mask] = item.mean()
        return result
    else:
        return x.mean()

def exp(x):
    if x.dtype == object:
        result = numpy.empty(x.shape, dtype=x.dtype)
        for mask, item in mask_item(x):
            result[mask] = numpy.exp(x[mask])
        return result
    else:
        return numpy.exp(x)

def sigmoid(eta):
    return 1. / (1. + exp(-eta))

def objarray(data, maxdim=None, dtype=object):
    o = numpy.asarray(data, dtype=object)
    r = numpy.empty(o.shape[:maxdim], dtype=dtype)
    for p in product(*[xrange(l) for l in r.shape]):
        r[p] = numpy.asarray(o[p], dtype=numpy.float64)
    return r

class RBM(object):
    '''A restricted boltzmann machine is a type of neural network auto-encoder.
    '''

    def __init__(self, num_visible, num_hidden, visible_mode='binary', 
                 scale=0.001):
        '''Initialize a restricted boltzmann machine.

        num_visible: The number of visible units.
        num_hidden: The number of hidden units.
        binary: True if the visible units are binary, False if the visible units
          are normally distributed.
        '''
        self.weights = scale * rng.randn(num_visible, num_hidden)
        self.hid_bias = 2 * scale * rng.randn(num_hidden)
        self.vis_bias = scale * rng.randn(num_visible)

        if visible_mode == "binary":
            self._visible == sigmoid
        else:
            self._visible = identity
                            
    def hidden_expectation(self, visible, bias=0.):
        '''Given visible data, return the expected hidden unit values.'''
        return sigmoid(numpy.dot(visible, self.weights) + self.hid_bias + bias)

    def visible_expectation(self, hidden, bias=0.):
        '''Given hidden states, return the expected visible unit values.'''
        return self._visible(numpy.dot(hidden, self.weights.T) + self.vis_bias + bias)

    def calculate_gradients(self, visible_batch):
        '''Calculate gradients for a batch of visible data.

        Returns a 3-tuple of gradients: weights, visible bias, hidden bias.

        visible_batch: A (batch size, visible units) array of visible data. Each
          row represents one visible data sample.
        '''
        passes = self.iter_passes(visible_batch)
        v0, h0 = passes.next()
        v1, h1 = passes.next()

        gw = numpy.asarray([
                [numpy.outer(v0[i], h0[i]) - numpy.outer(v1[i], h1[i])]
                for i in xrange(len(visible_batch))
             ]).mean(axis=0).reshape(self.weights.shape)
        gv = (v0 - v1).mean(axis=0)
        gh = (h0 - h1).mean(axis=0)

        #logging.debug('error: %.3g, hidden std: %.3g',
        #              numpy.linalg.norm(gv), h0.std(axis=1).mean())

        return gw, gv, gh

    def apply_gradient(self, name, grad, prev_grad=None, alpha=0.2, momentum=0.2, l2_reg=0):
        '''Apply a gradient to the named parameter array.

        Returns the applied gradient.

        name: The name (a string) of a parameter to adjust.
        grad: Gradient for the named parameters.
        prev_grad: If given, this should be a previous gradient to use for
          momentum calculations.
        alpha: Learning rate.
        momentum: A parameter in (0, 1) that determines the weight momentum.
        l2_reg: A parameter (usually << 1) that controls the amount of L2 weight
          regularization.
        '''
        target = getattr(self, name)
        if 1 > momentum > 0 and prev_grad is not None:
            grad = momentum * prev_grad + (1 - momentum) * alpha * (grad - l2_reg * target)
        target += grad
        return grad

    def iter_passes(self, visible):
        '''Repeatedly pass the given visible layer up and then back down.

        Generates the resulting sequence of (visible, hidden) states. The first
        pair will be the (original visible, resulting hidden) states, followed
        by the subsequent (visible down-pass, hidden up-pass) pairs.

        visible: The initial state of the visible layer.
        '''
        while True:
            hidden = self.hidden_expectation(visible)
            yield visible, hidden
            visible = self.visible_expectation(bernoulli(hidden))

    def reconstruct(self, visible, passes=1):
        '''Reconstruct a given visible layer through the hidden layer.

        visible: The initial state of the visible layer.
        passes: The number of up- and down-passes.
        '''
        for i, (visible, _) in enumerate(self.iter_passes(visible)):
            if i + 1 == passes:
                return visible

    def plot(self, fig):
        '''Given a figure, plot the parameters (weights and bias) of this RBM.
        '''
        T = self.order

        ax = fig.add_subplot(6, 1, 1)
        ax.imshow(self.weights)
        ax.colorbar()
        ax = fig.add_subplot(6, 1, 2)
        ax.hist(self.weights)

        ax = fig.add_subplot(6, 1, 3)
        ax.imshow(self.bias_visible)
        ax.colorbar()
        ax = fig.add_subplot(6, 1, 4)
        ax.hist(self.bias_visible)

        ax = fig.add_subplot(6, 1, 5)
        ax.imshow(self.bias_hidden)
        ax.colorbar()
        ax = fig.add_subplot(6, 1, 6)
        ax.hist(self.bias_hidden)

    class Trainer(object):
        '''
        '''

        def __init__(self, rbm, momentum, target_sparsity=None):
            '''
            '''
            self.rbm = rbm
            self.momentum = momentum
            self.target_sparsity = target_sparsity

            self.grad_weights = numpy.zeros(rbm.weights.shape, float)
            self.grad_vis = numpy.zeros(rbm.vis_bias.shape, float)
            self.grad_hid = numpy.zeros(rbm.hid_bias.shape, float)

        def learn(self, batch, l2_reg=0, alpha=0.2):
            '''
            '''
            w, v, h = self.rbm.calculate_gradients(batch)
            if self.target_sparsity is not None:
                _, h0 = self.rbm.iter_passes(batch).next()
                h = self.target_sparsity - h0.mean(axis=0)

            kwargs = dict(alpha=alpha, momentum=self.momentum)
            self.grad_vis = self.rbm.apply_gradient('vis_bias', v, self.grad_vis, **kwargs)
            self.grad_hid = self.rbm.apply_gradient('hid_bias', h, self.grad_hid, **kwargs)

            kwargs['l2_reg'] = l2_reg
            self.grad_weights = self.rbm.apply_gradient('weights', w, self.grad_weights, **kwargs)
            return numpy.linalg.norm(v)


class Temporal(RBM):
    '''An RBM that incorporates temporal (dynamic) visible and hidden biases.

    This RBM is based on work and code by Taylor, Hinton, and Roweis (2006).
    '''

    def __init__(self, num_visible, num_hidden, order, binary=True, scale=0.001):
        '''
        '''
        super(TemporalRBM, self).__init__(num_visible, num_hidden, binary=binary, scale=scale)

        self.order = order

        self.vis_dyn_bias = scale * rng.randn(order, num_visible, num_visible)
        self.hid_dyn_bias = scale * rng.randn(order, num_hidden, num_visible) - 1.

    def calculate_gradients(self, frames_batch):
        '''Calculate gradients using contrastive divergence.

        Returns a 5-tuple of gradients: weights, visible bias, hidden bias,
        dynamic visible bias, and dynamic hidden bias.

        frames_batch: An (order, visible units, batch size) array containing a
          batch of frames of visible data.

          Frames are assumed to be reversed temporally, across the order
          dimension, i.e., frames_batch[0] is the current visible frame in each
          batch element, frames_batch[1] is the previous visible frame,
          frames_batch[2] is the one before that, etc.
        '''
        order, _, batch_size = frames_batch.shape
        assert order == self.order

        vis_bias = sum(numpy.dot(self.vis_dyn_bias[i], f).T for i, f in enumerate(frames_batch))
        hid_bias = sum(numpy.dot(self.hid_dyn_bias[i], f).T for i, f in enumerate(frames_batch))

        v0 = frames_batch[0].T
        h0 = self.hidden_expectation(v0, hid_bias)
        v1 = self.visible_expectation(bernoulli(h0), vis_bias)
        h1 = self.hidden_expectation(v1, hid_bias)

        gw = (numpy.dot(h0.T, v0) - numpy.dot(h1.T, v1)) / batch_size
        gv = (v0 - v1).mean(axis=0)
        gh = (h0 - h1).mean(axis=0)

        gvd = numpy.zeros(self.vis_dyn_bias.shape, float)
        ghd = numpy.zeros(self.hid_dyn_bias.shape, float)
        v = v0 - self.vis_bias - vis_bias
        for i, f in enumerate(frames_batch):
            gvd[i] += numpy.dot(v.T, f)
            ghd[i] += numpy.dot(h0.T, f)
        v = v1 - self.vis_bias - vis_bias
        for i, f in enumerate(frames_batch):
            gvd[i] -= numpy.dot(v.T, f)
            ghd[i] -= numpy.dot(h1.T, f)

        return gw, gv, gh, gvd, ghd

    def iter_passes(self, frames):
        '''Repeatedly pass the given visible layer up and then back down.

        Generates the resulting sequence of (visible, hidden) states.

        visible: An (order, visible units) array containing frames of visible
          data to "prime" the network. The temporal order of the frames is
          assumed to be reversed, so frames[0] will be the current visible
          frame, frames[1] is the previous frame, etc.
        '''
        vdb = self.vis_dyn_bias[0]
        vis_dyn_bias = collections.deque(
            [numpy.dot(self.vis_dyn_bias[i], f).T for i, f in enumerate(frames)],
            maxlen=self.order)

        hdb = self.hid_dyn_bias[0]
        hid_dyn_bias = collections.deque(
            [numpy.dot(self.hid_dyn_bias[i], f).T for i, f in enumerate(frames)],
            maxlen=self.order)

        visible = frames[0]
        while True:
            hidden = self.hidden_expectation(visible, sum(hid_dyn_bias))
            yield visible, hidden
            visible = self.visible_expectation(bernoulli(hidden), sum(vis_dyn_bias))
            vis_dyn_bias.append(numpy.dot(vdb, visible))
            hid_dyn_bias.append(numpy.dot(hdb, visible))

    def plot(self, fig):
        '''
        '''
        T = self.order

        ax = fig.add_subplot(6, T + 1, 0 * T + 1)
        ax.imshow(self.weights)
        ax.colorbar()
        ax = fig.add_subplot(6, T + 1, 1 * T + 1)
        ax.hist(self.weights)

        ax = fig.add_subplot(6, T + 1, 2 * T + 1)
        ax.imshow(self.bias_visible)
        ax.colorbar()
        ax = fig.add_subplot(6, T + 1, 3 * T + 1)
        ax.hist(self.bias_visible)

        ax = fig.add_subplot(6, T + 1, 4 * T + 1)
        ax.imshow(self.bias_hidden)
        ax.colorbar()
        ax = fig.add_subplot(6, T + 1, 5 * T + 1)
        ax.hist(self.bias_hidden)

        for t in range(T):
            ax = fig.add_subplot(6, T + 1, 1 * t * T + 2)
            ax.imshow(self.vis_dyn_bias[t])
            ax.colorbar()
            ax = fig.add_subplot(6, T + 1, 2 * t * T + 2)
            ax.hist(self.vis_dyn_bias[t])

            ax = fig.add_subplot(6, T + 1, 1 * t * T + 3)
            ax.imshow(self.hid_dyn_bias[t])
            ax.colorbar()
            ax = fig.add_subplot(6, T + 1, 2 * t * T + 3)
            ax.hist(self.hid_dyn_bias[t])

    class Trainer(object):
        '''
        '''

        def __init__(self, rbm, momentum, target_sparsity=None):
            '''
            '''
            self.rbm = rbm
            self.momentum = momentum
            self.target_sparsity = target_sparsity

            self.grad_weights = numpy.zeros(rbm.weights.shape, float)
            self.grad_vis = numpy.zeros(rbm.vis_bias.shape, float)
            self.grad_hid = numpy.zeros(rbm.hid_bias.shape, float)
            self.grad_dyn_vis = numpy.zeros(rbm.hid_dyn_bias.shape, float)
            self.grad_dyn_hid = numpy.zeros(rbm.hid_dyn_bias.shape, float)

        def learn(self, batch, l2_reg=0, alpha=0.2):
            '''
            '''
            w, v, h, vd, hd = self.rbm.calculate_gradients(batch)
            if self.target_sparsity is not None:
                #h = self.target_sparsity - h0.mean(axis=0)
                pass

            kwargs = dict(alpha=alpha, momentum=self.momentum)
            self.grad_vis = self.rbm.apply_gradient('vis_bias', v, self.grad_vis, **kwargs)
            self.grad_hid = self.rbm.apply_gradient('hid_bias', h, self.grad_hid, **kwargs)

            kwargs['l2_reg'] = l2_reg
            self.grad_weights = self.rbm.apply_gradient('weights', w, self.grad_weights, **kwargs)
            self.grad_vis_dyn = self.rbm.apply_gradient('vis_dyn_bias', vd, self.grad_vis_dyn, **kwargs)
            self.grad_hid_dyn = self.rbm.apply_gradient('hid_dyn_bias', hd, self.grad_hid_dyn, **kwargs)


conv2d = scipy.signal.convolve


def atomic_conv(params):
    x, y, mode = params
    return conv2d(x, y, mode)

def atomic_mconv(params):
    return atomic_conv(params)/params[1].size

def conv3d(x, y, mode="valid", n_jobs=2):
    """
    Convolution of x (3 dimensional tensor) and y (3 dimensional tensors).
    Optionally when x and y have only two dimensions, it automatically adds
    one extra dimension to indicate number of batch is 1 or number of
    filters is 1.

    The result is a 4 dimensional tesnor where first dimension is batch index,
    second dimension is filter base index and and last two dimensions are 
    convolutions.

    """
    pool = Pool(processes=n_jobs)
    jj, ii = numpy.meshgrid(numpy.arange(y.shape[0]), numpy.arange(x.shape[0]))
    params = izip(imap(x.__getitem__, ii.ravel()),
                  imap(y.__getitem__, jj.ravel()),
                  cycle([mode]))
    convs = pool.map_async(atomic_conv, params).get()
    pool.close()
    pool.join()
    result = numpy.asarray(convs, dtype=object).reshape(x.shape[0], y.shape[0])
    return result

def conv4d(x, y, mode="valid", n_jobs=2, processes=None):
    """
    Convolution of x (4 dimensional tensor) and y (3 dimensional tensors)

    The result is a 4 dimesional tensor where first dimension is batch index,
    second dimension is filter base index and last two dimensions are 
    convolutions.

    """
    assert x.shape[1]==y.shape[0]
    if processes is None:
        pool = Pool(processes=n_jobs)
    else:
        pool = processes
    ii, jj = zip(*[((k, i), (i, j))\
                   for k in xrange(x.shape[0])\
                   for i in xrange(x.shape[1])\
                   for j in xrange(y.shape[1])])
    params = izip(map(x.__getitem__, ii),
                  map(y.__getitem__, jj),
                  cycle([mode]))

    chunksize = int(ceil(x.shape[0]*x.shape[1]*y.shape[1]*1./pool._processes))
    convs = pool.map(atomic_conv, params, chunksize)
    if processes is None:
        pool.close()
        pool.join()
    result = objarray(convs, dtype=object, maxdim=1)
    result = result.reshape(x.shape[0], x.shape[1], y.shape[1])
    return result

def pool_sum(params):
    active, pool_shape, scaled = params
    r, c = active.shape
    rows, cols = pool_shape
    if type(rows) is float:
        assert rows <= 1.0
        rows = int(numpy.ceil(rows*r))
    if type(cols) is float:
        assert cols <= 1.0
        cols = int(numpy.ceil(cols*c))
    _r = int(numpy.ceil(float(r)/rows))
    _c = int(numpy.ceil(float(c)/cols))
    if scaled:
        pool = numpy.zeros((_r, _c), dtype=numpy.float64)
    else:
        pool = numpy.zeros((r, c), dtype=numpy.float64)
    for i in xrange(_r):
        for j in xrange(_c):
            rslice = slice(i * rows, (i + 1) * rows)
            cslice = slice(j * cols, (j + 1) * cols)
            mask = (rslice, cslice)
            s = active[mask].sum()
            if scaled:
                pool[i, j] = s
            else:
                pool[mask] = s
    return pool

class Convolutional(RBM):
    '''
    '''
    def my_gaussian(self, mean, std):

        def f(eta):
            return gaussian(eta, mean, std)

        return f

    def __init__(self, inp_layers, num_filters, filter_shape, pool_shape,
                       visible_mode="binary", scale=0.001, 
                       hidden_prob="softmax", n_jobs=2,
                       noise_mean=0, noise_std=1):
        '''Initialize a convolutional restricted boltzmann machine.
        elif visilbe_mode=="normal":
            self._visible = normal

        num_filters: The number of convolution filters.
        filter_shape: An ordered pair describing the shape of the filters.
        pool_shape: An ordered pair describing the shape of the pooling groups.
        binary: True if the visible units are binary, False if the visible units
          are normally distributed.
        scale: Scale initial values by this parameter.
        @param pro: probability pooling function. If "raw", then a raw result 
            from convolution is applied, if "sigmoid" then a sigmoid of 
            result from convolution is applied.
        '''
        self.inp_layers = inp_layers
        self.num_filters = num_filters
        self.filter_shape = filter_shape
        self.weights = scale * rng.randn(inp_layers, num_filters, *filter_shape)
        self.vis_bias = scale * rng.randn()
        self.hid_bias = 2 * scale * rng.randn(num_filters)
        self.n_jobs = n_jobs
        self.processes = Pool(self.n_jobs)

        if visible_mode=="binary":
            self._visible = sigmoid
        else:
            self._visible = self.my_gaussian(noise_mean, noise_std)

        self.pool_shape = pool_shape

        if hidden_prob == "sigmoid":
            self.hidden_prob = self.my_sigmoid
        else:
            self.hidden_prob = self.softmax

    def _pool_sum(self, active, scaled=False):
        '''Given activity in the hidden units, pool it into groups.'''
        pool = Pool(processes=self.n_jobs)
        params = izip(active.ravel(),
                      cycle([self.pool_shape]),
                      cycle([scaled]))
        result = pool.map_async(pool_sum, params).get()
        pool.close()
        pool.join()
        result = objarray(result, maxdim=1)
        result = result.reshape(active.shape[0], active.shape[1])
        return result
        
    def hidden_raw(self, visible):
        '''Given visible data, return the expected pooling unit values.'''
        #hid_bias = self.hid_bias.reshape(1, self.num_filters)
        active = conv4d(visible, self.weights[:, :, ::-1, ::-1], 'valid',
                        processes=self.processes)
        active = active.sum(axis=1) + self.hid_bias
        return active

    def softmax(self, visible):
        '''Given visible data, return the expected hidden unit values.'''
        active = exp(self.hidden_raw(visible))
        return active / (1. + self._pool_sum(active))

    def my_sigmoid(self, visible):
        return sigmoid(self.hidden_raw(visible))

    def pooled_softmax(self, visible):
        active = exp(self.hidden_raw(visible))
        return 1. - 1./(1. + self._pool_sum(active, scaled=True)) 

    def visible_expectation(self, hidden):
        '''Given hidden states, return the expected visible unit values.'''
        result = conv4d(hidden, numpy.transpose(self.weights, [1, 0, 2, 3]),
                        'full', processes=self.processes).sum(axis=1) + self.vis_bias
        return self._visible(result)

    def close(self):
        self.processes.close()
        self.processes.join()

    def vh2w(self, v, h):
        ih = numpy.array([l[::-1, ::-1]/l.size for l in h.ravel()], dtype=object)
        ih = objarray(ih, maxdim=1).reshape(h.shape)
        tv = None
        if len(v.shape)==2:
            tv = v.T
        elif len(v.shape)==4:
            tv = numpy.transpose(v, (1, 0, 2, 3))
        w = conv4d(tv, ih, processes=self.processes).mean(axis=1)
        w = numpy.asarray(list(w.ravel())).reshape(self.inp_layers, 
                                                   self.num_filters,
                                                   *self.filter_shape)
        return w

    def hidden_mean(self, h):
        if h.dtype==object:
            return mean(h).mean(axis=0)
        else:
            return h.mean(axis=-1).mean(axis=-1).mean(axis=0)

    def calculate_gradients(self, visible, idx=None):
        '''Calculate gradients for an instance of visible data.

        Returns a 3-tuple of gradients: weights, visible bias, hidden bias.

        visible: A single array of visible data.
        '''
        v0 = visible
        h0 = self.hidden_prob(v0)

        if self.persistant:
            v1 = self.visible_expectation(bernoulli(h0))
            h1 = self.hidden_prob(v1)
        else:
            assert idx is not None
            assert self.V_ is not None, "You haven't initialised the model yet"
            h_ = self.hidden_prob(self.V_[idx])
            v1 = self.visible_expectation(bernoulli(h_))
            h1 = self.hidden_prob(v1)
            self.V_[idx] = v1

        gw = (self.vh2w(v0, h0) - self.vh2w(v1, h1))
        gv = mean(v0 - v1).mean()
        gh = self.hidden_mean(h0 - h1)

        return gw, gv, gh

    class Trainer:
        '''
        '''

        def __init__(self, rbm, momentum, target_sparsity=None,
                           persistant=False):
            '''
                
            '''
            self.rbm = rbm
            self.momentum = momentum
            selfj.target_sparsity = target_sparsity

            self.grad_weights = numpy.zeros(rbm.weights.shape, float)
            self.grad_vis = 0.
            self.grad_hid = numpy.zeros(rbm.hid_bias.shape, float)

            self.persistant = persistant
            self.V_ = None
            
        def init(self, V, iterations=100):
            """ Initialise ``iterations`` Markov Chains v_ to all zero states.
                Used only when ``peristant`` is True.
                
            """
            assert self.persistant
            V_ = V
            for i in xrange(iterations):
                H = self.hidden_prob(V_)
                V_ = self.visible_expectation(bernoulli(H))
            self.V_ = V_
                
        def batch_learn(self, visible, batchsize=100, epochs=100, lr=.01, verbose=True, print_after=10):
            if verbose:
                print 'Pre-training the model...'
            n_batches = X.shape[0]/batch_size
            cost = []
            for epoch in xrange(epochs):
                c = []
                # Shuffle X for stochastic gradient descent
                idx = permutation(X.shape[0])
                X_ = X[idx]        
                for batch_index in xrange(n_batches):
                    sl = slice(batch_index*batch_size, (batch_index + 1)*batch_size)
                    cost = trainer.learn(X_[sl], lr=lr, idx=idx[sl])     
                    c.append(cost)
                    if verbose and batch_index % print_after==0:
                        print "Epoch %d, Batch %d, costs: %.5f" % (epoch, batch_index, np.average(c))
                if verbose:
                    print "Epoch %d, cost %.5f" % (epoch, numpy.average(c))
                cost.append(numpy.average(c))
            return numpy.asarray(cost)

        def learn(self, visible, lr=0.1, l2_reg=0, idx=None):
            w, v, h = self.rbm.calculate_gradients(visible, idx)
            if self.target_sparsity is not None:
                # h -= self.target_sparsity*self.rbm.hidden_mean(self.rbm.hidden_prob(visible))
                h -= (self.rbm.hidden_mean(self.rbm.hidden_prob(visible)) - self.target_sparsity)

            kwargs = dict(alpha=lr, momentum=self.momentum)
            self.grad_vis = self.rbm.apply_gradient('vis_bias', v, self.grad_vis, **kwargs)
            self.grad_hid = self.rbm.apply_gradient('hid_bias', h, self.grad_hid, **kwargs)

            kwargs['l2_reg'] = l2_reg
            self.grad_weights = self.rbm.apply_gradient('weights', w, self.grad_weights, **kwargs)
            return numpy.linalg.norm(v)
