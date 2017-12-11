import numpy as np
import json
import keras.backend as K
from keras.regularizers import Regularizer
import minfs.feature_selection as fss
from progress.bar import IncrementalBar


class BipolarL1Regularizer(Regularizer):
    """Regularizer for L1-style bipolar regularization.
    # Arguments
        gamma: Float; bipolar regularization factor.
        alpha: Float; non-zero regularization factor.
        beta: Float; non-zero regularization decay.
    """

    def __init__(self, gamma=0.0, alpha=0.0, beta=0.0):
        self.gamma = K.cast_to_floatx(gamma)
        self.alpha = K.cast_to_floatx(alpha)
        self.beta = K.cast_to_floatx(beta)

    def __call__(self, x):
        regularization = 0.0
        if self.gamma:
            regularization += K.sum(self.gamma * K.abs(1-x) * K.abs(-1-x))
        if self.alpha and self.beta:
            regularization += K.sum(self.alpha * K.exp(-self.beta * K.abs(x)))
        return regularization

    def get_config(self):
        return {'gamma': float(self.gamma),
                'alpha': float(self.alpha),
                'beta': float(self.beta)}


# Optimised and simplified based on 
# https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
# def get_activations(model, model_inputs, layer_name=None):
#     # layer_name=None -> return all layer outputs
#     outputs = [layer.output
#                for layer in model.layers
#                if layer.name == layer_name or layer_name is None]
#      # evaluation function
#     functor = K.function([model.input] + [K.learning_phase()], outputs)
#     # Learning phase 0.0 = Test mode (no dropout or batch normalization)
#     activations = functor([model_inputs, 0.0])[0]
#     return activations
def get_activations(model, model_inputs, layer_index=None):
    if layer_index is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [model.layers[layer_index].output]
     # evaluation function
    functor = K.function([model.input] + [K.learning_phase()], outputs)
    # Learning phase 0.0 = Test mode (no dropout or batch normalization)
    activations = functor([model_inputs, 0.0])
    if layer_index is not None:
        activations = activations[0]
    return activations


def minfs_curriculum(X, Y):
    if len(np.unique(Y)) != 2:
        raise ValueError('Error: non-binary target')
    thresh = (Y.max() + Y.min()) / 2
    X = (X >= thresh).astype(np.uint8)
    Y = (Y >= thresh).astype(np.uint8)
    R, F = fss.ranked_feature_sets(X, Y)
    order = fss.order_from_rank(R)
    return order, F


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class BetterETABar(IncrementalBar):
    suffix = ('%(index)d/%(max)d | elapsed: %(elapsed)ds | '
              'eta: %(better_eta)ds')

    @property
    def better_eta(self):
        return self.elapsed / (self.index + 1) * self.remaining

    def writeln(self, line):
        if self.file.isatty():
            self.clearln()
            print('\x1b[?7l' + line + '\x1b[?7h', end='', file=self.file)
            self.file.flush()
