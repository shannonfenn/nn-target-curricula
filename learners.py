import numpy as np
from keras.layers import Input, Dense, Concatenate
from keras.models import Model
import keras.backend as K
import utils 
import nnutils 


def one_hidden_layer_learner(X, Y, Nh, nonlinearity, compile_kwargs, fit_kwargs):
    _, Ni = X.shape
    _, No = Y.shape

    # input tensor
    inp = Input(shape=(Ni,))
    h = Dense(Nh, activation=nonlinearity, name='hidden')(inp)
    out = Dense(No, activation=nonlinearity)(h)

    model = Model(inputs=inp, outputs=out)

    model.compile(**compile_kwargs)
    model.fit(X, Y, verbose=0, **fit_kwargs)

    return model


def shared_a_learner(X, Y, Nh, nonlinearity, compile_kwargs, fit_kwargs):
    # Single No*Nh-unit hidden layer
    model = one_hidden_layer_learner(X, Y, Nh*Y.shape[1], nonlinearity,
                                     compile_kwargs, fit_kwargs)
    return model, {}


def shared_b_learner(X, Y, Nh, nonlinearity, compile_kwargs, fit_kwargs):
    _, Ni = X.shape
    _, No = Y.shape
    # No, Nh-unit hidden layers
    inp = Input(shape=(Ni,))
    h = inp
    for i in range(No):
        h = Dense(Nh, activation=nonlinearity, name='hidden{}'.format(i))(h)
    out = Dense(No, activation=nonlinearity)(h)

    # training
    model = Model(inputs=inp, outputs=out)
    model.compile(**compile_kwargs)
    model.fit(X, Y, verbose=0, **fit_kwargs)

    return model, {}


def shared_c_learner(X, Y, Nh, nonlinearity, compile_kwargs, fit_kwargs):
    _, Ni = X.shape
    _, No = Y.shape
    # determine target order apriori
    order, F = utils.minfs_curriculum(X, Y)

    # Nt hidden layers - each output unit has its own hidden layer which are connected in succession
    inp = Input(shape=(Ni,))
    hidden_layers = []
    h = inp
    for i in range(No):
        h = Dense(Nh, activation=nonlinearity, name='hidden{}'.format(i))(h)
        hidden_layers.append(h)
    output_units = [None]*No
    for t in order:
        out = Dense(1, activation=nonlinearity)(hidden_layers[t])
        output_units[t] = out

    out = Concatenate()(output_units)

    # training
    model = Model(inputs=inp, outputs=out)
    model.compile(**compile_kwargs)
    model.fit(X, Y, verbose=0, **fit_kwargs)

    return model, {'tgt_order': order, 'feature_sets': F}


def parallel_learner(X, Y, Nh, nonlinearity, compile_kwargs, fit_kwargs):
    Ni = X.shape[1]
    No = Y.shape[1]

    # Single Nh-unit hidden layer - one net per tgt
    inp = Input(shape=(Ni,))

    # hidden layers - each connected to inp
    hidden_layers = []
    for i in range(No):
        h = Dense(Nh, activation=nonlinearity, name='hidden{}'.format(i))(inp)
        hidden_layers.append(h)

    # output units - one per hidden layer
    output_units = []
    for i, h in enumerate(hidden_layers):
        o = Dense(1, activation=nonlinearity)(h)
        output_units.append(o)
    out = Concatenate()(output_units)

    # training
    model = Model(inputs=inp, outputs=out)
    model.compile(**compile_kwargs)
    model.fit(X, Y, verbose=0, **fit_kwargs)

    return model, {}  # no supplementary record info


def layered_learner(X, Y, Nh, nonlinearity, compile_kwargs, fit_kwargs, 
                    use_mask=True, which_layers='input+prev',
                    rescale_epochs=True):
    # Single Nh-unit hidden layer - one net per tgt
    # Same layer structure as 'parallel'

    A = X
    Ni = X.shape[1]
    No = Y.shape[1]
    remaining_targets = np.arange(No, dtype=int)
    learned_targets = []
    feature_sets = []
    layer_configs = []

    if rescale_epochs:
        fit_kwargs = dict(fit_kwargs)
        fit_kwargs['epochs'] = fit_kwargs['epochs'] // No

    for i in range(No):
        # print('strata {}'.format(i))
        remaining_targets = np.setdiff1d(remaining_targets, learned_targets)

        order, F = utils.minfs_curriculum(A, Y[:, remaining_targets])
        t_ = order[0]                  # Get the target ranked as "easiest"
        fs = F[t_]                 # Get the feature set for that target
        t = remaining_targets[t_]  # Get the true index of that target

        # New input matrix
        if use_mask:
            X_prime = A[:, fs]
        else:
            X_prime = A
        submodel = one_hidden_layer_learner(X_prime, Y[:, [t]], Nh, 
                                            nonlinearity, compile_kwargs,
                                            fit_kwargs)

        # append new activations
        H = nnutils.get_activations(submodel, X_prime, 1)
        
        if which_layers == 'input+prev':
            # Use X instead of A to prevent growth
            A = np.hstack([X, H])
        elif which_layers == 'all':
            A = np.hstack([A, H])
        else:
            raise ValueError('Invalid value "{}" for option "which_layers"'.format(which_layers))
        
        learned_targets.append(t)

        layer_configs.append([(l.units, l.activation, l.get_weights())
                              for l in submodel.layers[1:]])
        feature_sets.append(fs)

        # prevent model accumulation causing keras to slow down
        K.clear_session()

    model = join_models(Ni, learned_targets, layer_configs,
                        feature_sets, use_mask, which_layers)

    return model, {'tgt_order': learned_targets, 'feature_sets': feature_sets}


def join_models(Ni, target_order, layer_configs, feature_sets, use_mask, which_layers):
    final_inp = Input(shape=(Ni, ))
    A = final_inp

    No = len(target_order)
    out_layers = [None]*No

    for i, (t, layer_config, fs) in enumerate(zip(target_order, layer_configs, feature_sets)):
        if use_mask:
            Nf = K.get_variable_shape(A)[1]
            F = np.eye(Nf)[:, fs]
            Af = Dense(len(fs), use_bias=False, weights=[F])(A)  # identity activation
        else:
            Af = A

        h = Af
        layers = []
        for units, nonlinearity, weights in layer_config:
            h = Dense(units, activation=nonlinearity, weights=weights)(h)
            layers.append(h)
       
        # build up a list of the output layers in the original target order
        # (the inverse of the permutation given by "target_order")
        hidden_layers = layers[:-1]
        out_layers[t] = layers[-1]

        if i < No - 1:
            if which_layers == 'input+prev':
                # Use X instead of A to prevent growth
                A = Concatenate()([final_inp] + hidden_layers)
            elif which_layers == 'all':
                A = Concatenate()([A] + hidden_layers)

    # final output layer
    final_out = Concatenate()(out_layers)

    model = Model(inputs=final_inp, outputs=final_out)

    return model
