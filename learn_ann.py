import numpy as np
import time

import bitpacking.packing as pk
import learners
from nnutils import BipolarL1Regularizer


LEARNERS = {
    'shared_a': learners.shared_a_learner,
    'shared_b': learners.shared_b_learner,
    'shared_c': learners.shared_c_learner,
    'parallel': learners.parallel_learner,
    'layered': learners.layered_learner,
}


def error(model, X, Y):
    # todo: is predict_on_batch faster?
    Y_pred = np.sign(model.predict(X))
    errs = (Y_pred != Y).mean(axis=0)
    return errs


def mcc(model, X, Y):
    # todo: is predict_on_batch faster?
    Y_pred = np.sign(model.predict(X))
    tp = np.logical_and(Y_pred >  0, Y >  0).sum(axis=0).astype(np.float64)
    fp = np.logical_and(Y_pred >  0, Y <= 0).sum(axis=0).astype(np.float64)
    tn = np.logical_and(Y_pred <= 0, Y <= 0).sum(axis=0).astype(np.float64)
    fn = np.logical_and(Y_pred <= 0, Y >  0).sum(axis=0).astype(np.float64)

    actual_p = tp + fn
    actual_n = tn + fp
    pred_p = tp + fp
    pred_n = tn + fn

    numerators = (tp * tn - tp * fn)
    denominators = np.sqrt(actual_p * actual_n * pred_p * pred_n)

    # only one given class give accuracy in [-1, 1]
    # note either tn and act_n both zero, or tp and act_p both zero
    numerators = np.where((actual_p * actual_n) == 0,
                          (tn + tp) / (actual_n + actual_p) * 2 - 1,
                          numerators)
    # normal limitting case when two classes present but only one predicted
    numerators = np.where((pred_p * pred_n) == 0, 0, numerators)
    # remove div 0 for limiting cases above
    denominators = np.where(denominators == 0, 1, denominators)

    return numerators / denominators


def errors(model, X_trg, Y_trg, X_test, Y_test):
    trg_errs = error(model, X_trg, Y_trg)
    test_errs = error(model, X_test, Y_test)
    trg_mccs = mcc(model, X_trg, Y_trg)
    test_mccs = mcc(model, X_test, Y_test)
    record = {'trg_errs': trg_errs,       'test_errs': test_errs,
              'trg_err': trg_errs.mean(), 'test_err': test_errs.mean(),
              'trg_mccs': trg_mccs,       'test_mccs': test_mccs,
              'trg_mcc': trg_mccs.mean(), 'test_mcc': test_mccs.mean()}
    return record


def load_dataset(mapping):
    if mapping['type'] == 'file_unsplit':
        with np.load(mapping['file']) as data:
            M = pk.unpackmat(data['matrix'], data['Ne']).astype(np.float32)
            # Inputs and Outputs in the range [-1,+1]
            M = 2 * M - 1.0
            X, Y = np.hsplit(M, [data['Ni']])
    else:
        raise ValueError()

    if 'targets' in mapping:
        Y = Y[:, mapping['targets']]

    train_sample = mapping['training_indices']
    test_sample = mapping['test_indices']
    if test_sample is None:
        test_sample = [i
                       for i in range(X.shape[0])
                       if i not in train_sample]

    X_train, Y_train = X[train_sample], Y[train_sample]
    X_test, Y_test = X[test_sample], Y[test_sample]

    return X_train, Y_train, X_test, Y_test


def learn(params, verbose=False):
    # params:
    #     data: 
    #         filename: str   e.g.  '/home/shannon/HMRI/experiments/datasets/functions/add5.npz'
    #     sampling:
    #         Ne: int
    #         seed: int

    #     Nh: int     e.g. 60
    #     nonlinearity: str
    #     nolearn_params: dict
    #     architecture: 
    #         name: str
    #         params: dict (opt)

    t_start = time.time()


    # mode = -1 if nonlinearity == 'tanh' else 0

    X_trg, Y_trg, X_test, Y_test = load_dataset(params['mapping'])

    Ne = X_trg.shape[0]

    learner = LEARNERS[params['architecture']['name']]
    learner_kwargs = params['architecture']['params']
    Nh = params['architecture']['Nh']
    nonlin = params['architecture']['nonlinearity']
    regularizer = params['architecture'].get('regularizer', None)
    if regularizer is not None:
        regularizer = BipolarL1Regularizer(regularizer.get('gamma', 0.0), 
                                           regularizer.get('alpha', 0.0), 
                                           regularizer.get('beta', 0.0))

    compile_params = dict(params['compile'])
    fit_params = dict(params['fit'])

    if 'batch_ratio' in params:
        fit_params['batch_size'] = int(Ne * params['batch_ratio'])
        # for fair speed comparison
        fit_params['epochs'] = int(fit_params['epochs'] * params['batch_ratio'])
    elif 'batch_size' not in fit_params:
        # no specified batch size -> full batch
        fit_params['batch_size'] = Ne

    t_setup = time.time()
    model, learner_record = learner(X_trg, Y_trg, Nh, nonlin, compile_params,
                                    fit_params, regularizer, **learner_kwargs)

    t_learn = time.time()

    errors_record = errors(model, X_trg, Y_trg, X_test, Y_test)

    t_final = time.time()

    results = {'Ne': X_trg.shape[0],
               'Ni': X_trg.shape[1],
               'No': Y_trg.shape[1],
               'Nh': Nh,
               'batch_size': fit_params['batch_size'],
               'architecture': params['architecture']['name'],
               'setup_time': t_setup - t_start,
               'learning_time': t_learn - t_setup,
               'eval_time': t_final - t_learn}

    if 'seed' in params['sampling']:
        results['sample_seed'] = params['sampling']['seed']
    if 'targets' in params['data']:
        results['given_tgts'] = params['data']['targets']

    results.update(learner_record)
    results.update(errors_record)

    return results
