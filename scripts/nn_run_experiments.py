#! /usr/bin/env python

import argparse
import pickle
import json
import gzip
import keras.backend as K

from utils import NumpyAwareJSONEncoder
from learn_ann import learn


def run_multiple_experiments(explistfile, verbose):
    resultfile = explistfile + '.json'
    with open(explistfile) as tasks, open(resultfile, 'w') as ostream:
        for i, line in enumerate(tasks):
            result = run_single_experiment(line.strip(), verbose)
            json.dump(result, ostream, cls=NumpyAwareJSONEncoder,
                      separators=(',', ':'))
            ostream.write('\n')


def run_single_experiment(expfile, verbose):
    K.clear_session()
    # task = pickle.load(open(expfile, 'rb'))
    task = pickle.load(gzip.open(expfile, 'rb'))
    result, model = learn(task, verbose)
    result['id'] = task['id']

    model.save_weights(expfile + '.weights.h5')
    with open(expfile + '.arch.yaml', 'w') as f:
        f.write(model.to_yaml())

    return result


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('experiment', type=str,
                        help='.exp or .explist file to run.')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    if args.experiment.endswith('.explist'):
        run_multiple_experiments(args.experiment, args.verbose)
    elif args.experiment.endswith('.exp'):
        result = run_single_experiment(args.experiment, args.verbose)
        resultfile = args.experiment + '.json'
        with open(resultfile, 'w') as ostream:
            json.dump(result, ostream, cls=NumpyAwareJSONEncoder,
                      separators=(',', ':'))
    else:
        parser.error('[experiment] must be .exp or .explist')


if __name__ == '__main__':
    main()
