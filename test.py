import learn_ann
import numpy as np

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

def main():
    seed = np.random.randint(2**32)

    print(f'seed: {seed}')

    Ne = 64

    params = {
        'data': {
            'filename': '/home/shannon/HMRI/experiments/datasets/functions/add5.npz'
        },
        'sampling': {
            'Ne': Ne,
            'seed': seed
        },
        'architecture': {
            'name'
            'Nh': 120,
            'nonlinearity': 'tanh',
            'params': {}
        },
        'compile': {
            'optimizer': 'RMSprop',
            'loss': 'squared_hinge',
        },
        'fit': {
            'epochs': 10000,
            'batch_size': Ne,
            # 'steps_per_epoch': 1,
        }
    }

    for name in ['shared_a', 'shared_b', 'shared_c', 'parallel', 'layered']:
        params['architecture']['name'] = name
        result = learn_ann.learn(params)

        print()
        print(name)
        print_results(result)

    print('\nSECOND')
    params['fit']['batch_size'] //= 2
    params['fit']['epochs'] //= 2

    for name in ['shared_a', 'shared_b', 'shared_c', 'parallel', 'layered']:
        params['architecture']['name'] = name
        result = learn_ann.learn(params)

        print()
        print(name)
        print_results(result)



def print_results(results):
    print(f'learning_time: {results["learning_time"]}\n'
          f'train: {results["trg_err"]} {results["trg_errs"]}\n'
          f'test: {results["test_err"]} {results["test_errs"]}')


if __name__ == '__main__':
    import tensorflow as tf
    with tf.device('/cpu:0'):
        main()
