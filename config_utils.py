from copy import deepcopy
from itertools import product
from collections import MutableMapping
from good import Invalid
import numpy as np
import os
import random

import schema as sch
import utils


def build_filename(params, extension, key='filename'):
    ''' filename helper with optional directory'''
    filename = os.path.expanduser(params[key])
    location = params.get('dir', None)
    # if 'filename' is absolute then ignore 'dir'
    if location and not os.path.isabs(filename):
        filename = os.path.join(location, filename)
    # add extension if missing
    if os.path.splitext(filename)[-1] == '':
        filename += extension
    return filename


def get_seed(key):
    ''' Keeps a registry of seeds for each key, if given a new
        key get_seed() generates a new seed for that key, but if
        given an existing key it returns that seed. Allows any number
        of named seeds.'''
    if 'registry' not in get_seed.__dict__:
        # first call, create the registry
        get_seed.registry = {}
    if key not in get_seed.registry:
        # non-existant key, generate a seed
        random.seed()  # use default randomness source to get a seed
        get_seed.registry[key] = random.randint(1, 2**32-1)
    return get_seed.registry[key]


def load_samples(params, N, Ni, Nt=None):
    if params['type'] == 'given':
        training_indices = np.array(params['indices'], dtype=np.uintp)
        if 'test' in params:
            test_indices = np.array(params['test'], dtype=np.uintp)
        else:
            test_indices = [None]*training_indices.shape[0]

    if params['type'] == 'file':
        filename = build_filename(params, '.npy')
        training_indices = np.load(filename)
        if 'test' in params:
            filename = build_filename(params, '.npy', 'test')
            test_indices = np.load(filename)
        else:
            test_indices = [None]*training_indices.shape[0]

    elif params['type'] == 'generated':
        # this provided seed allows us to generate the same set
        # of training indices across multiple configurations
        Ns = params['Ns']
        Ne = params['Ne']
        s = params['seed']
        if isinstance(s, str):
            # s is actually a name
            s = get_seed(s)
            params['seed'] = s
        random.seed(s)
        if 'test' in params:
            Ne_test = params['test']
            if Nt is None:
                # combined training and test data
                all_indices = np.array([
                    random.sample(range(N), Ne+Ne_test) for i in range(Ns)])
                training_indices, test_indices = np.hsplit(all_indices, [Ne])
            else:
                # separate training and test data
                training_indices = np.array([
                    random.sample(range(N), Ne) for i in range(Ns)])
                test_indices = np.array([
                    random.sample(range(N), Ne_test) for i in range(Ns)])
        else:
            training_indices = np.array([
                random.sample(range(N), Ne) for i in range(Ns)])
            test_indices = [None]*Ns
    return training_indices, test_indices


def load_dataset(settings):
    data_settings = settings['data']

    dtype = data_settings['type']

    if dtype == 'file':
        instance, N, Ni = file_instance(data_settings)
    elif dtype == 'split':
        instance, N, Ni, Nt = split_instance(data_settings)
    elif dtype == 'generated':
        raise ValueError()
        # instance, N, Ni = generated_instance(data_settings)

    # check for problematic case
    problematic = (dtype == 'split' and
                   settings['sampling']['type'] == 'blank' and
                   'test' not in settings['sampling'])
    if problematic:
        raise ValueError('Cannot use implicit test sampling with split data.')

    # Only handle sampling if necessary
    if dtype == 'split':
        if settings['sampling']['type'] == 'blank':
            return [instance]
        else:
            training_indices, test_indices = load_samples(
                settings['sampling'], N, Ni, Nt)
    else:
        training_indices, test_indices = load_samples(
            settings['sampling'], N, Ni)

    contexts = []
    for trg, test in zip(training_indices, test_indices):
        context = instance.copy()
        context.update({'training_indices': trg, 'test_indices': test})
        contexts.append(context)
    return contexts


def file_instance(params):
    filename = build_filename(params, '.npz')
    with np.load(filename) as dataset:
        # Ne in the dataset is the full number of examples which we are
        # referring to herein as 'N' to differentiate from the sample size
        N = dataset['Ne']
        Ni = dataset['Ni']
        No = dataset['matrix'].shape[0] - Ni

    instance = {'type': 'file_unsplit',
                'file': filename}

    if 'targets' in params:
        targets = params['targets']
        if targets == 'random':
            # create a random permutation of size No
            targets = np.random.permutation(No)
            params['targets'] = targets
        instance['targets'] = targets

    return instance, N, Ni


def split_instance(params):
    trg_filename = build_filename(params, '.npz', key='training_filename')
    test_filename = build_filename(params, '.npz', key='test_filename')
    with np.load(trg_filename) as container:
        Ne_trg = container['Ne']
        Ni = container['Ni']
        No = container['matrix'].shape[0] - Ni
    with np.load(test_filename) as container:
        Ne_test = container['Ne']
        assert Ni == container['Ni']
        assert Ni + No == container['matrix'].shape[0]

    instance = {'type': 'file_split',
                'trg_file': trg_filename,
                'test_file': test_filename}

    if 'targets' in params:
        targets = params['targets']
        if targets == 'random':
            # create a random permutation of size No
            targets = np.random.permutation(No)
            params['targets'] = targets
        instance['targets'] = targets

    return instance, Ne_trg, Ni, Ne_test


# def generated_instance(params):
#     Nb = params['bits']
#     operator = params['operator']

#     No = params.get('out_width', Nb)  # defaults to operand width
#     targets = params.get('targets', None)
#     if targets == 'random':
#         # create a random permutation of size No
#         targets = np.random.permutation(No)
#         params['targets'] = targets

#     instance = {
#         'type': 'operator',
#         'operator': operator,
#         'Nb': Nb,
#         'No': No,
#         'targets': targets
#     }
#     Ni = opit.num_operands[operator] * Nb

#     return instance, 2**Ni, Ni


# to signal schema validation failure
# (with custom message formatting)
class ValidationError(Exception):
    pass


def validate_schema(config, schema, config_num, msg):
    try:
        schema(config)
    except Invalid as err:
        msg = ('Experiment instance {} invalid: {}'
               '\nConfig generation aborted.').format(config_num + 1, err)
        raise ValidationError(msg)


def update_nested(d, u):
    ''' this updates a dict with another where the two may contain nested
        dicts themselves (or more generally nested mutable mappings). '''
    for k, v in u.items():
        if isinstance(v, MutableMapping):
            r = update_nested(d.get(k, {}), v)
            d[k] = r
        else:
            # preference to second mapping if k exists in d
            d[k] = u[k]
    return d


def split_variables_from_base(settings):
    # configuration sub-dicts are popped
    try:
        variable_sets = settings['list']
    except KeyError:
        try:
            # build merged mappings for each combination
            lists = settings['product']
            variable_sets = []
            for combination in product(*lists):
                merged = dict()
                for var_set in combination:
                    update_nested(merged, var_set)
                variable_sets.append(merged)
        except KeyError:
            print('Warning: no variable configuration found.\n')
            variable_sets = [{}]

    return variable_sets, settings['base_config']


def generate_configurations(settings, batch_mode):
    # validate the given schema
    try:
        sch.experiment_schema(settings)
    except Invalid as err:
        raise ValidationError(
            'Top-level config invalid: {}'.format(err))

    # the configurations approach involves having a multiple config dicts and
    # updating them with each element of the configurations list or product
    variable_sets, base_settings = split_variables_from_base(settings)

    # Build up the configuration list
    configurations = []

    if not batch_mode:
        bar = utils.BetterETABar('Generating configurations',
                                 max=len(variable_sets))
        bar.update()
    try:
        for conf_num, variables in enumerate(variable_sets):
            # keep contexts isolated
            context = deepcopy(base_settings)
            # update the settings dict with the values for this configuration
            update_nested(context, variables)
            # check the given config is a valid experiment
            validate_schema(context, sch.instance_schema, conf_num, variables)
            # record the config number for debugging
            context['configuration_number'] = conf_num
            # load the data for this configuration
            instances = load_dataset(context)

            configurations.append((context, instances))
            if not batch_mode:
                bar.next()
    finally:
        # clean up progress bar before printing anything else
        if not batch_mode:
            bar.finish()
    return configurations


def generate_tasks(configurations, batch_mode):
    # Build up the task list
    tasks = []

    if not batch_mode:
        bar = utils.BetterETABar('Generating training tasks',
                                 max=len(configurations))
        bar.update()
    try:
        for context, instances in configurations:
            # for each sample
            for i, instance in enumerate(instances):
                task = deepcopy(context)
                task['mapping'] = instance
                task['training_set_number'] = i
                tasks.append(task)
            if not batch_mode:
                bar.next()
    finally:
        # clean up progress bar before printing anything else
        if not batch_mode:
            bar.finish()
    return tasks
