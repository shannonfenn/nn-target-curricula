from good import (
    Schema, All, Any, Range, IsDir, Allow, Match,
    Msg, Optional, Exclusive, Length, Entire)


seed_schema = Any(None, str, All(int, Range(min=0)))

target_subset_schema = Any('random', [All(int, Range(min=0))])

data_schema = Any(
    # generated from operator
    # Schema({
    #     'type':                     'generated',
    #     'operator':                 str,
    #     'bits':                     All(int, Range(min=1)),
    #     Optional('out_width'):      All(int, Range(min=1)),
    #     Optional('add_noise'):      Range(min=0.0),
    #     Optional('targets'):        target_subset_schema,
    #     }),
    # read from file
    Schema({
        'type':                 'file',
        'filename':             str,
        Optional('dir'):        IsDir(),
        Optional('add_noise'):  Range(min=0.0),
        Optional('targets'):    target_subset_schema,
        }),
    # pre-split, read from file
    Schema({
        'type':                 'split',
        'training_filename':    str,
        'test_filename':        str,
        Optional('dir'):        IsDir(),
        Optional('add_noise'):  Range(min=0.0),
        Optional('targets'):    target_subset_schema,
        })
    )

sampling_schema = Any(
    # randomly generated
    Schema({
        'type':           'generated',
        'Ns':             All(int, Range(min=1)),
        'Ne':             All(int, Range(min=1)),
        'seed':           seed_schema,
        Optional('test'): All(int, Range(min=0))
        }),
    # read from file
    Schema({
        'type':             'file',
        'filename':         str,
        Optional('test'):   str,
        Optional('dir'):    IsDir(),
        # allow for now, but don't force
        Optional('seed'):   seed_schema,
        }),
    # given in config file
    Schema({
        'type':             'given',
        'indices':          [[All(int, Range(min=0))]],
        Optional('test'):   [[All(int, Range(min=0))]],
        # allow for now, but don't force
        Optional('seed'):   seed_schema,
        }),
    # blank - data is already split
    Schema({'type': 'blank'})
    )

minfs_params_schema = Any(
    # CPLEX
    Schema({
        Optional('time_limit'):       Range(min=0.0),
        }),
    # Meta-RaPS
    Schema({
        Optional('iterations'):             All(int, Range(min=1)),
        Optional('improvement_iterations'): All(int, Range(min=1)),
        Optional('search_magnitude'):       All(float, Range(min=0, max=1)),
        Optional('priority'):               All(float, Range(min=0, max=1)),
        Optional('restriction'):            All(float, Range(min=0, max=1)),
        Optional('improvement'):            All(float, Range(min=0, max=1))
        }),
    )

architecture_schema = Schema({
    'name': Any('shared_a', 'shared_b', 'shared_c', 'parallel', 'layered'),
    'Nh':                       int,
    'nonlinearity':             str,
    Optional('regularizer'):    Schema({}, extra_keys=Allow),
    Optional('params'):         Schema({}, extra_keys=Allow)
    })

compile_schema = Schema({
    'optimizer':    str,
    'loss':         str,
    }, extra_keys=Allow)

fit_schema = Schema({'epochs': int}, extra_keys=Allow)

instance_schema = Schema({
    'data':         data_schema,
    'sampling':     sampling_schema,
    'architecture': architecture_schema,
    'compile':      compile_schema,
    'fit':          fit_schema,
    'batch_ratio':  float,
    Optional(Match(r'notes.*')):  str,
    })


# ########## Schemata for base configs ########## #
list_msg = '\'list\' must be a sequence of mappings.'
prod_msg = '\'product\' must be a sequence of sequences of mappings.'
experiment_schema = Schema({
    'name':                     str,
    # Must be any dict
    'base_config':              Schema({}, extra_keys=Allow),
    # only one of 'list_config' or 'product_config' are allowed
    # Must be a list of dicts
    Optional('list'):    Msg(All(Schema([Schema({}, extra_keys=Allow)]),
                                 Length(min=1)), list_msg),
    # Must be a length >= 2 list of lists of dicts
    Optional('product'): Msg(All([All(Schema([Schema({}, extra_keys=Allow)]),
                                      Length(min=1))],
                                 Length(min=2)), prod_msg),
    Entire: Exclusive('list', 'product')

    })
