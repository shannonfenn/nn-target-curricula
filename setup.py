try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='nn-target-curricula',
    scripts=['scripts/prepare_nn_experiment.py',
             'scripts/run_nn_experiments.py']
    )
