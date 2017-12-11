try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='nn-target-curricula',
    scripts=['scripts/nn_prepare_experiment.py',
             'scripts/nn_run_experiments.py']
    )
