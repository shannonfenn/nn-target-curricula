import numpy as np
import json
import minfs.feature_selection as fss
from progress.bar import IncrementalBar


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
