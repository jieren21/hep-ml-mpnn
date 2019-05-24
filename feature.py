""" Construction of Event Graphs. (Internal version)

Author: Jie Ren <renjie@itp.ac.cn>
Last modified: Nov 4, 2018

Dependences:
1. Python 3 (>=3.5)
2. numpy
3. h5py
4. xhep (Internal code)

Please cite our paper arXiv:1807.09088 [hep-ph].

Disclaimer: this program is an internal version which comes without any guarantees.
"""

import sys
import numpy as np
import h5py
from xhep import Vec4


def build_features(in_file, start, end):
    events = h5py.File(in_file, 'r')

    features = np.empty([end - start, 21, 7], np.float32)
    distances = np.empty([end - start, 21, 21], np.float32)
    masks = np.zeros([end - start, 21], np.float32)

    for i in range(end - start):
        if i % 1000 == 0:
            print(in_file, i, end='          \r')

        photons = events['Photon/Momentum'][start + i].reshape(-1, 4)
        leptons = events['Lepton/Momentum'][start + i].reshape(-1, 4)
        lepton_charges = events['Lepton/Charge'][start + i]
        jets = events['Jet/Momentum'][start + i].reshape(-1, 4)
        jet_btags = events['Jet/BTag'][start + i]
        missET = events['MissET/Momentum'][start + i]

        n = len(photons) + len(leptons) + len(jets) + 1

        p4 = []

        j = 0
        for k in range(len(photons)):
            p = Vec4(*photons[k])
            p4.append(p)
            features[i, j] = [1, 0, 0, 0, p.pt / 1000, p.E / 1000, p.m / 1000]
            j += 1
        for k in range(len(leptons)):
            p = Vec4(*leptons[k])
            p4.append(p)
            features[i, j] = [0, lepton_charges[k], 0, 0, p.pt / 1000, p.E / 1000, p.m / 1000]
            j += 1
        for k in range(len(jets)):
            p = Vec4(*jets[k])
            p4.append(p)
            features[i, j] = [0, 0, 1 if jet_btags[k] == 1 else -1, 0, p.pt / 1000, p.E / 1000, p.m / 1000]
            j += 1
        p = Vec4(*missET)
        p4.append(p)
        features[i, j] = [0, 0, 0, 1, p.pt / 1000, p.E / 1000, p.m / 1000]

        masks[i, 0:n] = 1

        for j in range(n):
            for k in range(n):
                distances[i, j, k] = Vec4.delta_R_rapidity(p4[j], p4[k])

    return features, distances, masks


S_FILE = 'samples/stop.h5'
B_FILE = 'samples/tt.h5'
O_FILE = 'input.h5'
N_TRAIN = 200000
N_VALID = 200000

f = h5py.File(O_FILE, 'w')

# training set
print('training set')
features1, distances1, masks1 = build_features(S_FILE, 0, N_TRAIN)
features2, distances2, masks2 = build_features(B_FILE, 0, N_TRAIN)
features = np.concatenate([features1, features2])
distances = np.concatenate([distances1, distances2])
masks = np.concatenate([masks1, masks2])
targets = np.concatenate([np.full([N_TRAIN, 1], 1, np.float32), np.full([N_TRAIN, 1], 0, np.float32)])
g = f.create_group('Train')
g.create_dataset('Feature', data=features, compression='gzip')
g.create_dataset('Distance', data=distances, compression='gzip')
g.create_dataset('Mask', data=masks, compression='gzip')
g.create_dataset('Target', data=targets, compression='gzip')

# validation set
print('validation set')
features1, distances1, masks1 = build_features(S_FILE, N_TRAIN, N_TRAIN + N_VALID)
features2, distances2, masks2 = build_features(B_FILE, N_TRAIN, N_TRAIN + N_VALID)
features = np.concatenate([features1, features2])
distances = np.concatenate([distances1, distances2])
masks = np.concatenate([masks1, masks2])
targets = np.concatenate([np.full([N_VALID, 1], 1, np.float32), np.full([N_VALID, 1], 0, np.float32)])
g = f.create_group('Validation')
g.create_dataset('Feature', data=features, compression='gzip')
g.create_dataset('Distance', data=distances, compression='gzip')
g.create_dataset('Mask', data=masks, compression='gzip')
g.create_dataset('Target', data=targets, compression='gzip')
