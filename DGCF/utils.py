import scipy.sparse as sp
import pandas as pd
import numpy as np
import torch
import h5py


def get_adj(num_rows, num_cols, row_idx, col_idx, device):

    adj = torch.zeros((num_rows, num_cols), dtype=torch.float32, device=device)
    adj[row_idx, col_idx] = 1.

    adj = adj / adj.sum(dim=1, keepdim=True)
    adj.masked_fill_(torch.isnan(adj), 0)

    return adj


def load_matlab_file(path_file, name_field):

    db = h5py.File(path_file, 'r')
    ds = db[name_field]

    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds['ir'])
            jc = np.asarray(ds['jc'])
            out = sp.csc_matrix((data, ir, jc))
    except AttributeError:
        out = np.asarray(ds).T

    db.close()

    return out.astype(int)


def matrix2data(matrix, rating):

    idx = np.argwhere(matrix > 0)

    rows = idx[:, 0]
    columns = idx[:, 1]
    ratings = rating[rows, columns].reshape(-1, 1)

    data = np.concatenate([idx, ratings], axis=1)
    data = pd.DataFrame(data, columns=('user', 'movie', 'rating'))

    return data
