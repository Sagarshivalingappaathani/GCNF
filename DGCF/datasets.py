import pandas as pd
import torch
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import load_matlab_file, matrix2data


class ML1M:
    """
    ML-1M with time-ordered split + snapshot helpers.
    """
    def __init__(self, root_dir, device):
        data_path = os.path.join(root_dir, 'ratings.dat')
        user_info_path = os.path.join(root_dir, 'users.dat')
        movie_info_path = os.path.join(root_dir, 'movies.dat')

        self._data = pd.read_csv(
            data_path,
            sep='::',
            names=['user', 'movie', 'rating', 'time'],
            engine='python'
        )
        self._user_info = pd.read_csv(
            user_info_path,
            sep='::',
            names=['id', 'gender', 'age', 'occupation', 'zip-code'],
            engine='python'
        )
        self._movie_info = pd.read_csv(
            movie_info_path,
            sep='::',
            names=['id', 'title', 'genres'],
            engine='python',
            encoding='iso-8859-1'
        )

        # Label-encode user/movie ids to 0..N-1
        user_le = LabelEncoder()
        movie_le = LabelEncoder()
        self._user_info['id'] = user_le.fit_transform(self._user_info['id'])
        self._movie_info['id'] = movie_le.fit_transform(self._movie_info['id'])
        self._data['user'] = user_le.transform(self._data['user'])
        self._data['movie'] = movie_le.transform(self._data['movie'])

        # ---- Time-ordered train/test split (no leakage) ----
        self._data = self._data.sort_values(by='time').reset_index(drop=True)
        num_records = len(self._data)
        train_end = int(num_records * 0.8)  # 80% earliest for train, 20% latest for test
        self._train_df = self._data.iloc[:train_end].copy()
        self._test_df = self._data.iloc[train_end:].copy()

        self._device = device

    def _split_user_movie_rating(self, df):
        user = torch.tensor(df['user'].values, dtype=torch.int64, device=self._device)
        movie = torch.tensor(df['movie'].values, dtype=torch.int64, device=self._device)
        rating = torch.tensor(df['rating'].values, dtype=torch.float32, device=self._device) / 5.0
        return user, movie, rating

    # ---- Original API (kept) ----
    def get_train_data(self):
        return self._split_user_movie_rating(self._train_df)

    def get_test_data(self):
        return self._split_user_movie_rating(self._test_df)

    def get_num_users(self):
        return int(self._user_info['id'].max()) + 1

    def get_num_movies(self):
        return int(self._movie_info['id'].max()) + 1

    @staticmethod
    def inverse_transform(values):
        return values * 5.0

    # ---- Snapshot helpers (NEW) ----
    def get_train_snapshots(self, window_size_days=30):
        """
        Returns a list of (user, movie, rating) tensors per time window from TRAIN data.
        """
        df = self._train_df.copy()
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['snapshot'] = df['time'].dt.to_period(f'{window_size_days}D')

        snapshots = []
        for _, g in df.groupby('snapshot'):
            if len(g) == 0:
                continue
            snapshots.append(self._split_user_movie_rating(g))
        return snapshots

    def get_full_train_edges(self):
        """
        Returns (user_idx, movie_idx) tensors for ALL train interactions — used to
        build evaluation adjacencies when predicting on test.
        """
        u = torch.tensor(self._train_df['user'].values, dtype=torch.int64, device=self._device)
        i = torch.tensor(self._train_df['movie'].values, dtype=torch.int64, device=self._device)
        return u, i


class ML100K:
    def __init__(self, root_dir, device):
        data_path = os.path.join(root_dir, 'split_1.mat')
        rating = load_matlab_file(data_path, 'M')
        training = load_matlab_file(data_path, 'Otraining')
        test = load_matlab_file(data_path, 'Otest')
        self._num_users = rating.shape[0]
        self._num_movies = rating.shape[1]
        self._train_data = matrix2data(training, rating)
        self._test_data = matrix2data(test, rating)
        self._device = device

    def _split_user_movie_rating(self, data):
        user = torch.tensor(data['user'].values, dtype=torch.int64, device=self._device)
        movie = torch.tensor(data['movie'].values, dtype=torch.int64, device=self._device)
        rating = torch.tensor(data['rating'].values, dtype=torch.float32, device=self._device) / 5.
        return user, movie, rating

    def get_train_data(self):
        return self._split_user_movie_rating(self._train_data)

    def get_test_data(self):
        return self._split_user_movie_rating(self._test_data)

    def get_num_users(self):
        return self._num_users

    def get_num_movies(self):
        return self._num_movies

    @staticmethod
    def inverse_transform(values):
        return values * 5


class Flixster:
    def __init__(self, root_dir, device):
        data_path = os.path.join(root_dir, 'training_test_dataset_10_NNs.mat')
        rating = load_matlab_file(data_path, 'M')
        training = load_matlab_file(data_path, 'Otraining')
        test = load_matlab_file(data_path, 'Otest')
        self._num_users = rating.shape[0]
        self._num_movies = rating.shape[1]
        self._train_data = matrix2data(training, rating)
        self._test_data = matrix2data(test, rating)
        self._device = device

    def _split_user_movie_rating(self, data):
        user = torch.tensor(data['user'].values, dtype=torch.int64, device=self._device)
        movie = torch.tensor(data['movie'].values, dtype=torch.int64, device=self._device)
        rating = torch.tensor(data['rating'].values, dtype=torch.float32, device=self._device) / 5.
        return user, movie, rating

    def get_train_data(self):
        return self._split_user_movie_rating(self._train_data)

    def get_test_data(self):
        return self._split_user_movie_rating(self._test_data)

    def get_num_users(self):
        return self._num_users

    def get_num_movies(self):
        return self._num_movies

    @staticmethod
    def inverse_transform(values):
        return values * 5


class Douban:
    def __init__(self, root_dir, device):
        data_path = os.path.join(root_dir, 'training_test_dataset.mat')
        rating = load_matlab_file(data_path, 'M')
        training = load_matlab_file(data_path, 'Otraining')
        test = load_matlab_file(data_path, 'Otest')
        self._num_users = rating.shape[0]
        self._num_movies = rating.shape[1]
        self._train_data = matrix2data(training, rating)
        self._test_data = matrix2data(test, rating)
        self._device = device

    def _split_user_movie_rating(self, data):
        user = torch.tensor(data['user'].values, dtype=torch.int64, device=self._device)
        movie = torch.tensor(data['movie'].values, dtype=torch.int64, device=self._device)
        rating = torch.tensor(data['rating'].values, dtype=torch.float32, device=self._device) / 5.
        return user, movie, rating

    def get_train_data(self):
        return self._split_user_movie_rating(self._train_data)

    def get_test_data(self):
        return self._split_user_movie_rating(self._test_data)

    def get_num_users(self):
        return self._num_users

    def get_num_movies(self):
        return self._num_movies

    @staticmethod
    def inverse_transform(values):
        return values * 5


class YahooMusic:
    def __init__(self, root_dir, device):
        data_path = os.path.join(root_dir, 'training_test_dataset_10_NNs.mat')
        rating = load_matlab_file(data_path, 'M')
        training = load_matlab_file(data_path, 'Otraining')
        test = load_matlab_file(data_path, 'Otest')
        self._num_users = rating.shape[0]
        self._num_movies = rating.shape[1]
        self._train_data = matrix2data(training, rating)
        self._test_data = matrix2data(test, rating)
        self._device = device

    def _split_user_movie_rating(self, data):
        user = torch.tensor(data['user'].values, dtype=torch.int64, device=self._device)
        movie = torch.tensor(data['movie'].values, dtype=torch.int64, device=self._device)
        rating = torch.tensor(data['rating'].values, dtype=torch.float32, device=self._device) / 100.
        return user, movie, rating

    def get_train_data(self):
        return self._split_user_movie_rating(self._train_data)

    def get_test_data(self):
        return self._split_user_movie_rating(self._test_data)

    def get_num_users(self):
        return self._num_users

    def get_num_movies(self):
        return self._num_movies

    @staticmethod
    def inverse_transform(values):
        return values * 100
