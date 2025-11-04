import pandas as pd
import torch
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import load_matlab_file, matrix2data


class ML1M:

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

        user_label_encoder = LabelEncoder()
        movie_label_encoder = LabelEncoder()

        self._user_info['id'] = user_label_encoder.fit_transform(self._user_info['id'])
        self._movie_info['id'] = movie_label_encoder.fit_transform(self._movie_info['id'])
        self._data['user'] = user_label_encoder.transform(self._data['user'])
        self._data['movie'] = movie_label_encoder.transform(self._data['movie'])

        self._train_data, self._test_data = train_test_split(self._data, test_size=0.1)
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
        return max(self._user_info['id']) + 1

    def get_num_movies(self):
        return max(self._movie_info['id']) + 1

    @staticmethod
    def inverse_transform(values):
        return values * 5


class ML100K:

    def __init__(self, root_dir, device):

        data_path = os.path.join(root_dir, 'split_1.mat')

        rating = load_matlab_file(data_path, 'M')  # rating matrix
        training = load_matlab_file(data_path, 'Otraining')  # train matrix
        test = load_matlab_file(data_path, 'Otest')  # test matrix

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

        rating = load_matlab_file(data_path, 'M')  # rating matrix
        training = load_matlab_file(data_path, 'Otraining')  # train matrix
        test = load_matlab_file(data_path, 'Otest')  # test matrix

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

        rating = load_matlab_file(data_path, 'M')  # rating matrix
        training = load_matlab_file(data_path, 'Otraining')  # train matrix
        test = load_matlab_file(data_path, 'Otest')  # test matrix

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

        rating = load_matlab_file(data_path, 'M')  # rating matrix
        training = load_matlab_file(data_path, 'Otraining')  # train matrix
        test = load_matlab_file(data_path, 'Otest')  # test matrix

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
