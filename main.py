import numpy as np
import random
import torch

from torch.backends import cudnn
from absl import app, flags

from datasets import ML1M, ML100K, Flixster, Douban, YahooMusic
from model import GCCF
from hyperparameters import hparams
from utils import get_adj

cudnn.deterministic = True
cudnn.benchmark = False

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device('cpu')  # Force CPU usage due to GPU compatibility issues

FLAGS = flags.FLAGS
flags.DEFINE_string('data_name', '', 'dataset name')
flags.DEFINE_string('root_dir', '', 'dataset directory path')


def main(argv):

    if FLAGS.data_name == 'ml-1m':
        dataset = ML1M(FLAGS.root_dir, device)
    elif FLAGS.data_name == 'ml-100k':
        dataset = ML100K(FLAGS.root_dir, device)
    elif FLAGS.data_name == 'flixster':
        dataset = Flixster(FLAGS.root_dir, device)
    elif FLAGS.data_name == 'douban':
        dataset = Douban(FLAGS.root_dir, device)
    elif FLAGS.data_name == 'yahoo_music':
        dataset = YahooMusic(FLAGS.root_dir, device)
    else:
        raise Exception

    data_hparams = hparams[FLAGS.data_name]

    train_user, train_movie, train_rating = dataset.get_train_data()
    test_user, test_movie, test_rating = dataset.get_test_data()

    num_users = dataset.get_num_users()
    num_movies = dataset.get_num_movies()

    user_adj = get_adj(num_users, num_movies, train_user, train_movie, device)
    movie_adj = get_adj(num_movies, num_users, train_movie, train_user, device)

    epochs = 1000

    model = GCCF(num_users, num_movies, data_hparams).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=data_hparams["lr"], weight_decay=data_hparams["weight_decay"])

    min_test_loss = 999.

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        predict = model(user_adj, movie_adj, train_user, train_movie)
        loss = criterion(predict, train_rating)

        loss.backward()
        optimizer.step()

        with torch.no_grad():

            model.eval()
            test_predict = model(user_adj, movie_adj, test_user, test_movie)
            test_loss = criterion(dataset.inverse_transform(test_predict), dataset.inverse_transform(test_rating))

            if min_test_loss > test_loss:
                min_test_loss = test_loss

        print('Epoch {:04d}, Train Loss: {:.6f}, Test Loss: {:.6f}'.format(epoch+1, loss.item(), test_loss.item()))

    print('Min Test Loss: {:.6f}'.format(min_test_loss))


if __name__ == '__main__':
    app.run(main)
