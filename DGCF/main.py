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

device = torch.device('cpu')

FLAGS = flags.FLAGS
flags.DEFINE_string('data_name', '', 'dataset name')
flags.DEFINE_string('root_dir', '', 'dataset directory path')
flags.DEFINE_integer('window_days', 30, 'snapshot window size in days')
flags.DEFINE_integer('epochs', 100, 'training epochs')


def main(argv):

    # ---- Dataset ----
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
        raise Exception("Unknown dataset")

    data_hparams = hparams[FLAGS.data_name]

    train_user, train_movie, _ = dataset.get_train_data()
    test_user, test_movie, test_rating = dataset.get_test_data()

    num_users = dataset.get_num_users()
    num_movies = dataset.get_num_movies()

    # ---- Build snapshots ----
    if FLAGS.data_name == 'ml-1m':
        snapshots = dataset.get_train_snapshots(window_size_days=FLAGS.window_days)
    else:
        snapshots = [(train_user, train_movie, dataset.get_train_data()[2])]

    # ---- Model / Optimizer ----
    model = GCCF(num_users, num_movies, data_hparams).to(device)
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=data_hparams["lr"] * 0.25,
                                 weight_decay=data_hparams["weight_decay"])

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    epochs = FLAGS.epochs

    # ---- Training Loop ----
    for epoch in range(epochs):
        model.train()

        epoch_loss = 0.0
        snapshot_count = 0

        for (snap_user, snap_movie, snap_rating) in snapshots:
            user_adj = get_adj(num_users, num_movies, snap_user, snap_movie, device)
            movie_adj = get_adj(num_movies, num_users, snap_movie, snap_user, device)

            optimizer.zero_grad()
            # Model now returns prediction plus the final propagated
            # user/movie states for the snapshot so we can persist them.
            predict, last_user_state, last_movie_state = model(user_adj, movie_adj, snap_user, snap_movie)
            loss = criterion(predict, snap_rating)
            loss.backward()
            optimizer.step()

            # Persist propagated states into model buffers (no grads)
            with torch.no_grad():
                # Copy latest propagated full-node states into buffers
                model.prev_user_state.copy_(last_user_state)
                model.prev_movie_state.copy_(last_movie_state)

            epoch_loss += loss.item()
            snapshot_count += 1

        scheduler.step()

        avg_loss = epoch_loss / snapshot_count
        print(f"Epoch {epoch+1:04d} | Training Loss: {avg_loss:.6f}")

    # ---- Final Evaluation ----
    with torch.no_grad():
        model.eval()

        eval_user_adj = get_adj(num_users, num_movies, train_user, train_movie, device)
        eval_movie_adj = get_adj(num_movies, num_users, train_movie, train_user, device)

        test_predict, _, _ = model(eval_user_adj, eval_movie_adj, test_user, test_movie)

        test_pred_actual = dataset.inverse_transform(test_predict)
        test_true_actual = dataset.inverse_transform(test_rating)

        final_loss = criterion(test_pred_actual, test_true_actual)
        final_rmse = torch.sqrt(final_loss)

        print("\n========= FINAL METRICS =========")
        print(f"Test Loss: {final_loss.item():.6f}")
        print(f"RMSE:      {final_rmse.item():.6f}")
        print("=================================\n")


if __name__ == '__main__':
    app.run(main)
