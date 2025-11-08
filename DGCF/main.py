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
flags.DEFINE_integer('epochs', 50, 'training epochs')


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
    # Print effective dynamic settings so it's obvious at runtime whether AGate is used
    print(f"Dataset: {FLAGS.data_name} | use_agate: {bool(data_hparams.get('use_agate', False))} | carry_alpha: {data_hparams.get('carry_alpha', None)}")

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
        gate_means = []

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
                model.prev_user_state.copy_(last_user_state.detach())
                model.prev_movie_state.copy_(last_movie_state.detach())

            # collect gate statistics if AGate is enabled
            if getattr(model, '_use_agate', False):
                try:
                    u_gate = model._agate_user.last_gate
                    m_gate = model._agate_movie.last_gate
                    # mean over nodes and dimensions
                    mean_gate = float((u_gate.mean() + m_gate.mean()) / 2.0)
                    gate_means.append(mean_gate)
                except Exception:
                    # last_gate may not exist in some code paths
                    pass

            epoch_loss += loss.item()
            snapshot_count += 1

        scheduler.step()

        avg_loss = epoch_loss / snapshot_count
        print(f"Epoch {epoch+1:04d} | Training Loss: {avg_loss:.6f}")

        # print gate usage statistics
        if len(gate_means) > 0:
            print(f"Epoch {epoch+1:04d} | mean AGate activation: {np.mean(gate_means):.4f}")

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

        # ----- Ranking evaluation: Recall@K -----
        # Build train and test interaction maps per user
        train_user_list = train_user.cpu().numpy()
        train_item_list = train_movie.cpu().numpy()
        test_user_list = test_user.cpu().numpy()
        test_item_list = test_movie.cpu().numpy()

        from collections import defaultdict

        train_by_user = defaultdict(set)
        for u, i in zip(train_user_list, train_item_list):
            train_by_user[int(u)].add(int(i))

        test_by_user = defaultdict(set)
        for u, i in zip(test_user_list, test_item_list):
            test_by_user[int(u)].add(int(i))

        ks = [5, 10, 20]
        recall_at_k = {k: [] for k in ks}

        # For each user in the test set, score all items and compute Recall@K
        all_items = torch.arange(num_movies, device=device, dtype=torch.int64)

        users_to_eval = sorted(test_by_user.keys())
        for u in users_to_eval:
            # score all items for user u
            user_ids = torch.full((num_movies,), u, dtype=torch.int64, device=device)
            with torch.no_grad():
                scores, _, _ = model(eval_user_adj, eval_movie_adj, user_ids, all_items)

            scores = scores.cpu().numpy()

            # mask training items so we don't recommend seen items
            train_items = train_by_user.get(u, set())
            mask = np.array([False] * num_movies)
            if train_items:
                mask[list(train_items)] = True
            scores[mask] = -np.inf

            # get top-k
            ranked_items = np.argsort(-scores)

            relevant = test_by_user[u]
            if len(relevant) == 0:
                continue

            for k in ks:
                topk = set(ranked_items[:k])
                hits = len(topk & relevant)
                recall = hits / float(len(relevant))
                recall_at_k[k].append(recall)

        # Report averaged Recall@K
        for k in ks:
            vals = recall_at_k[k]
            mean_recall = float(np.mean(vals)) if len(vals) > 0 else 0.0
            print(f"Recall@{k}: {mean_recall:.6f}")

if __name__ == '__main__':
    app.run(main)
