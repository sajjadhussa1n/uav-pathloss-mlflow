# src/evaluator.py

import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.tensorflow
import tensorflow as tf

# -------------------------
# Helper functions
# -------------------------

def generate_random_chunks(full_grid, chunk_size=128, num_chunks=100):
    H, W, C = full_grid.shape
    chunks, chunk_info = [], []
    used_combinations = set()

    systematic_patterns = [
        {'i_strides': (128, 1), 'j_strides': (128, 1)},  # 3x2 grid
        {'i_strides': (128, 1), 'j_strides': (256, 2)},  # 3x1 grid
        {'i_strides': (256, 2), 'j_strides': (128, 1)},  # 2x2 grid
        {'i_strides': (256, 2), 'j_strides': (256, 2)},  # 2x1 grid
        {'i_strides': (384, 3), 'j_strides': (128, 1)},  # 1x2 grid
        {'i_strides': (384, 3), 'j_strides': (256, 2)},  # 1x1 grid
    ]

    for pattern in systematic_patterns:
        i_step, i_stride = pattern['i_strides']
        j_step, j_stride = pattern['j_strides']
        for i in range(0, H - (chunk_size * i_stride) + 1, i_step):
            for j in range(0, W - (chunk_size * j_stride) + 1, j_step):
                window = full_grid[i:i+chunk_size*i_stride:i_stride,
                                   j:j+chunk_size*j_stride:j_stride, :]
                chunks.append(window)
                chunk_info.append((i, j, i_stride, j_stride))

    total_runs = len(chunks)
    while total_runs < num_chunks:
        i_stride = np.random.choice([1, 2, 3])
        j_stride = np.random.choice([1, 2])
        i = np.random.randint(0, H - chunk_size * i_stride + 1)
        j = np.random.randint(0, W - chunk_size * j_stride + 1)
        combination = (i, j, i_stride, j_stride)
        if combination in used_combinations:
            continue
        used_combinations.add(combination)
        window = full_grid[i:i+chunk_size*i_stride:i_stride,
                           j:j+chunk_size*j_stride:j_stride, :]
        chunks.append(window)
        chunk_info.append((i, j, i_stride, j_stride))
        total_runs += 1

    return np.array(chunks), chunk_info


def aggregate_predictions(full_grid_shape, predictions, chunk_info):
    H, W = full_grid_shape[:2]
    sum_grid = np.zeros((H, W), dtype=np.float32)
    count_grid = np.zeros((H, W), dtype=np.float32)

    for (pred, (i, j, i_stride, j_stride)) in zip(predictions, chunk_info):
        for x in range(pred.shape[0]):
            for y in range(pred.shape[1]):
                true_i = i + x * i_stride
                true_j = j + y * j_stride
                if true_i < H and true_j < W:
                    sum_grid[true_i, true_j] += pred[x, y, 0]
                    count_grid[true_i, true_j] += 1

    count_grid[count_grid == 0] = 1
    return sum_grid / count_grid


# -------------------------
# Evaluator class
# -------------------------

class Evaluator:
    def __init__(self, run_id, global_mins, global_maxs, ny=384, nx=256):
        """
        run_id: MLflow run ID where the best model was logged
        global_mins/maxs: list of min/max for normalization
        """
        self.run_id = run_id
        self.global_mins = global_mins
        self.global_maxs = global_maxs
        self.ny, self.nx = ny, nx

        # Load model from MLflow
        model_uri = f"runs:/{run_id}/model"
        print(f"Loading best model from {model_uri}")
        self.model = mlflow.tensorflow.load_model(model_uri)

    def evaluate(self, test_dir, metrics_fn):
        """
        test_dir: directory containing test CSVs
        metrics_fn: dict of { "rmse": func, "mae": func, "nmse": func }
        """
        RMSE, MAE, NMSE = [], [], []
        PL_actual, PL_pred = [], []

        files = [f for f in os.listdir(test_dir) if f.endswith(".csv")]

        for fname in files:
            df = pd.read_csv(os.path.join(test_dir, fname))

            # Normalize input
            dist = df['Distance_3d'].values.reshape(self.ny, self.nx)
            dist_log = 20.0 * np.log10(dist)
            dist_norm = (dist_log - self.global_mins[0]) / (self.global_maxs[0] - self.global_mins[0])
            los = df['LOS_mask'].values.reshape(self.ny, self.nx)
            bmask = df['Is_building'].values.reshape(self.ny, self.nx)
            pl = df['Path_loss'].values.reshape(self.ny, self.nx)
            pl_norm = (pl - self.global_mins[3]) / (self.global_maxs[3] - self.global_mins[3])

            full_grid = np.stack([dist_norm, los, bmask], axis=-1)

            # Predict
            chunks, chunk_info = generate_random_chunks(full_grid, num_chunks=100)
            predictions = self.model.predict(chunks)
            final_pred = aggregate_predictions(full_grid.shape, predictions, chunk_info)

            # Compute metrics
            RMSE.append(metrics_fn["rmse"](pl_norm, final_pred))
            MAE.append(metrics_fn["mae"](pl_norm, final_pred))
            NMSE.append(metrics_fn["nmse"](pl_norm, final_pred))

            # Denormalize prediction for saving
            final_pred = final_pred * (self.global_maxs[3] - self.global_mins[3]) + self.global_mins[3]
            PL_actual.append(pl)
            PL_pred.append(final_pred)

            print(f"[{fname}] RMSE={RMSE[-1]:.4f}, MAE={MAE[-1]:.4f}, NMSE={NMSE[-1]:.4f}")

            del df, dist, dist_norm, los, bmask, pl, pl_norm, full_grid, chunks, chunk_info, predictions, final_pred

        print("\n==== Final Averages ====")
        print(f"Mean RMSE: {np.mean(RMSE):.4f}")
        print(f"Mean MAE : {np.mean(MAE):.4f}")
        print(f"Mean NMSE: {np.mean(NMSE):.4f}")

        return {
            "RMSE": RMSE, "MAE": MAE, "NMSE": NMSE,
            "Mean_RMSE": np.mean(RMSE),
            "Mean_MAE": np.mean(MAE),
            "Mean_NMSE": np.mean(NMSE),
            "PL_actual": PL_actual,
            "PL_pred": PL_pred
        }
