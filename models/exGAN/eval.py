import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import directed_hausdorff
from sklearn.neighbors import BallTree


def evaluate_on_batch(real_traj, gen_traj, batch_size):
    mask_real = real_traj[4]
    mask_gen = gen_traj[4]

    results = Parallel(n_jobs=10)(
        delayed(compute_hausdorff)(
            real_traj[0][i], mask_real[i], gen_traj[0][i], mask_gen[i]
        )
        for i in range(batch_size)
    )

    return np.array(results)


def evaluate_on_test(model, test_trajs):
    num_traj = test_trajs[0].shape[0]

    gen_traj_test = model.generator.predict(test_trajs)

    gen_traj_test_denorm = np.add(
        gen_traj_test[0][:], np.array([model.lat_centroid, model.lon_centroid])
    )
    real_traj_test_denorm = np.add(
        test_trajs[0][:], np.array([model.lat_centroid, model.lon_centroid])
    )

    # only use real mask
    mask_real = test_trajs[4]
    mask_gen = gen_traj_test[4]

    results = Parallel(n_jobs=10)(
        delayed(compute_hausdorff)(
            real_traj_test_denorm[i], mask_real[i], gen_traj_test_denorm[i], mask_gen[i]
        )
        for i in range(num_traj)
    )

    haus_dist = np.array(results)

    unique_real = get_unique_locations(
        extract_valid_coords(model.X_val[0], model.X_val[4]), radius_meters=100
    )
    unique_gen = get_unique_locations(
        extract_valid_coords(gen_traj_test[0], gen_traj_test[4]), radius_meters=100
    )
    tree_gen = BallTree(np.radians(unique_gen), metric="haversine")
    real_in_gen = tree_gen.query_radius(np.radians(unique_real), r=100 / 6371000.0)
    overlap_count = sum([len(x) > 0 for x in real_in_gen])
    recall = overlap_count / len(unique_real) if len(unique_real) > 0 else 0

    # traj_length = keras.backend.sum(mask_real,axis=1)

    mask_real_squeezed = np.squeeze(mask_real, axis=-1).astype(np.float32)

    mean_acc_day_value = np.sum(
        (
            np.argmax(gen_traj_test[1], axis=-1) == np.argmax(model.X_val[1], axis=-1)
        ).astype(np.float32)
        * mask_real_squeezed
    ) / np.sum(mask_real_squeezed)
    mean_acc_hour_value = np.sum(
        (
            np.argmax(gen_traj_test[2], axis=-1) == np.argmax(model.X_val[2], axis=-1)
        ).astype(np.float32)
        * mask_real_squeezed
    ) / np.sum(mask_real_squeezed)
    mean_acc_category_value = np.sum(
        (
            np.argmax(gen_traj_test[3], axis=-1) == np.argmax(model.X_val[3], axis=-1)
        ).astype(np.float32)
        * mask_real_squeezed
    ) / np.sum(mask_real_squeezed)

    return (
        haus_dist,
        mean_acc_day_value,
        mean_acc_hour_value,
        mean_acc_category_value,
        recall,
    )


def compute_hausdorff(real_traj, real_mask, gen_traj_test, gen_mask):
    real_traj_valid = real_traj[real_mask.flatten().astype(bool)]
    gen_traj_test_valid = gen_traj_test[gen_mask.flatten().astype(bool)]
    if len(real_traj_valid) > 0 and len(gen_traj_test_valid) > 0:
        return directed_hausdorff(real_traj_valid, gen_traj_test_valid)[0]


def extract_valid_coords(X, mask):
    coords = []

    for traj, m in zip(X, mask):
        valid_points = traj[m.flatten().astype(bool)]
        coords.append(valid_points)
    if coords:
        return np.vstack(coords)
    else:
        return np.empty((0, 2))


def get_unique_locations(coords, radius_meters=100):
    """
    coords: np.array of shape (N, 2), where each row is [lat, lon] in degrees
    radius_meters: radius for uniqueness, in meters
    Returns: np.array of unique locations
    """

    # Convert degrees to radians for BallTree
    coords_rad = np.radians(coords)
    # BallTree with haversine metric (earth radius = 6371000 m)
    tree = BallTree(coords_rad, metric="haversine")

    unique_indices = []

    visited = np.zeros(len(coords), dtype=bool)

    for i in range(len(coords)):
        if not visited[i]:
            # Query neighbors within radius
            ind = tree.query_radius([coords_rad[i]], r=radius_meters / 6371000.0)[0]
            visited[ind] = True
            unique_indices.append(i)

    return coords[unique_indices]
