import warnings

import hdbscan
import numpy as np
from sklearn.metrics import silhouette_score

from syntraj_utileval.datasets.dataset import TrajectoryDataset


def trajectory_clustering_prediction(
    original: TrajectoryDataset,
    syn: TrajectoryDataset,
    min_cluster_size=5,
    min_samples=None,
    cluster_selection_epsilon=0.0,
):
    """
    Evaluate function utility using the train on synthetic test on real protocol TSTR

    Returns
    -------
    results : list of dict
        Each dict contains statistics about a centroid:
        - centroid': cluster' centroids
        - 'size': cluster size
        - 'spread': cluster spread (variance)
    """

    # TRAIN ON SYNTH: Define the 'World View' using Synthetic Data

    syn_traj_centroid = syn.df_discretized.groupby("tid")[["x_m", "y_m"]].mean().values
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        prediction_data=True,  # prediction_data=True is mandatory for approximate_predict
        gen_min_span_tree=True,
    ).fit(syn_traj_centroid)

    syn_labels = clusterer.labels_
    num_syn_clusters = len(set(syn_labels)) - (1 if -1 in syn_labels else 0)

    if num_syn_clusters < 2:
        warnings.warn(
            f"Insufficient synthetic clusters found in dataset {syn.dataset_name}"
        )
        return {
            "HSU_Master": None,
            "Silhouette": None,
            "Coverage": None,
            "Fidelity": None,
        }

    # TEST ON REAL (Approximate Prediction)
    real_traj_centroid = (
        original.df_discretized.groupby("tid")[["x_m", "y_m"]].mean().values
    )
    real_labels, membership_strengths = hdbscan.approximate_predict(
        clusterer, real_traj_centroid
    )

    # PREDICTIVE UTILITY METRICS CALCULATION
    # Coverage: Points that HDBSCAN actually assigned to a cluster
    mask = real_labels != -1
    coverage = np.mean(mask)

    real_assigned_coords = real_traj_centroid[mask]
    real_assigned_labels = real_labels[mask]

    # Silhouette Score
    # How well do the SYNTHETIC boundaries fit the REAL data?
    # High score = Synthetic clusters are "True" representations of Real movement.
    # Silhouette Score (On the mapped points)
    if len(np.unique(real_assigned_labels)) < 2:
        silh_score = 0.0
    else:
        # Standard Silhouette (Distance-based)
        silh_score = silhouette_score(real_assigned_coords, real_assigned_labels)

    # 5. MASTER METRIC: Harmonic Spatial Utility (HSU)
    norm_s = max(0, (silh_score + 1) / 2)
    hsu = 2 * (norm_s * coverage) / (norm_s + coverage + 1e-9)

    return {
        "HSU_Master": hsu,  # * fidelity, # Weighted by structural fidelity
        "Silhouette": silh_score,
        "Coverage": coverage,
        "Num_Syn_Clusters": num_syn_clusters,
        "Num_Populated_Clusters": len(np.unique(real_assigned_labels)),
    }
