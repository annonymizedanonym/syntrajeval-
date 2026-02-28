# UtilEval_Traj

The objective of this project is to:

- Evaluate a set of synthetic dataset versions (variants) of an original, true one
- compare them, on a chosen metrics (a chosen evaluation aspect), according to their similarity (fidelity)with respect to the original dataset.
- Keep it simple: While aiming for one single score per metric is not realistic, as each error/distance is by nature imperfect to be conculsive.

Some requirements arise to make evaluation both reproducible and comparable:

- the evaluation results need not change as the set of variants changes, we must ensure that the addition of a variant does not influence the evaluation results of other existing/future variants and for that, it is crucial to ensure:

  - Consistant projection to a meter CRS: projection must be unchanged across current and future variants.
  - Anticipatory measures must be taken for when adding a variant changes the crs_meter, therefore the projection.
  - Fixed absolute grid for descritization (ensure same cell_id for the same continuous coodinates (x,y) across all synthetic datasets)

- with the reproductibility constaints in mind, comparable evaluation is achieved through ensureing that the metric evaluates a certain synethic dataset with respect to the original one by:

- - Treating synthetic mass outside the original as “penalty mass” that increases error. So a metric needs to be conceived using fixed parameters, calculated based on the original dataset only and reused for all variants. This avoids adaptive behavior that would invalidate cross-variant comparisons.
