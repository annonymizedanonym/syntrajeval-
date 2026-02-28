class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0, verbose=False):
        """
        patience: number of epochs with no improvement after which to stop
        min_delta: minimum change in the monitored metric to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best = None
        self.counter = 0
        self.early_stop = False

    def step(self, metric, epoch=None):
        if self.best is None:
            self.best = metric
        elif metric > self.best - self.min_delta:  # no improvement (since lower = better)
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement at epoch {epoch}, count = {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f"[EarlyStopping] Improvement at epoch {epoch}: {self.best:.6f} → {metric:.6f}")
            self.best = metric
            self.counter = 0
