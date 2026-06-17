import json

class TrainingLogger:

    def __init__(self):
        self.logs = []

    def log(self, epoch, K, beta, loss, sparsity):

        self.logs.append({
            "epoch": epoch,
            "K": K,
            "beta": beta,
            "loss": loss,
            "sparsity": sparsity
        })

    def save(self, path="adaptive_log.json"):
        with open(path, "w") as f:
            json.dump(self.logs, f, indent=4)