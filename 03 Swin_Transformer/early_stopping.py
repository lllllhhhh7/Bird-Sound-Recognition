import torch

class EarlyStopping:
    def __init__(self, patience: int = 15, mode: str = 'min', min_delta: float = 0,
                 metric_name: str = 'val_loss', checkpoint_path: str = 'best_model.pth'):
        self.patience = patience
        assert mode in ['min', 'max']
        self.mode = mode
        self.min_delta = min_delta
        self.wait = 0
        self.best_metric = None
        self.early_stop = False
        self.metric_name = metric_name
        self.checkpoint_path = checkpoint_path

        if self.mode == 'max':
            self.metric_name = 'val_acc'
        elif self.mode == 'min':
            self.metric_name = 'val_loss'

    def __call__(self, epoch: int, model: torch.nn.Module, val_metric: float):
        if self.best_metric is None:
            self.best_metric = val_metric
            self.save_checkpoint(model)
        elif self.mode == 'min' and val_metric < self.best_metric - self.min_delta:
            self.best_metric = val_metric
            self.wait = 0
            self.save_checkpoint(model)
            print(f'第 {epoch} 轮时，验证集 {self.metric_name} 降低，为 {self.best_metric}')
        elif self.mode == 'max' and val_metric > self.best_metric + self.min_delta:
            self.best_metric = val_metric
            self.wait = 0
            self.save_checkpoint(model)
            print(f'第 {epoch} 轮时，验证集 {self.metric_name} 提升，为 {self.best_metric}')
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.early_stop = True
                print(f'在连续 {self.patience} 个周期内，第 {epoch} 轮时 {self.metric_name} 未能进步，触发提前停止。')

    def save_checkpoint(self, model: torch.nn.Module):
        torch.save(model.state_dict(), self.checkpoint_path)

    def load_best_model(self, model: torch.nn.Module):
        model.load_state_dict(torch.load(self.checkpoint_path))
