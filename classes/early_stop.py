import math, copy, torch

class EarlyStopper:
    def __init__(self, monitor: str, mode: str = "auto",
                 patience: int = 5, min_delta: float = 0.0,
                 save_path: str | None = None, restore_best: bool = True):
        """
        monitor: key in your epoch row, e.g. 'eval_loss', 'auc_fused', 'acc_fused'
        mode: 'max' (higher is better), 'min', or 'auto' (min for '*loss*', else max)
        patience: epochs without improvement before stopping
        min_delta: required improvement magnitude
        save_path: optional .pth file to save best weights each time it improves
        restore_best: if True, load best weights back at the end
        """
        self.monitor = monitor
        if mode == "auto":
            mode = "min" if "loss" in monitor.lower() else "max"
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.save_path = save_path
        self.restore_best = restore_best

        self.best = -math.inf if mode == "max" else math.inf
        self.bad_epochs = 0
        self.best_state = None
        self.best_epoch = -1
        self.last_improved = False

    def _is_better(self, val):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return False
        if self.mode == "max":
            return val > (self.best + self.min_delta)
        else:
            return val < (self.best - self.min_delta)

    def step(self, metrics: dict, trainer, epoch: int) -> bool:
        val = metrics.get(self.monitor, None)
        improved = self._is_better(val)
        self.last_improved = improved

        if improved:
            self.best = val
            self.best_epoch = epoch
            self.bad_epochs = 0
            # snapshot + optional save
            state = {
                "img_tower": trainer.img_tower.state_dict(),
                "md_tower":  trainer.md_tower.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
            }
            if hasattr(trainer, "bridge"):   state["bridge"]   = trainer.bridge.state_dict()
            if hasattr(trainer, "head_img"): state["head_img"] = trainer.head_img.state_dict()
            if hasattr(trainer, "head_md"):  state["head_md"]  = trainer.head_md.state_dict()
            # keep an in-memory copy for restore(); file save is optional
            self.best_state = copy.deepcopy(state)
            if self.save_path: torch.save(state, self.save_path)
            print(f"[early] ↑ new best {self.monitor}={val:.5f} at epoch {epoch+1}")
        else:
            self.bad_epochs += 1

        stop = self.bad_epochs >= self.patience
        if stop:
            print(f"[early] stopping: no improvement in {self.patience} epochs "
                  f"(best {self.monitor}={self.best:.5f} @ epoch {self.best_epoch+1})")
        return stop

    def restore(self, trainer):
        if not self.restore_best:
            return
        # Prefer in-memory best state; otherwise try loading from save_path
        st = self.best_state
        if st is None and self.save_path:
            try:
                st = torch.load(self.save_path, map_location="cpu")
            except Exception:
                st = None
        if st is None:
            return
        trainer.img_tower.load_state_dict(st["img_tower"])
        trainer.md_tower.load_state_dict(st["md_tower"])
        if "bridge" in st and hasattr(trainer, "bridge"):
            trainer.bridge.load_state_dict(st["bridge"])
        if "head_img" in st and hasattr(trainer, "head_img"):
            trainer.head_img.load_state_dict(st["head_img"])
        if "head_md" in st and hasattr(trainer, "head_md"):
            trainer.head_md.load_state_dict(st["head_md"])
        trainer.optimizer.load_state_dict(st["optimizer"])
