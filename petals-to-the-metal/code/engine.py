"""Training engine: Timer, Accumulator, train/eval loops, curve saving."""

import time
from pathlib import Path

import torch
from torch import nn


# -- Basic utilities -----------------------------------------------

class Timer:
    """Record and aggregate elapsed time across multiple intervals.

    Usage::

        timer = Timer()
        timer.start(); ... do work ...; timer.stop()
        print(f'{timer.sum():.1f} sec  |  avg {timer.avg():.3f}s')
    """

    def __init__(self):
        self.times = []
        self._start = None

    def start(self):
        self._start = time.time()

    def stop(self):
        if self._start is not None:
            self.times.append(time.time() - self._start)
            self._start = None

    def sum(self):
        return sum(self.times)

    def avg(self):
        return sum(self.times) / len(self.times) if self.times else 0.0

    def reset(self):
        self.times.clear()
        self._start = None

    def __str__(self):
        return _fmt_time(self.sum())


def _fmt_time(seconds):
    if seconds < 60:
        return f'{seconds:.1f}s'
    elif seconds < 3600:
        return f'{seconds / 60:.1f}min'
    else:
        return f'{seconds / 3600:.2f}h'


class Accumulator:
    """Accumulate values across *n* variables (e.g. loss, correct, total)."""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    """Count correct predictions.

    Handles GoogLeNetOutputs (torchvision pretrained) transparently.
    """
    if hasattr(y_hat, 'logits'):
        y_hat = y_hat.logits
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


# -- Evaluation & training loops -----------------------------------

def evaluate_accuracy(net, data_iter, device=None):
    """Compute accuracy on a dataset, showing a spinner in the terminal."""
    if isinstance(net, nn.Module):
        net.eval()
    metric = Accumulator(2)
    spinner = '|/-\\'
    with torch.no_grad():
        for i, (X, y) in enumerate(data_iter):
            if device is not None:
                X, y = X.to(device), y.to(device)
            y_hat = net(X)
            metric.add(accuracy(y_hat, y), y.numel())
            if i % 15 == 0:
                print(f'\033[2K\r  Validating... {spinner[(i // 15) % 4]}',
                      end='', flush=True)
    print('\033[2K\r', end='', flush=True)
    return metric[0] / metric[1]


def train_epoch(net, train_iter, loss_fn, optimizer, device=None):
    """Train for one epoch, showing a spinner in the terminal."""
    if isinstance(net, nn.Module):
        net.train()
    metric = Accumulator(3)
    spinner = '|/-\\'
    for i, (X, y) in enumerate(train_iter):
        if i % 30 == 0:
            print(f'\033[2K\r  Training... {spinner[(i // 30) % 4]}',
                  end='', flush=True)
        if device is not None:
            X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = net(X)
        logits = y_hat.logits if hasattr(y_hat, 'logits') else y_hat
        l = loss_fn(logits, y)
        l.mean().backward()
        optimizer.step()
        metric.add(l.sum().item() * y.numel(), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


# -- Curve plotting ------------------------------------------------

def save_curves(train_losses, train_accs, val_accs, filename='training_curves.png'):
    """Save a two-panel loss / accuracy plot after training finishes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, train_losses, '-', label='train loss')
    ax1.set_xlabel('epoch'); ax1.set_ylabel('loss')
    ax1.set_title('Training Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, '-', label='train acc')
    ax2.plot(epochs, val_accs, 'm--', label='val acc')
    ax2.set_xlabel('epoch'); ax2.set_ylabel('accuracy')
    ax2.set_title('Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)

    best_idx = val_accs.index(max(val_accs))
    ax2.axvline(x=best_idx + 1, color='r', linestyle=':', alpha=0.5)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f'Curves saved to {filename}')


# -- Main training function ----------------------------------------

def train(net, train_iter, val_iter, num_epochs, lr, device=None, resume=False,
          output_dir='.', model_name='unknown'):
    """Full training pipeline with terminal progress, checkpointing, and plots.

    Args:
        net: model (nn.Module)
        train_iter: training DataLoader
        val_iter: validation DataLoader
        num_epochs: total number of epochs
        lr: learning rate
        device: 'cuda' / 'cpu' (auto-detected if None)
        resume: whether to load checkpoint.pth and continue
        output_dir: where to save models and curves
        model_name: stored in checkpoint; validated on resume

    Returns:
        (train_losses, train_accs, val_accs)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out = Path(output_dir)

    header = f'Training on {device} | epochs={num_epochs} | lr={lr}'
    print(header)
    if device == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    net.to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, train_accs, val_accs = [], [], []
    start_epoch = 1
    best_val_acc = 0.0

    # -- Resume from checkpoint -----------------------------------
    ckpt_path = out / 'checkpoint.pth'
    if resume and ckpt_path.exists():
        print(f'  Resuming from {ckpt_path} ...')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_model = ckpt.get('model_name', 'unknown')
        if ckpt_model != model_name:
            raise RuntimeError(
                f"Model mismatch: checkpoint uses '{ckpt_model}', "
                f"but current model is '{model_name}'. "
                f"Resume only works with the same model type."
            )
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        train_losses = ckpt.get('train_losses', [])
        train_accs = ckpt.get('train_accs', [])
        val_accs = ckpt.get('val_accs', [])
        print(f'  Model: {ckpt_model} | Restored epoch {ckpt["epoch"]}, '
              f'best val acc: {best_val_acc:.4f}')

    timer_epoch = Timer()
    timer_total = Timer()
    timer_total.start()
    has_temp = False  # track whether a temporary status line sits above spinner

    for epoch in range(start_epoch, num_epochs + 1):
        timer_epoch.start()
        train_l, train_acc = train_epoch(net, train_iter, loss_fn, optimizer, device)
        val_acc = evaluate_accuracy(net, val_iter, device)
        timer_epoch.stop()
        scheduler.step()

        train_losses.append(train_l)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # track best & save full checkpoint every epoch
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        torch.save({
            'model_name': model_name,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_acc': best_val_acc,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
        }, out / 'checkpoint.pth')

        # -- Terminal display -----------------------------------
        if epoch > start_epoch:
            done = epoch - start_epoch + 1
            eta = timer_epoch.sum() / done * (num_epochs - epoch)
            eta_str = f' | ETA: {_fmt_time(eta)}'
        else:
            eta_str = ''

        status = (f'epoch {epoch:3d}/{num_epochs} | '
                  f'train loss {train_l:.4f}, train acc {train_acc:.4f}, '
                  f'val acc {val_acc:.4f} | '
                  f'{_fmt_time(timer_epoch.times[-1])}/epoch{eta_str}')

        is_milestone = (epoch % 5 == 0 or epoch == start_epoch)
        is_last = (epoch == num_epochs)
        if is_milestone or is_last:
            with open(out / 'notes.txt', 'a', encoding='utf-8') as f:
                f.write(status + '\n')

        if is_milestone:
            # Milestone: saved permanently.  Discard any temp line above.
            if has_temp:
                print(f'\033[A\033[2K', end='', flush=True)
                has_temp = False
            print(f'\033[2K\r{status}')
        else:
            # Non-milestone: keep as a temp line above the spinner area.
            if has_temp:
                # Overwrite the previous temp line (one line up).
                print(f'\033[A\033[2K\r{status}\033[B', end='', flush=True)
            else:
                # First temp after a milestone — carve out a line.
                print()                                # blank line → spinner area
                print(f'\033[A\033[2K\r{status}\033[B', end='', flush=True)
                has_temp = True

    print()
    timer_total.stop()

    # -- Final save: export best weights for inference -------------
    best_ckpt = torch.load(out / 'checkpoint.pth', map_location='cpu',
                           weights_only=False)
    torch.save(best_ckpt['model'], out / 'best_model.pth')

    summary = (f'Training complete.  Total: {timer_total}  |  '
               f'Best val acc: {best_val_acc:.4f}')
    print(f'\n{summary}')
    print(f'Saved to {out}/')
    print(f'  checkpoint.pth  (for resuming)')
    print(f'  best_model.pth  (val acc={best_val_acc:.4f}, for inference)')

    save_curves(train_losses, train_accs, val_accs,
                filename=str(out / 'training_curves.png'))
    return train_losses, train_accs, val_accs
