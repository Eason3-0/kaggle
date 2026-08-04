"""Entry point for Petals to the Metal training.

Run::

    python main.py

Prompts for run name, learning rate, epochs, and whether to resume.
Each training run is stored in ``output/<run_name>/``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import PetalsDataset
from googlenet import GoogLeNet
from engine import train


def _list_runs(output_root):
    """Return sorted list of existing run folder names under *output_root*."""
    if not output_root.exists():
        return []
    return sorted(
        d.name for d in output_root.iterdir()
        if d.is_dir() and (d / 'checkpoint.pth').exists()
    )


def _pick_run(output_root):
    """Let the user pick an existing run to resume, or start fresh."""
    runs = _list_runs(output_root)
    if not runs:
        print('No existing runs found. Starting fresh.')
        return None

    print('\nExisting runs:')
    for i, name in enumerate(runs):
        ckpt = torch.load(output_root / name / 'checkpoint.pth',
                          map_location='cpu', weights_only=False)
        print(f'  [{i}] {name}  (epoch {ckpt["epoch"]}, '
              f'val acc={ckpt.get("best_val_acc", 0):.4f})')

    print(f'  [N] New run')
    ans = input('Choose a run to resume, or N for new: ').strip()

    if ans.lower() == 'n':
        return None
    try:
        return runs[int(ans)]
    except (ValueError, IndexError):
        return None


def main():
    # -- Paths ----------------------------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = str(PROJECT_ROOT / 'data')
    OUTPUT_ROOT = PROJECT_ROOT / 'output'
    OUTPUT_ROOT.mkdir(exist_ok=True)

    # -- New run or resume? ---------------------------------------
    resume_run = _pick_run(OUTPUT_ROOT)

    if resume_run:
        # Resume → new folder, copy checkpoint from source
        src_dir = OUTPUT_ROOT / resume_run
        src_ckpt = torch.load(src_dir / 'checkpoint.pth',
                              map_location='cpu', weights_only=False)
        print(f'Source: {resume_run} (epoch {src_ckpt["epoch"]}, '
              f'val acc={src_ckpt.get("best_val_acc", 0):.4f})')

        name = input('Run name [default: gn_v1_cont]: ').strip()
        if not name:
            name = resume_run + '_cont'
        output_dir = OUTPUT_ROOT / name
        output_dir.mkdir(exist_ok=True)

        # copy checkpoint to new folder
        import shutil
        shutil.copy(src_dir / 'checkpoint.pth', output_dir / 'checkpoint.pth')
        resume = True
    else:
        name = input('Run name: ').strip()
        if not name:
            name = 'run01'
        output_dir = OUTPUT_ROOT / name
        output_dir.mkdir(exist_ok=True)
        resume = False

    # -- Hyperparameters ------------------------------------------
    if resume:
        ckpt = torch.load(output_dir / 'checkpoint.pth',
                          map_location='cpu', weights_only=False)
        extra_str = input(f'Additional epochs [default 20]: ').strip()
        extra = int(extra_str) if extra_str else 20
        num_epochs = ckpt['epoch'] + extra
        print(f'  Will run epochs {ckpt["epoch"] + 1}–{num_epochs} '
              f'({extra} additional)')
    else:
        ep_str = input(f'Total epochs [default 60]: ').strip()
        num_epochs = int(ep_str) if ep_str else 60

    lr_str = input(f'Learning rate [default 1e-3]: ').strip()
    lr = float(lr_str) if lr_str else 1e-3

    print(f'\nRun: {output_dir.name}  |  lr={lr}  |  epochs={num_epochs}  |  '
          f'resume={resume}')

    # -- Description -----------------------------------------------
    desc = input('Description (optional): ').strip()
    import datetime
    header = (f'### {output_dir.name}  |  '
              f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}  |  '
              f'lr={lr}  epochs={num_epochs}')
    lines = [header]
    if resume:
        src_notes = output_dir.parent / resume_run / 'notes.txt'
        if src_notes.exists():
            last = src_notes.read_text(encoding='utf-8').strip().split('\n')[-1]
            lines.append(f'source: {resume_run} — {last}')
        else:
            lines.append(f'source: {resume_run} (notes unavailable)')
    if desc:
        lines.append(desc)
    lines.append('')
    with open(output_dir / 'notes.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    # -- Layer-by-layer shape trace -------------------------------
    print('\n=== Layer shape test ===')
    model = GoogLeNet(num_classes=104, dropout=0.5)
    x = torch.randn(1, 3, 224, 224)
    print(f'  {"input":>16s}  ->  {tuple(x.shape)}')
    with torch.no_grad():
        for name, layer in [
            ('stem',           model.stem),
            ('inception3a',    model.inception3a),
            ('inception3b',    model.inception3b),
            ('pool3',          model.pool3),
            ('inception4a',    model.inception4a),
            ('inception4b',    model.inception4b),
            ('inception4c',    model.inception4c),
            ('inception4d',    model.inception4d),
            ('inception4e',    model.inception4e),
            ('pool4',          model.pool4),
            ('inception5a',    model.inception5a),
            ('inception5b',    model.inception5b),
            ('avgpool',        model.avgpool),
            ('classifier',     model.classifier),
        ]:
            x = layer(x)
            print(f'  {name:>16s}  ->  {tuple(x.shape)}')
    print(f'  {"params":>16s}  =  {sum(p.numel() for p in model.parameters()):,}')

    # -- Transforms -----------------------------------------------
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=25),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # -- Data -----------------------------------------------------
    print('\n=== Loading data ===')
    train_ds = PetalsDataset(DATA_DIR, image_size=224, split='train',
                             transform=train_transform, predecode=False)
    val_ds = PetalsDataset(DATA_DIR, image_size=224, split='val',
                           transform=val_transform, predecode=False)
    print(f'  train: {len(train_ds)} samples')
    print(f'  val:   {len(val_ds)} samples')

    train_iter = DataLoader(train_ds, batch_size=128, shuffle=True,
                            num_workers=0, pin_memory=True)
    val_iter = DataLoader(val_ds, batch_size=128, shuffle=False,
                          num_workers=0, pin_memory=True)

    # -- Train ----------------------------------------------------
    print('\n=== Training ===')
    train(model, train_iter, val_iter,
          num_epochs=num_epochs, lr=lr,
          device=None, resume=resume,
          output_dir=str(output_dir))

    print(f'\nRun saved to {output_dir}/')
    print('Training mission completed!')


if __name__ == '__main__':
    main()
