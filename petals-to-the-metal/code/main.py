"""Entry point for Petals to the Metal — multi-model training.

Run::

    python main.py

Prompts for model, run name, hyperparams, resume, and description.
Each training run is stored in ``output/<run_name>/``.
"""

import shutil
import sys
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import PetalsDataset
from models import AVAILABLE_MODELS, build_model
from engine import train


def _list_runs(output_root):
    """Return sorted names of run folders that have a checkpoint."""
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
        print('No existing runs found.  Starting fresh.')
        return None

    print('\nExisting runs:')
    for i, name in enumerate(runs):
        ckpt = torch.load(output_root / name / 'checkpoint.pth',
                          map_location='cpu', weights_only=False)
        model = ckpt.get('model_name', '?')
        print(f'  [{i}] {name}  ({model}, epoch {ckpt["epoch"]}, '
              f'val acc={ckpt.get("best_val_acc", 0):.4f})')

    print(f'  [N] New run')
    ans = input('Choose a run to resume, or N for new: ').strip()
    if ans.lower() == 'n':
        return None
    try:
        return runs[int(ans)]
    except (ValueError, IndexError):
        return None


def _build_pretrained_model(model_name, checkpoint_path, device='cpu'):
    """Build a model and load pretrained weights for inference only."""
    model = build_model(model_name, num_classes=104)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    return model


def _layer_test(model, model_name='unknown'):
    """Layer-by-layer shape trace (GoogLeNet) or input→output (others)."""
    x = torch.randn(1, 3, 224, 224)
    print(f'  {"input":>16s}  ->  {tuple(x.shape)}')
    with torch.no_grad():
        if model_name == 'googlenet':
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
        else:
            y = model(x)
            print(f'  {"forward":>16s}  ->  {tuple(y.shape)}')
    print(f'  {"params":>16s}  =  '
          f'{sum(p.numel() for p in model.parameters()):,}')


def main():
    # -- Paths ----------------------------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = str(PROJECT_ROOT / 'data')
    OUTPUT_ROOT = PROJECT_ROOT / 'output'
    OUTPUT_ROOT.mkdir(exist_ok=True)

    # -- New run or resume? ---------------------------------------
    resume_run = _pick_run(OUTPUT_ROOT)

    if resume_run:
        src_dir = OUTPUT_ROOT / resume_run
        ckpt = torch.load(src_dir / 'checkpoint.pth',
                          map_location='cpu', weights_only=False)
        model_name = ckpt.get('model_name', 'unknown')
        print(f'Source: {resume_run}  ({model_name}, epoch {ckpt["epoch"]}, '
              f'val acc={ckpt.get("best_val_acc", 0):.4f})')

        name = input('Run name [default: gn_v1_cont]: ').strip()
        if not name:
            name = resume_run + '_cont'
        output_dir = OUTPUT_ROOT / name
        output_dir.mkdir(exist_ok=True)
        shutil.copy(src_dir / 'checkpoint.pth', output_dir / 'checkpoint.pth')
        resume = True
    else:
        # -- Pick model -------------------------------------------
        print('\nAvailable models:')
        models_list = list(AVAILABLE_MODELS.items())
        for i, (key, desc) in enumerate(models_list):
            print(f'  [{i}] {desc}')
        ans = input('Choose a model [default 0]: ').strip()
        try:
            model_name = models_list[int(ans) if ans else 0][0]
        except (ValueError, IndexError):
            model_name = models_list[0][0]
        print(f'  Selected: {model_name}')

        name = input('Run name: ').strip()
        if not name:
            name = 'run01'
        output_dir = OUTPUT_ROOT / name
        output_dir.mkdir(exist_ok=True)
        resume = False

    # -- Hyperparameters ------------------------------------------
    if resume:
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

    print(f'\nRun: {output_dir.name}  |  model={model_name}  |  '
          f'lr={lr}  |  epochs={num_epochs}  |  resume={resume}')

    # -- Description & notes --------------------------------------
    desc = input('Description (optional): ').strip()
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    header = (f'### {output_dir.name}  |  {now}  |  '
              f'model={model_name}  lr={lr}  epochs={num_epochs}')
    lines = [header]
    if resume:
        src_notes = src_dir / 'notes.txt'
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

    # -- Build model & layer test ---------------------------------
    print('\n=== Layer shape test ===')
    model = build_model(model_name, num_classes=104, dropout=0.5)
    _layer_test(model, model_name)

    # -- Transforms -----------------------------------------------
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.2,
                               contrast=0.2,
                               saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # -- Data -----------------------------------------------------
    # Auto-pick batch size: ResNet50 needs less VRAM than GoogLeNet
    _default_bs = {'googlenet': 128, 'resnet50': 32}
    default_bs = _default_bs.get(model_name, 64)
    bs_str = input(f'Batch size [default {default_bs}]: ').strip()
    batch_size = int(bs_str) if bs_str else default_bs

    print(f'\n=== Loading data (batch_size={batch_size}) ===')
    train_ds = PetalsDataset(DATA_DIR, image_size=224, split='train',
                             transform=train_transform, predecode=False)
    val_ds   = PetalsDataset(DATA_DIR, image_size=224, split='val',
                             transform=val_transform, predecode=False)
    print(f'  train: {len(train_ds)} samples')
    print(f'  val:   {len(val_ds)} samples')

    train_iter = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)
    val_iter   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # -- Train ----------------------------------------------------
    print('\n=== Training ===')
    train(model, train_iter, val_iter,
          num_epochs=num_epochs, lr=lr,
          device=None, resume=resume,
          output_dir=str(output_dir),
          model_name=model_name)

    print(f'\nRun saved to {output_dir}/')
    print('Training mission completed!')


if __name__ == '__main__':
    main()
