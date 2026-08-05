"""Inference on test set → Kaggle submission.csv

Pick a run, load its best_model.pth, predict on all test images,
and write ``submission.csv`` into the run folder.

Usage::

    python infer.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import PetalsDataset
from models import GoogLeNet


def _list_runs(output_root):
    """Return sorted names of run folders that have a best_model.pth."""
    if not output_root.exists():
        return []
    return sorted(
        d.name for d in output_root.iterdir()
        if d.is_dir() and (d / 'best_model.pth').exists()
    )


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = str(PROJECT_ROOT / 'data')
    OUTPUT_ROOT = PROJECT_ROOT / 'output'

    # -- Pick a run ------------------------------------------------
    runs = _list_runs(OUTPUT_ROOT)
    if not runs:
        print('No runs with best_model.pth found. Train a model first.')
        return

    print('Available runs:')
    for i, name in enumerate(runs):
        ckpt = torch.load(OUTPUT_ROOT / name / 'checkpoint.pth',
                          map_location='cpu', weights_only=False)
        print(f'  [{i}] {name}  (best val acc={ckpt.get("best_val_acc", 0):.4f})')

    ans = input('Choose a run: ').strip()
    try:
        run = runs[int(ans)]
    except (ValueError, IndexError):
        print('Invalid choice.')
        return

    run_dir = OUTPUT_ROOT / run
    weights_path = run_dir / 'best_model.pth'

    SUBMISSION_DIR = PROJECT_ROOT / 'submissions'
    SUBMISSION_DIR.mkdir(exist_ok=True)
    out_path = SUBMISSION_DIR / f'{run}.csv'

    # -- Model ----------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\nLoading {weights_path} on {device} ...')
    model = GoogLeNet(num_classes=104)
    model.load_state_dict(torch.load(weights_path, map_location=device,
                                     weights_only=True))
    model.to(device)
    model.eval()

    # -- Test data ------------------------------------------------
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_ds = PetalsDataset(DATA_DIR, image_size=224, split='test',
                            transform=test_transform, predecode=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                             num_workers=0, pin_memory=True)
    print(f'Test images: {len(test_ds)}')

    # -- Predict --------------------------------------------------
    ids_all, preds_all = [], []
    spinner = '|/-\\'
    with torch.no_grad():
        for i, (imgs, img_ids) in enumerate(test_loader):
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu()
            ids_all.extend(img_ids)
            preds_all.extend(preds.tolist())
            if i % 10 == 0:
                print(f'\033[2K\r  Predicting... {spinner[(i // 10) % 4]}  '
                      f'batch {i}/{len(test_loader)}', end='', flush=True)

    print(f'\033[2K\r  Done. {len(ids_all)} predictions.')

    # -- Write submission -----------------------------------------
    with open(out_path, 'w') as f:
        f.write('id,label\n')
        for img_id, pred in zip(ids_all, preds_all):
            f.write(f'{img_id},{pred}\n')

    print(f'Submission saved to {out_path}')


if __name__ == '__main__':
    main()
