"""Build data/<ds>/cache.pt (uint8 images NCHW + int64 targets) from torchvision, the
format dp_fcp_gate.load_cifar_cached reads. Needed on a fresh machine (fallback VM)."""
import os
import sys

import numpy as np
import torch
import torchvision

ds = sys.argv[1] if len(sys.argv) > 1 else 'cifar10'
cls = torchvision.datasets.CIFAR10 if ds == 'cifar10' else torchvision.datasets.CIFAR100
root = os.path.join('data', ds)
os.makedirs(root, exist_ok=True)
tr = cls(root, train=True, download=True)
te = cls(root, train=False, download=True)
d = {'train_imgs': torch.from_numpy(np.asarray(tr.data)).permute(0, 3, 1, 2).contiguous(),
     'train_targets': np.asarray(tr.targets, dtype=np.int64),
     'test_imgs': torch.from_numpy(np.asarray(te.data)).permute(0, 3, 1, 2).contiguous(),
     'test_targets': np.asarray(te.targets, dtype=np.int64)}
torch.save(d, os.path.join(root, 'cache.pt'))
print(ds, d['train_imgs'].shape, d['test_imgs'].shape)
