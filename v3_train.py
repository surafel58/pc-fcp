"""v3 full rerun, part 1 (2026-10-14): train the federated classifiers and cache their
outputs, so every calibration method can be evaluated (and re-evaluated) from the cache.

Models (decided after the mock defense, see results/t1_pilot and results/step3_text):
  cifar10 / cifar100 : ImageNet-pretrained ResNet-18 (torchvision), inputs resized to 224,
                       FedAvg fine-tuning, 30 rounds, lr 0.01, 2 local epochs, bf16 autocast
  text (20 Newsgroups): frozen all-mpnet-base-v2 sentence embeddings + the thesis MLP,
                       logit-adjusted loss with each client's label prior, lr 1.0, 60 rounds
Splits, long-tail construction, client partitions and the calibration partition are
exactly those of dp_fcp_gate.run_seed / dp_fcp_text.run_seed (same seeds, same calls).

Each run writes results/v3/cache/<dataset>_K<K>_dir<dir>_seed<seed>.npz with
cal_probs, cal_labels, cal_client (client index of each calibration row), test_probs,
test_labels, and one row of top-1 accuracy metrics to results/v3/accuracy.csv.
"""
import argparse
import copy
import csv
import os
import time

import numpy as np
import torch
from torch.utils.data import Subset

import dp_fcp_gate as g

OUT = 'results/v3'
IMG = {'cifar10': dict(n_classes=10, n_max=5000, if_factor=100, cache='data/cifar10/cache.pt'),
       'cifar100': dict(n_classes=100, n_max=500, if_factor=10, cache='data/cifar100/cache.pt')}


# ---------------------------------------------------------------- images
def train_images(ds_name, seed, n_clients, dirichlet, device, rounds, lr, loss='la', if_factor=None):
    from t1_model_pilot import PretrainedResNet18, local_train
    cfg = IMG[ds_name]
    cifar_train, cifar_test = g.load_cifar_cached(cfg['cache'])
    tr_targets = cifar_train.targets
    lt_idx, _ = g.lt_indices(tr_targets, if_factor or cfg['if_factor'], n_max=cfg['n_max'], seed=seed)
    rng = np.random.RandomState(seed + 1)
    lt_arr = np.asarray(lt_idx); lt_y = tr_targets[lt_arr]
    train_idx, cal_idx = [], []
    for c in range(g.N_CLASSES):
        ci = lt_arr[lt_y == c].tolist(); rng.shuffle(ci)
        n_tr = max(1, int(0.5 * len(ci)))
        train_idx += ci[:n_tr]; cal_idx += ci[n_tr:]
    lists = g.dirichlet_partition(train_idx, tr_targets, n_clients, dirichlet, seed)
    client_ds = [Subset(cifar_train, l) for l in lists if len(l) >= 2]
    torch.manual_seed(seed); np.random.seed(seed)
    model = PretrainedResNet18(g.N_CLASSES).to(device)
    t0 = time.time()
    total = float(sum(len(ds) for ds in client_ds))
    for rnd in range(1, rounds + 1):
        cur_lr = lr * (0.99 ** (rnd - 1))
        # running FedAvg: the same weighted sum, in the same order, as g.aggregate, without
        # holding every client's state (100 ResNet-18 copies exhausted the VM's 16 GB)
        avg = None
        for ds in client_ds:
            m = copy.deepcopy(model)
            opt = torch.optim.SGD(m.parameters(), lr=cur_lr, momentum=0.9, weight_decay=5e-4)
            # 2026-10-19: logit adjustment (Menon et al. 2021) as the text model already
            # uses: the client's own add-one smoothed label prior is added to the logits
            # during training only. loss='ce' is the previous plain cross-entropy.
            lp = None
            if loss == 'la':
                cnt = torch.bincount(torch.as_tensor(tr_targets[np.asarray(ds.indices)]),
                                     minlength=g.N_CLASSES).float() + 1.0
                lp = torch.log(cnt / cnt.sum())
            local_train(m, ds, opt, device, 2, 64, lp, amp=(device == 'cuda'))
            with torch.no_grad():
                sd = m.state_dict()
                if avg is None:
                    avg = {k: torch.zeros_like(v, dtype=torch.float32, device='cpu') for k, v in sd.items()}
                for k, v in sd.items():
                    avg[k] += (len(ds) / total) * v.detach().cpu().to(torch.float32)
            del m, opt
        gs = model.state_dict()
        model.load_state_dict({k: avg[k].to(gs[k].dtype) for k in avg})
        if rnd % 10 == 0:
            print(f'  [{ds_name} K={n_clients} seed={seed}] round {rnd}/{rounds} ({time.time()-t0:.0f}s)', flush=True)
    cal_src = g._CachedCIFAR(cifar_train.images, cifar_train.targets, train=False)
    cal_probs, cal_labels = g.softmax_on(model, Subset(cal_src, cal_idx), device)
    test_probs, test_labels = g.softmax_on(model, cifar_test, device)
    cal_index_of = {gi: r for r, gi in enumerate(cal_idx)}
    cal_lists = g.dirichlet_partition(cal_idx, tr_targets, n_clients, dirichlet, seed + 7)
    cal_client = np.full(len(cal_idx), -1, dtype=np.int64)
    for k, lst in enumerate(cal_lists):
        for gi in lst:
            cal_client[cal_index_of[gi]] = k
    return cal_probs, cal_labels, cal_client, test_probs, test_labels, len(train_idx), time.time() - t0


# ---------------------------------------------------------------- text
def train_text(seed, n_clients, dirichlet, device, rounds, lr):
    import dp_fcp_text as T
    from step3b_text_select import la_local_train, load
    Xtr, ytr, Xte, yte, n_feat = load('mpnet')
    C = g.N_CLASSES
    lt_idx, _ = T.lt_subsample(ytr, 450, 10, C, seed)
    lt_idx = np.asarray(lt_idx); lt_y = ytr[lt_idx]
    rng = np.random.RandomState(seed + 1)
    train_idx, cal_idx = [], []
    for c in range(C):
        ci = lt_idx[lt_y == c].tolist(); rng.shuffle(ci)
        n_tr = max(1, int(0.5 * len(ci)))
        train_idx += ci[:n_tr]; cal_idx += ci[n_tr:]
    lists = g.dirichlet_partition(train_idx, ytr, n_clients, dirichlet, seed)
    Xt = torch.from_numpy(Xtr).to(device); yt = torch.from_numpy(ytr).long().to(device)
    torch.manual_seed(seed); np.random.seed(seed)
    model = T.MLP(n_feat, C).to(device)
    t0 = time.time()
    for rnd in range(1, rounds + 1):
        cur_lr = lr * (0.99 ** (rnd - 1))
        states, weights = [], []
        for lst in lists:
            if len(lst) < 2:
                continue
            m = copy.deepcopy(model)
            opt = torch.optim.SGD(m.parameters(), lr=cur_lr, momentum=0.9, weight_decay=5e-4)
            idx = torch.tensor(lst, device=device)
            la_local_train(m, Xt[idx], yt[idx], opt, device, 2, 64)
            states.append({k: v.detach().cpu() for k, v in m.state_dict().items()})
            weights.append(len(lst))
            del m, opt
        T.aggregate(model, states, weights)
    cal_idx = np.asarray(cal_idx)
    cal_probs = T.softmax_on(model, Xt[torch.tensor(cal_idx, device=device)], device)
    test_probs = T.softmax_on(model, torch.from_numpy(Xte).to(device), device)
    cal_index_of = {int(gi): r for r, gi in enumerate(cal_idx)}
    cal_lists = g.dirichlet_partition(cal_idx.tolist(), ytr, n_clients, dirichlet, seed + 7)
    cal_client = np.full(len(cal_idx), -1, dtype=np.int64)
    for k, lst in enumerate(cal_lists):
        for gi in lst:
            cal_client[cal_index_of[int(gi)]] = k
    return cal_probs, ytr[cal_idx], cal_client, test_probs, yte, len(train_idx), time.time() - t0


def accuracy_row(tag, ds_name, seed, K, dirichlet, test_probs, test_labels, n_train, secs):
    C = g.N_CLASSES
    pred = test_probs.argmax(1)
    acc_c = np.array([(pred[test_labels == c] == c).mean() for c in range(C)])
    tail = list(range(C - C // 3, C)); head = [c for c in range(C) if c not in tail]
    top5 = (np.argsort(-test_probs, 1)[:, :5] == test_labels[:, None]).any(1).mean()
    return {'tag': tag, 'dataset': ds_name, 'seed': seed, 'n_clients': K, 'dirichlet': dirichlet,
            'n_train': n_train, 'top1': round(float((pred == test_labels).mean()), 4),
            'top1_head': round(float(acc_c[head].mean()), 4), 'top1_tail': round(float(acc_c[tail].mean()), 4),
            'top1_worst': round(float(acc_c.min()), 4), 'top5': round(float(top5), 4),
            'per_class_top1': ' '.join(f'{a:.3f}' for a in acc_c), 'train_seconds': round(secs, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['cifar10', 'cifar100', 'text'], required=True)
    p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--n-clients', type=int, default=10)
    p.add_argument('--dirichlet', type=float, default=0.5)
    p.add_argument('--rounds', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--loss', choices=['la', 'ce'], default='la',
                   help='image loss: la = logit-adjusted cross-entropy (default), ce = plain')
    p.add_argument('--if-factor', type=int, default=None, help='override the imbalance factor (images)')
    p.add_argument('--tag-suffix', default=None,
                   help='cache tag suffix; default _la for loss=la, _if<IF> appended when --if-factor is given')
    a = p.parse_args()
    os.makedirs(os.path.join(OUT, 'cache'), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    g.N_CLASSES = 20 if a.dataset == 'text' else IMG[a.dataset]['n_classes']
    g.TAIL_CLASSES = list(range(g.N_CLASSES - g.N_CLASSES // 3, g.N_CLASSES))
    rounds = a.rounds or (60 if a.dataset == 'text' else 30)
    lr = a.lr or (1.0 if a.dataset == 'text' else 0.01)
    for seed in [int(s) for s in a.seeds.split(',')]:
        suffix = a.tag_suffix
        if suffix is None:
            suffix = '' if a.dataset == 'text' else ('_la' if a.loss == 'la' else '')
            if a.if_factor:
                suffix += f'_if{a.if_factor}'
        tag = f'{a.dataset}_K{a.n_clients}_dir{a.dirichlet:g}{suffix}_seed{seed}'
        if os.path.exists(os.path.join(OUT, 'cache', tag + '.npz')):
            print(f'[v3_train] {tag} cached, skipping', flush=True)
            continue
        print(f'[v3_train] {tag} device={device} rounds={rounds} lr={lr}', flush=True)
        if a.dataset == 'text':
            out = train_text(seed, a.n_clients, a.dirichlet, device, rounds, lr)
        else:
            out = train_images(a.dataset, seed, a.n_clients, a.dirichlet, device, rounds, lr,
                               loss=a.loss, if_factor=a.if_factor)
        cal_probs, cal_labels, cal_client, test_probs, test_labels, n_train, secs = out
        assert (cal_client >= 0).all()
        np.savez_compressed(os.path.join(OUT, 'cache', tag + '.npz'), cal_probs=cal_probs,
                            cal_labels=cal_labels, cal_client=cal_client,
                            test_probs=test_probs, test_labels=test_labels)
        row = accuracy_row(tag, a.dataset, seed, a.n_clients, a.dirichlet, test_probs, test_labels, n_train, secs)
        path = os.path.join(OUT, 'accuracy.csv')
        new = not os.path.exists(path)
        with open(path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"[v3_train] {tag} DONE top1={row['top1']} tail={row['top1_tail']} worst={row['top1_worst']} ({secs:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
