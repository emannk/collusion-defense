#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import sys
print("Current Python interpreter:", sys.executable)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision import datasets, transforms
import torch
import os, sys, time, copy, random, signal, json, csv
import math
from utils.sampling import (mnist_noniid_qty, mnist_noniid_dirichlet, cifar10_noniid_qty, cifar10_noniid_prob,
                            mnist_noniid_prob, cifar10_noniid_dirichlet, fmnist_noniid_qty, fmnist_noniid_dirichlet,
                            fmnist_noniid_prob, sent140_dir, sent140_qty, sent140_prob,
                            cifar100_iid, cifar100_noniid_qty, cifar100_noniid_dirichlet)
from utils.options import args_parser
from models.Update import LocalUpdateDP, LocalUpdateDPSerial, LocalUpdateNeuroMNIST, LocalUpdateNeuroMNISTSerial
from models.Nets import CNNMnist, CNNCifar_ResNet18, FastTextBinary
from models.Fed import FedWeightAvg, FedWeightAvg_noise, FedWeightAvg_random
from models.test import test_img, test_bd, test_txt, test_bd_txt
from opacus.grad_sample import GradSampleModule
from models.MnistBackdoor import (Mnist_bd, Cifar_bd, Cifar100_bd,
                                  MutableDBAMNISTTrainDataset, DBAMNISTFullTriggerTestDataset,
                                  MutableNeuroMNISTTrainDataset, NeuroMNISTFullTriggerTestDataset,
                                  get_mnist_dba_6piece_coords, get_mnist_full_trigger_coords,
                                  get_mnist_visible_trigger_coords, sample_shard_poison_indices)
from models.DeepS import DeepSight
from tensorflow_privacy.compute_noise_from_budget_lib import compute_noise
from scipy.stats import norm
from models.Krum import multi_krum
from models.Flame import flame
from models.FLShield import FLShield
from models.Freqfed import FreqFed
from models.Measa import Mesas
import torch.nn as nn
import pickle

_PREFIX = "_module."
RESULTS_ROOT = 'Results'

def get_result_root(backdoor_baseline):
    return {
        'DBA': 'Results_DBA',
        'Neurotoxin': 'Results_Neuro',
    }.get(backdoor_baseline, 'Results')

def to_float(x):
    if isinstance(x, torch.Tensor):
        # handle 0-d CPU/GPU tensors
        return x.item()
    if isinstance(x, (np.floating,)):
        return float(x)
    return float(x)  # plain int/float

def strip_prefix_dict(sd, prefix=_PREFIX):
    return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items() }

def sd_unwrapped(model):
    # Works for nn.Module and GradSampleModule
    try:
        return strip_prefix_dict(model._module.state_dict())
    except AttributeError:
        return strip_prefix_dict(model.state_dict())

def replace_inplace_activations(module: nn.Module):
    """
    Recursively replace any activation with `inplace=True` by a fresh module with `inplace=False`.
    Handles ReLU/LeakyReLU/ELU/SiLU/GELU/Hardtanh.
    """
    for name, child in module.named_children():
        needs_replace = (
            isinstance(child, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SiLU, nn.GELU, nn.Hardtanh))
            and getattr(child, "inplace", False)
        )
        if needs_replace:
            cls = child.__class__
            setattr(module, name, cls(inplace=False))
        else:
            replace_inplace_activations(child)

def assert_no_inplace(model: nn.Module):
    bad = [m for m in model.modules()
           if hasattr(m, "inplace") and getattr(m, "inplace", False)]
    assert not bad, f"Found inplace activations: {bad}"


def write_to_file(num, fname, dname, clear=False):
    file_path = os.path.join(RESULTS_ROOT, dname + fname + '.txt')
    mode = "w" if clear else "a"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode) as file:
        file.write(f"{num}\n")


CLIENT_LATENCY_FIELDS = [
    "run", "round", "client_id", "selected_slot", "role", "num_samples",

    # Final effective submission latency.
    # For malicious RING clients this becomes:
    # local training + peer wait + peer communication + collusion adjustment + upload.
    "latency_s",

    # Breakdown fields for debugging.
    "local_train_latency_s",
    "peer_wait_latency_s",
    "peer_communication_latency_s",
    "collusion_adjustment_latency_s",
    "upload_latency_s",

    "loss", "learning_rate", "attack_type", "defense",
    "dataset", "iid", "frac", "num_attacker", "dp_epsilon", "dp_clip",
]

ROUND_METRIC_FIELDS = [
    "run", "round", "dataset", "iid", "attack_type", "defense", "frac",
    "num_attacker", "dp_epsilon", "dp_clip", "lr", "PDR", "local_ep",
    "main_loss", "main_acc_test", "train_acc", "backdoor_loss", "backdoor_acc_test",
    "round_time_s", "benign_latency_mean_s", "malicious_latency_mean_s",
    "benign_latency_total_s", "malicious_latency_total_s", "benign_latency_count",
    "malicious_latency_count", "defense_malicious_retention_rate",
    "defense_benign_retention_rate", "defense_selected_count",
    "selected_client_ids", "benign_client_ids", "malicious_client_ids",
    "collusion_adjustment_latency_s",
]

def result_dir(dname):
    path = os.path.join(RESULTS_ROOT, dname)
    os.makedirs(path, exist_ok=True)
    return path

def safe_to_float(value, default=""):
    try:
        return to_float(value)
    except Exception:
        return default

def csv_value(value):
    if isinstance(value, torch.Tensor):
        return safe_to_float(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value)
    if value is None:
        return ""
    return value

def write_csv_row(filename, dname, row, fieldnames, clear=False):
    file_path = os.path.join(result_dir(dname), filename)
    mode = "w" if clear or not os.path.exists(file_path) else "a"
    with open(file_path, mode, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        writer.writerow({name: csv_value(row.get(name, "")) for name in fieldnames})

def write_json_file(filename, dname, payload):
    file_path = os.path.join(result_dir(dname), filename)
    with open(file_path, "w") as file:
        json.dump(payload, file, indent=2, sort_keys=True, default=str)

def args_to_config(args, per_run, directory_name):
    config = {}
    for key, value in vars(args).items():
        if key.startswith("last_"):
            continue
        config[key] = csv_value(value)
    config["run"] = per_run
    config["results_root"] = RESULTS_ROOT
    config["results_directory"] = os.path.join(RESULTS_ROOT, directory_name)
    return config

def append_client_latency(client_latency_rows, args, per_run, iter_num, client_id,
                          selected_slot, role, num_samples, latency_s, loss="", learning_rate=""):
    client_latency_rows.append({
        "run": per_run,
        "round": iter_num,
        "client_id": client_id,
        "selected_slot": selected_slot,
        "role": role,
        "num_samples": num_samples,

        # At first, latency_s is only local training latency.
        # Later we overwrite latency_s with effective submission latency.
        "latency_s": latency_s,
        "local_train_latency_s": latency_s,
        "peer_wait_latency_s": 0.0,
        "peer_communication_latency_s": 0.0,
        "collusion_adjustment_latency_s": 0.0,
        "upload_latency_s": 0.0,

        "loss": safe_to_float(loss),
        "learning_rate": safe_to_float(learning_rate),
        "attack_type": getattr(args, "attack_type", ""),
        "defense": getattr(args, "defense", ""),
        "dataset": getattr(args, "dataset", ""),
        "iid": getattr(args, "iid", ""),
        "frac": getattr(args, "frac", ""),
        "num_attacker": getattr(args, "num_attacker", ""),
        "dp_epsilon": getattr(args, "dp_epsilon", ""),
        "dp_clip": getattr(args, "dp_clip", ""),
    })

def latency_summary(client_latency_rows, role, field="latency_s"):
    vals = [
        safe_to_float(row.get(field), None)
        for row in client_latency_rows
        if row.get("role") == role
    ]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"mean": "", "total": 0.0, "count": 0, "max": ""}
    return {
        "mean": sum(vals) / len(vals),
        "total": sum(vals),
        "count": len(vals),
        "max": max(vals),
    }

def get_float_arg(args, name, default=0.0):
    value = getattr(args, name, default)
    if value is None or value == "":
        return float(default)
    return float(value)


def apply_submission_latency_model(client_latency_rows, args,
                                   is_collusion_round,
                                   collusion_adjustment_latency_s):
    """
    Convert local training latency into effective submission latency.

    For benign clients:
        latency_s = local training + upload

    For malicious RING/Collusion clients:
        latency_s = local training
                    + waiting for slower malicious peers
                    + peer communication
                    + collusion adjustment
                    + upload

    Because this simulator trains clients sequentially, we approximate parallel FL
    waiting time as:

        peer_wait = max_malicious_local_train_time - this_client_local_train_time

    This means all malicious clients become ready to submit after the slowest
    malicious client finishes, plus RING coordination overhead.
    """

    benign_upload_latency_s = get_float_arg(
        args,
        "benign_upload_latency_s",
        get_float_arg(args, "upload_latency_s", 0.0)
    )

    malicious_upload_latency_s = get_float_arg(
        args,
        "malicious_upload_latency_s",
        get_float_arg(args, "upload_latency_s", 0.0)
    )

    ring_peer_communication_latency_s = get_float_arg(
        args,
        "ring_peer_communication_latency_s",
        0.0
    )

    collusion_adjustment_latency_s = safe_to_float(collusion_adjustment_latency_s, 0.0)
    if collusion_adjustment_latency_s == "":
        collusion_adjustment_latency_s = 0.0

    malicious_rows = [
        row for row in client_latency_rows
        if row.get("role") == "malicious"
    ]

    max_malicious_local_train_s = 0.0
    if malicious_rows:
        max_malicious_local_train_s = max(
            float(row.get("local_train_latency_s", row.get("latency_s", 0.0)))
            for row in malicious_rows
        )

    for row in client_latency_rows:
        local_train_latency_s = float(
            row.get("local_train_latency_s", row.get("latency_s", 0.0))
        )

        if row.get("role") == "malicious" and is_collusion_round:
            peer_wait_latency_s = max(
                0.0,
                max_malicious_local_train_s - local_train_latency_s
            )

            row["peer_wait_latency_s"] = peer_wait_latency_s
            row["peer_communication_latency_s"] = ring_peer_communication_latency_s
            row["collusion_adjustment_latency_s"] = collusion_adjustment_latency_s
            row["upload_latency_s"] = malicious_upload_latency_s

            row["latency_s"] = (
                local_train_latency_s
                + peer_wait_latency_s
                + ring_peer_communication_latency_s
                + collusion_adjustment_latency_s
                + malicious_upload_latency_s
            )

        elif row.get("role") == "malicious":
            row["peer_wait_latency_s"] = 0.0
            row["peer_communication_latency_s"] = 0.0
            row["collusion_adjustment_latency_s"] = 0.0
            row["upload_latency_s"] = malicious_upload_latency_s
            row["latency_s"] = local_train_latency_s + malicious_upload_latency_s

        else:
            row["peer_wait_latency_s"] = 0.0
            row["peer_communication_latency_s"] = 0.0
            row["collusion_adjustment_latency_s"] = 0.0
            row["upload_latency_s"] = benign_upload_latency_s
            row["latency_s"] = local_train_latency_s + benign_upload_latency_s

def default_defense_stats(defense_name, users_idx, idx_benign, idx_attacker):
    selected_indices = list(range(len(users_idx)))
    return {
        "defense": defense_name,
        "selected_indices": selected_indices,
        "selected_client_ids": list(users_idx),
        "selected_count": len(selected_indices),
        "malicious_rate": 1.0 if len(idx_attacker) > 0 else 0.0,
        "benign_rate": 1.0 if len(idx_benign) > 0 else 0.0,
    }

def get_update(update, model):
    '''get the update weight'''
    update2 = {}
    for key in update:
        u = update[key]
        m = model[key]
        if u.device != m.device:
            m = m.to(u.device)
        update2[key] = u - m
    return update2

#Groups update gradients into sub-groups of size group_size, and returns the averaged update for each group 
# Defense Idea - Mimicking larger dataset
def grouped_updates(updates, lengths, group_size, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(updates))
    rng.shuffle(indices)

    grouped = []
    grouped_lengths = []
    group_members = []

    for start in range(0, len(indices), group_size):
        members = indices[start:start + group_size]
        if len(members) == 0:
            continue

        total_len = sum(lengths[i] for i in members)
        avg_update = {}

        for k in updates[members[0]].keys():
            avg_update[k] = sum(
                updates[i][k] * (lengths[i] / total_len)
                for i in members
            )

        grouped.append(avg_update)
        grouped_lengths.append(total_len)
        group_members.append(list(map(int, members)))

    return grouped, grouped_lengths, group_members

def generate_attack_updates(target_matrix, num_attackers, noise_std, CGS, clip_min=None, clip_max=None):
    """
    For attackers divided into sub-groups of size CGS, generate updates where
    each group's noise cancels out (mean-zero). If the leftover attackers form
    a group >1, their noise is also mean-centered; if only one attacker remains,
    add noise directly.

    Args:
        target_matrix: torch.Tensor or list of torch.Tensor
        num_attackers: int
        noise_std: float or list of float
        CGS: int, collusion group size
        clip_min, clip_max: optional clipping bounds

    Returns:
        attack_updates: list of torch.Tensor
    """
    if CGS == -1 or CGS >= num_attackers:
        CGS = num_attackers
    attack_updates = []
    num_full_groups = num_attackers // CGS
    leftover = num_attackers % CGS

    if isinstance(target_matrix, torch.Tensor):
        target_matrix = [target_matrix] * num_attackers
    if isinstance(noise_std, (float, int)):
        std_list = [noise_std] * num_attackers
    else:
        std_list = noise_std

    idx = 0
    # Full groups
    for _ in range(num_full_groups):
        group_indices = range(idx, idx + CGS)
        group_targets = [target_matrix[i] for i in group_indices]
        group_noises = [
            torch.normal(mean=0.0, std=std_list[i], size=group_targets[i-idx].shape)
            for i in group_indices
        ]
        group_noise_mean = sum(group_noises) / CGS
        for i, gi in enumerate(group_indices):
            update = group_targets[i] + (group_noises[i] - group_noise_mean)
            if clip_min is not None and clip_max is not None:
                update = torch.clamp(update, min=clip_min, max=clip_max)
            attack_updates.append(update)
        idx += CGS

    # Leftover group
    if leftover > 1:
        group_indices = range(idx, idx + leftover)
        group_targets = [target_matrix[i] for i in group_indices]
        group_noises = [
            torch.normal(mean=0.0, std=std_list[i], size=group_targets[i-idx].shape)
            for i in group_indices
        ]
        group_noise_mean = sum(group_noises) / leftover
        for i, gi in enumerate(group_indices):
            update = group_targets[i] + (group_noises[i] - group_noise_mean)
            if clip_min is not None and clip_max is not None:
                update = torch.clamp(update, min=clip_min, max=clip_max)
            attack_updates.append(update)

    elif leftover == 1:
        attacker_idx = idx
        tgt = target_matrix[attacker_idx]
        noise = torch.normal(mean=0.0, std=std_list[attacker_idx], size=tgt.shape)
        update = tgt + noise
        if clip_min is not None and clip_max is not None:
            update = torch.clamp(update, min=clip_min, max=clip_max)
        attack_updates.append(update)

    return attack_updates

def get_output_layer_name(glob_model):
    '''
    according to global model to get the output_layer_name
    '''
    param_keys = list(glob_model.state_dict().keys())
    output_layer_name = param_keys[-2]
    return output_layer_name

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if args.backdoor_baseline in ('DBA', 'Neurotoxin') and (args.dataset != 'mnist' or args.model != 'cnn'):
        exit('{} baseline currently supports MNIST with CNN only.'.format(args.backdoor_baseline))
    RESULTS_ROOT = get_result_root(args.backdoor_baseline)
    dict_users = {}
    dataset_train, dataset_test = None, None

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        args.num_channels = 1
        # sample users
        if args.iid == 'qty':
            dict_users = mnist_noniid_qty(dataset_train, args.num_users, int(args.thre_labels))
        elif args.iid == 'dir':
            dict_users = mnist_noniid_dirichlet(dataset_train, args.num_users, args.alpha)
        elif args.iid == 'prob':
            dict_users = mnist_noniid_prob(np.array(dataset_train.targets), args.num_users, args.num_classes, args.alpha)
        else:
            print('No vaild Non-iid type')
            exit(0)
    elif args.dataset == 'cifar':
        args.num_channels = 3
        args.num_classes = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2470, 0.2435, 0.2616))
        ])
        if args.attack:
            dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=transform)
            dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=transform)
        else:
            dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=transform)
            dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=transform)

        if args.iid == 'qty':
            dict_users = cifar10_noniid_qty(dataset_train, args.num_users, int(args.thre_labels))
        elif args.iid == 'dir':
            dict_users = cifar10_noniid_dirichlet(dataset_train, args.num_users, args.alpha)
        elif args.iid == 'prob':
            dict_users = cifar10_noniid_prob(np.array(dataset_train.targets), args.num_users, args.num_classes, args.alpha)
        else:
            print('No vaild Non-iid type')
            exit(0)
    elif args.dataset == 'cifar100':
        args.num_channels = 3
        args.num_classes = 100
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                        (0.2675, 0.2565, 0.2761))
        ])
        dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=transform)

        if args.iid == 'iid':
            dict_users = cifar100_iid(dataset_train, args.num_users)
        elif args.iid == 'qty':
            dict_users = cifar100_noniid_qty(dataset_train, args.num_users, int(args.thre_labels))
        elif args.iid == 'dir':
            dict_users = cifar100_noniid_dirichlet(dataset_train, args.num_users, args.alpha)
        elif args.iid == 'prob':
            print('CIFAR100 prob split is not supported by the reference implementation.')
            exit(0)
        else:
            print('No vaild Non-iid type')
            exit(0)
    elif args.dataset == 'fashion-mnist':
        args.num_channels = 1
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
        #print(dataset_train.__getitem__)
        if args.iid == 'qty':
            dict_users = fmnist_noniid_qty(dataset_train, args.num_users, int(args.thre_labels))
        elif args.iid == 'dir':
            dict_users = fmnist_noniid_dirichlet(dataset_train, args.num_users, args.alpha)
        elif args.iid == 'prob':
            dict_users = fmnist_noniid_prob(np.array(dataset_train.targets), args.num_users, args.num_classes, args.alpha)
        lengths = [len(samples) for samples in dict_users.values()]
    elif args.dataset == 'sent140':
        from torch.utils.data import Dataset
        class Sent140ArrayDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.as_tensor(X, dtype=torch.long)  # [N, seq_len]
                self.y = torch.as_tensor(y, dtype=torch.float32)  # [N]
            def __len__(self): return self.y.shape[0]
            def __getitem__(self, i): return self.X[i], self.y[i]

        args.num_channels = 1
        args.num_classes = getattr(args, "num_classes", 2)

        base_dir = "./data/sentiment-140"
        # load clean arrays
        trainX = np.loadtxt(os.path.join(base_dir, "sent140_0.05_0_trainX.np"), dtype=int, delimiter=",")
        trainY = np.loadtxt(os.path.join(base_dir, "sent140_0.05_0_trainY.np"), dtype=int)
        testX = np.loadtxt(os.path.join(base_dir, "sent140_0.05_0_testX.np"), dtype=int, delimiter=",")
        testY = np.loadtxt(os.path.join(base_dir, "sent140_0.05_0_testY.np"), dtype=int)
        vocab = pickle.load(open(base_dir+'/vocabGood_0.05_0.pkl', 'rb'))

        # Build dict_users on CLEAN labels
        labels = trainY
        if args.iid == 'qty':
            dict_users = sent140_qty(labels, args.num_users, 2,
                                     k_labels=1)
        elif args.iid == 'dir':
            dict_users = sent140_dir(labels, args.num_users, args.alpha, 2)
        elif args.iid == 'prob':
            dict_users = sent140_prob(labels, args.num_users, 2, q=args.alpha)
        else:
            print('No valid Non-iid type')
            exit(0)
        dict_users = {i: set(idxs) for i, idxs in dict_users.items()}
        # Start from clean dataset
        clean_N = trainX.shape[0]
        dataset_train = Sent140ArrayDataset(trainX, trainY)
        dataset_test = Sent140ArrayDataset(testX, testY)

        # edge-case backdoor
        if args.attack:
            # Load poisoned rows
            bd_dir = os.path.join(base_dir, "greek-director-backdoor")
            bX = np.loadtxt(os.path.join(bd_dir, "b_trainX_0.05_0.np"), dtype=int, delimiter=",")
            # backdoor target labels; set to 0
            bY = np.zeros(len(bX), dtype=int)
            #print('edge-case: ', len(bX))
            # Append poisoned rows to the end of the clean pool
            trainX_all = np.vstack([trainX, bX])
            trainY_all = np.concatenate([trainY, bY])
            dataset_train_extend = Sent140ArrayDataset(trainX_all, trainY_all)
            vocab = pickle.load(open(bd_dir + '/vocabFull_0.05_0.pkl', 'rb'))

            # Indices of the appended poisoned rows in the *combined* dataset
            backdoor_idx = list(range(clean_N, clean_N + len(bX)))

            # Augment ONLY the attackers’ index sets with all poisoned indices
            dict_users_attack = {i: (set(idxs) | set(backdoor_idx)) for i, idxs in dict_users.items()}
        else:
            dataset_train_extend = dataset_train  # unused when no attack
            dict_users_attack = dict_users
            backdoor_idx = []
        vocabSize = len(vocab) + 1
        args.vocabSize = vocabSize
        print('vocabSize: ', vocabSize)

    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
    output_layer_name = None
    net_glob = None
    # build model
    model_bd = None
    model_bd_test = None
    dba_trigger_pieces = None
    neuro_trigger = None
    if args.model == 'cnn' and (args.dataset == 'cifar' or args.dataset == 'cifar100'):
        net_glob = CNNCifar_ResNet18(args=args).to(args.device)
        output_layer_name = get_output_layer_name(net_glob)
        args.num_channels = 3
        if args.attack:
            if args.dataset == 'cifar':
                model_bd = Cifar_bd(train=True, poison_ratio=args.PDR)
                model_bd_test = Cifar_bd(train=False, poison_ratio=1.0)
            else:
                model_bd = Cifar100_bd(train=True, poison_ratio=args.PDR)
                model_bd_test = Cifar100_bd(train=False, poison_ratio=1.0)
            print(model_bd.__getitem__)
            print(model_bd_test.__getitem__)

    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net_glob = CNNMnist(args=args).to(args.device)
        output_layer_name = get_output_layer_name(net_glob)
        if args.attack:
            if args.backdoor_baseline == 'DBA':
                if args.dataset != 'mnist':
                    exit('DBA baseline currently supports MNIST only.')
                dba_trigger_pieces = get_mnist_dba_6piece_coords()
                model_bd_test = DBAMNISTFullTriggerTestDataset(
                    target_label=1,
                    full_trigger_coords=get_mnist_full_trigger_coords(),
                    exclude_target_label=False,
                )
            elif args.backdoor_baseline == 'Neurotoxin':
                if args.dataset != 'mnist':
                    exit('Neurotoxin baseline currently supports MNIST only.')
                neuro_trigger = get_mnist_visible_trigger_coords()
                model_bd_test = NeuroMNISTFullTriggerTestDataset(
                    target_label=1,
                    full_trigger_coords=neuro_trigger,
                    exclude_target_label=True,
                )
            else:
                model_bd = Mnist_bd(train=True, poison_ratio=args.PDR)
                model_bd_test = Mnist_bd(train=False, poison_ratio=1.0)
    elif args.dataset == 'sent140' and args.model == 'lstm':
        net_glob = FastTextBinary(vocabSize).to(args.device)
    else:
        exit('Error: unrecognized model')
    net_glob_clean = copy.deepcopy(net_glob)

    total_run = args.total_run
    round_indicator = args.round_indicator
    for per_run in range(round_indicator, round_indicator+total_run):
        net_glob = copy.deepcopy(net_glob_clean).to(args.device)
        replace_inplace_activations(net_glob)
        assert_no_inplace(net_glob)
        net_glob.train()
        # copy weights
        w_glob = sd_unwrapped(net_glob)
        all_clients = list(range(args.num_users))

        rounds = int(1 / args.frac)
        m = args.num_users * args.frac
        m = int(m)
        sizes = np.array([len(v) for v in dict_users.values()])
        acc_test = []
        Cls = LocalUpdateDPSerial if args.serial else LocalUpdateDP
        NeuroCls = LocalUpdateNeuroMNISTSerial if args.serial else LocalUpdateNeuroMNIST
        clients = [Cls(args, dataset_train, dict_users[i], attack=False, attacker=False) for i in range(args.num_users)]

        if args.attack:
            attacker_datasets = None
            if args.backdoor_baseline == 'DBA':
                if args.attack_type not in ('Input', 'Output', 'Collusion'):
                    print('No Valid attack type for DBA!======')
                    exit(0)
                attacker_datasets = [
                    MutableDBAMNISTTrainDataset(train=True, target_label=1, trans=True)
                    for _ in range(args.num_users)
                ]
                clients_attacker = [
                    Cls(args, attacker_datasets[i], dict_users[i],
                        attack=args.attack_type in ('Output', 'Collusion'), attacker=True)
                    for i in range(args.num_users)
                ]
            elif args.backdoor_baseline == 'Neurotoxin':
                if args.attack_type not in ('Input', 'Output', 'Collusion'):
                    print('No Valid attack type for Neurotoxin!======')
                    exit(0)
                attacker_datasets = [
                    MutableNeuroMNISTTrainDataset(train=True, target_label=1, trans=True)
                    for _ in range(args.num_users)
                ]
                clients_attacker = [
                    NeuroCls(args, attacker_datasets[i], dataset_train, dict_users[i],
                             mode=args.attack_type, attack=args.attack_type != 'Input', attacker=True)
                    for i in range(args.num_users)
                ]
            elif args.dataset == 'sent140':
                ds = dataset_train_extend  # clean + poisoned
                dict_for_attack = dict_users_attack  # clean shard ∪ backdoor_idx for sent140
            else:
                ds = model_bd  # prebuilt attacker dataset for other tasks
                dict_for_attack = dict_users
            if args.backdoor_baseline == 'standard':
                if args.attack_type == 'Input':
                    clients_attacker = [
                        Cls(args, ds, dict_for_attack[i], attack=False, attacker=True)
                        for i in range(args.num_users)
                    ]

                elif args.attack_type in ('Collusion', 'Output', 'C_2', 'C_3'):
                    clients_attacker = [
                        Cls(args, ds, dict_for_attack[i], attack=True, attacker=True)
                        for i in range(args.num_users)
                    ]
                else:
                    print('No Valid attack type!======')
                    exit(0)
        m, loop_index = max(int(args.frac * args.num_users), 1), int(1 / args.frac)

        first_call = True
        directory_name = '{}/{}/DP-SGD_Attack_{}_Defense_{}_frac={}_nattacker={}_iid_{}_epsilon_{}_clip_{}_lr_{}_PDR_{}_local_ep_{}_CDP_{}_random_drop_{}/'.format(args.dataset, args.iid, args.attack_type, args.defense,
                                                                        args.frac, args.num_attacker, str(args.iid), str(args.dp_epsilon), str(args.dp_clip), str(args.lr), str(args.PDR), str(args.local_ep),
                                                                                                                                                             str(args.central_noise), str(args.random_drop))
        print('directory_name: ', directory_name)
        write_json_file("run_config_{}.json".format(per_run), directory_name, args_to_config(args, per_run, directory_name))

        start_iter = 0

        m, loop_index = max(int(args.frac * args.num_users), 1), int(1 / args.frac)

        for iter in range(start_iter, args.epochs):

            t_start = time.time()
            w_locals, loss_locals, length_locals, w_updates = [], [], [], []
            client_latency_rows = []
            collusion_adjustment_latency_s = 0.0
            bd_loss = 0
            # round-robin selection
            begin_index = (iter % loop_index) * m
            end_index = begin_index + m
            idxs_users = all_clients[begin_index:end_index]
            idx_benign = []
            idx_attacker = []
            if args.attack:
                if iter < 4:
                    idx_benign = idxs_users[:]
                    idx_attacker = []
                else:
                    idx_benign = idxs_users[args.num_attacker:]
                    idx_attacker = idxs_users[:args.num_attacker]
                    print('idx_attacker: ', idx_attacker)
                    print('idx_benign: ', idx_benign)
                for idx in idx_benign:

                    local = clients[idx]
                    length_locals.append(len(local.idxs))
                    local_model = copy.deepcopy(net_glob)
                    if args.dp_mechanism != 'no_dp':
                        local_model = GradSampleModule(local_model)
                    client_t0 = time.perf_counter()
                    w, loss, blr = local.train(local_model)
                    client_latency_s = time.perf_counter() - client_t0
                    append_client_latency(client_latency_rows, args, per_run, iter, idx, len(client_latency_rows),
                                          "benign", len(local.idxs), client_latency_s, loss, blr)
                    w = strip_prefix_dict(w)
                    w_locals.append(copy.deepcopy(w))
                    w_updates.append(get_update(w, w_glob))
                    loss_locals.append(copy.deepcopy(loss))
                    for p in local_model.parameters():
                        if hasattr(p, "grad_sample"):
                            del p.grad_sample
                    del local_model, w, loss
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()

                if iter >= 4:
                    print("number of attacker:", args.num_attacker)
                    current_lr = 0
                    for slot, idx in enumerate(idx_attacker):
                        model = copy.deepcopy(net_glob)
                        if args.dp_mechanism != 'no_dp':
                            model = GradSampleModule(model)
                        if args.backdoor_baseline == 'DBA':
                            poison_indices = sample_shard_poison_indices(
                                dict_users[idx],
                                args.PDR,
                                int(args.seed + iter * args.num_users + idx),
                            )
                            attacker_datasets[idx].set_poison_spec(
                                poison_indices,
                                dba_trigger_pieces[slot % len(dba_trigger_pieces)],
                            )
                        elif args.backdoor_baseline == 'Neurotoxin':
                            poison_indices = sample_shard_poison_indices(
                                dict_users[idx],
                                args.PDR,
                                int(args.seed + iter * args.num_users + idx),
                            )
                            attacker_datasets[idx].set_poison_spec(poison_indices, neuro_trigger)
                        local = clients_attacker[idx]
                        length_locals.append(len(local.idxs))
                        client_t0 = time.perf_counter()
                        w, loss, current_lr = local.train(model)
                        client_latency_s = time.perf_counter() - client_t0
                        append_client_latency(client_latency_rows, args, per_run, iter, idx, len(client_latency_rows),
                                              "malicious", len(local.idxs), client_latency_s, loss, current_lr)
                        w = strip_prefix_dict(w)
                        w_locals.append(copy.deepcopy(w))
                        w_updates.append(get_update(w, w_glob))
                        bd_loss += loss
                        for p in model.parameters():
                            if hasattr(p, "grad_sample"):
                                del p.grad_sample
                        del w, loss, model
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                    if args.attack_type == "Collusion":
                        collusion_t0 = time.perf_counter()
                        w_attackers = w_locals[-args.num_attacker:]
                        w_attackers = [{k: v.cpu() for k, v in w.items()} for w in w_attackers]
                        w_attackers_target_scaled = []
                        diff_state = []
                        for i in range(args.num_attacker):
                            w_attackers_target_scaled.append(w_attackers[i])

                        attacker_state_dicts = [{} for _ in range(args.num_attacker)]
                        attacker_state_updates = [{} for _ in range(args.num_attacker)]
                        pre_noise_scale = compute_noise(1, 1, args.dp_epsilon, args.epochs * args.frac * args.local_ep, args.dp_delta, 1e-5)
                        for key, weight in w_attackers_target_scaled[0].items():

                            std = [0 for i in range(args.num_attacker)]

                            noise_scale = []
                            for idx in idx_attacker:
                                noise_scale.append(current_lr * args.dp_clip * np.sqrt(args.local_ep) / len(dict_users[idx]) * pre_noise_scale)
                            newstd = [max(std[i], j) for i, j in enumerate(noise_scale)]
                            values_for_this_key = [
                                w_scaled[key]
                                for w_scaled in w_attackers_target_scaled
                            ]
                            if args.attack_type == 'Collusion':
                                CGS = -1
                            else:
                                print('No valid CGS')
                                exit(0)
                            # Coordinate the noise
                            attack_updates_list = generate_attack_updates(values_for_this_key, args.num_attacker,
                                                                              noise_std=newstd, CGS = CGS,
                                                                              clip_min=None,
                                                                              clip_max=None)
                            w_glob_device = w_glob[key].device
                            attack_updates_list = [t.cpu().to(w_glob_device) for t in attack_updates_list]
                            if key == output_layer_name:
                                print('noise_scale = ', noise_scale)
                                print('std: ', std)
                                print('new_std: ', newstd)
                                print('current_lr: ', current_lr)
                            for i in range(args.num_attacker):
                                attacker_state_dicts[i][key] = attack_updates_list[i]
                                attacker_state_updates[i][key] = attack_updates_list[i] - w_glob[key]
                        del w_attackers
                        del attack_updates_list
                        collusion_adjustment_latency_s = time.perf_counter() - collusion_t0

                if iter >= 4 and args.attack_type in ('Collusion'):
                    w_locals_combine = w_locals[:-args.num_attacker] + attacker_state_dicts
                    w_updates_combine = w_updates[:-args.num_attacker] + attacker_state_updates
                else:
                    w_locals_combine = w_locals
                    w_updates_combine = w_updates

                is_collusion_round = bool(
                    args.attack
                    and iter >= 4
                    and args.attack_type == "Collusion"
                    and len(idx_attacker) > 0
                )

                apply_submission_latency_model(
                    client_latency_rows=client_latency_rows,
                    args=args,
                    is_collusion_round=is_collusion_round,
                    collusion_adjustment_latency_s=collusion_adjustment_latency_s,
                )
            else:
                idx_benign = idxs_users[:]
                for idx in idxs_users:
                    local = clients[idx]
                    length_locals.append(len(local.idxs))
                    local_model = copy.deepcopy(net_glob)
                    if args.dp_mechanism != 'no_dp':
                        local_model = GradSampleModule(local_model)
                    client_t0 = time.perf_counter()
                    w, loss, blr = local.train(local_model)
                    client_latency_s = time.perf_counter() - client_t0
                    append_client_latency(client_latency_rows, args, per_run, iter, idx, len(client_latency_rows),
                                          "benign", len(local.idxs), client_latency_s, loss, blr)
                    w = strip_prefix_dict(w)
                    w_locals.append(copy.deepcopy(w))
                    w_updates.append(get_update(w, w_glob))
                    loss_locals.append(copy.deepcopy(loss))
                    for p in local_model.parameters():
                        if hasattr(p, "grad_sample"):
                            del p.grad_sample
                    del local_model
                    torch.cuda.empty_cache()
                w_locals_combine = w_locals
                w_updates_combine = w_updates
            print('length of w_combine: ', len(w_locals_combine))
            users_idx = idx_benign + idx_attacker
            print('all users: ', users_idx)
            print('length of each client: ', length_locals)
            if args.re_weight:
                length_locals = length_locals
            else:
                length_locals = [1 for i in length_locals]
            args.num_attacker_active = len(idx_attacker)
            args.last_user_order = list(users_idx)
            args.last_defense_stats = default_defense_stats(args.defense, users_idx, idx_benign, idx_attacker)
            if args.defense == 'Deepsight':
                w_glob = DeepSight(global_model=copy.deepcopy(net_glob), user_list=users_idx, args=args, w_list=w_locals_combine,
                                   w_update=w_updates_combine, per_run=per_run, first_call=first_call, w_length=length_locals, debug=True)
            # running Krum defense on the groups of gradients
            elif args.defense == 'Krum':
                group_size = getattr(args, "group_size", 1)
                if group_size > 1:
                    grouped_updates_list, grouped_lengths, group_members = grouped_updates(
                        updates=w_updates_combine,
                        lengths=length_locals,
                        group_size=group_size,
                        seed=args.seed + iter
                    )

                    # Conservative estimate: number of malicious groups.
                    grouped_n_attackers = min(args.num_attacker, len(grouped_updates_list) // 3)

                    w_glob = multi_krum(
                        gradients=grouped_updates_list,
                        n_attackers=grouped_n_attackers,
                        args=args,
                        per_run=per_run,
                        first_call=first_call,
                        w_length=grouped_lengths,
                        global_model=copy.deepcopy(net_glob),
                        multi_k=True,
                        debug=True
                    )
                else:
                    w_glob = multi_krum(gradients=w_updates_combined, n_attackers=args.num_attacker, args=args, per_run=per_run,
                                        first_call=first_call, w_length=length_locals, global_model=copy.deepcopy(net_glob), multi_k=True, debug=True)
            elif args.defense == 'Flame':
                w_glob = flame(local_model=w_locals_combine, update_params=w_updates_combine, global_model=copy.deepcopy(net_glob), args=args, per_run=per_run,
                                    first_call=first_call, w_length=length_locals, debug=True)
            elif args.defense == 'Flshield':
                w_glob = FLShield(w_list=w_locals_combine, global_model=copy.deepcopy(net_glob), dataset_test=dataset_test,
                                  args=args, w_updates=w_updates_combine, per_run=per_run, first_call=first_call, w_length=length_locals, debug=True)
            elif args.defense == 'Freqfed':
                w_glob = FreqFed(w_updates=w_locals_combine, global_model=copy.deepcopy(net_glob), num_clients=len(idxs_users), args=args, per_run=per_run,
                                 first_call=first_call, w_length=length_locals, debug=True)
            elif args.defense == 'Mesas':
                w_glob = Mesas(params=w_locals_combine, global_model=copy.deepcopy(net_glob), args=args, per_run=per_run,
                                 first_call=first_call, w_length=length_locals, debug=True)
            else:
                print('No valid defense=====')
                args.last_defense_stats = default_defense_stats('FedAvg', users_idx, idx_benign, idx_attacker)
                w_update_filtered = w_updates_combine
                if args.central_noise == 0:
                    if args.random_drop != 0:
                        w_glob = FedWeightAvg_random(w_update_filtered, w_glob, length_locals, args.random_drop)
                    else:
                        w_glob = FedWeightAvg(w_update_filtered, w_glob, length_locals)
                else:
                    w_glob = FedWeightAvg_noise(w_update_filtered, w_glob, length_locals, args.central_noise)
            print('attackers: ', idx_attacker)
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)
            del w_locals, w_locals_combine, w_updates, w_updates_combine
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            if args.attack:
                bd_loss /= args.num_attacker
                write_to_file(bd_loss, "bd_loss_{}".format(per_run), directory_name, clear=first_call)
            loss_avg = sum(loss_locals) / len(loss_locals)
            write_to_file(loss_avg, "main_loss_{}".format(per_run), directory_name, clear=first_call)
            # test accuracy
            net_glob.eval()
            if args.dataset == 'sent140':
                acc_t, loss_t = test_txt(net_glob, dataset_test, args)
                acc_train, loss_train = test_txt(net_glob, dataset_train, args)
            else:
                acc_t, loss_t = test_img(net_glob, dataset_test, args)
                acc_train, loss_train = test_img(net_glob, dataset_train, args)

            if args.attack:
                if args.dataset == 'sent140':
                    bd_acc_test = test_bd_txt(net_glob, bX, bY, args, threshold=0.5, bs=args.bs)
                    write_to_file(bd_acc_test, "BD_Acc_test_{}".format(per_run), directory_name, clear=first_call)
                else:
                    bd_acc_test = test_bd(net_glob, model_bd_test, args)
                    write_to_file(bd_acc_test,"BD_Acc_test_{}".format(per_run), directory_name, clear=first_call)
            write_to_file(acc_t,
                              "Main_Acc_test_{}".format(per_run), directory_name, clear=first_call)
            benign_latency = latency_summary(client_latency_rows, "benign")
            malicious_latency = latency_summary(client_latency_rows, "malicious")
            defense_stats = getattr(args, "last_defense_stats", {}) or {}
            selected_indices = defense_stats.get("selected_indices", [])
            selected_client_ids = defense_stats.get("selected_client_ids")
            if selected_client_ids is None:
                selected_client_ids = [users_idx[i] for i in selected_indices if i < len(users_idx)]
            t_end = time.time()
            round_row = {
                "run": per_run,
                "round": iter,
                "dataset": args.dataset,
                "iid": args.iid,
                "attack_type": args.attack_type,
                "defense": defense_stats.get("defense", args.defense),
                "frac": args.frac,
                "num_attacker": args.num_attacker,
                "dp_epsilon": args.dp_epsilon,
                "dp_clip": args.dp_clip,
                "lr": args.lr,
                "PDR": args.PDR,
                "local_ep": args.local_ep,
                "main_loss": safe_to_float(loss_avg),
                "main_acc_test": safe_to_float(acc_t),
                "train_acc": safe_to_float(acc_train),
                "backdoor_loss": safe_to_float(bd_loss) if args.attack else "",
                "backdoor_acc_test": safe_to_float(bd_acc_test) if args.attack else "",
                "round_time_s": t_end - t_start,
                "benign_latency_mean_s": benign_latency["mean"],
                "malicious_latency_mean_s": malicious_latency["mean"],
                "benign_latency_total_s": benign_latency["total"],
                "malicious_latency_total_s": malicious_latency["total"],
                "benign_latency_count": benign_latency["count"],
                "malicious_latency_count": malicious_latency["count"],
                "defense_malicious_retention_rate": defense_stats.get("malicious_rate", ""),
                "defense_benign_retention_rate": defense_stats.get("benign_rate", ""),
                "defense_selected_count": defense_stats.get("selected_count", len(selected_indices)),
                "selected_client_ids": selected_client_ids,
                "benign_client_ids": idx_benign,
                "malicious_client_ids": idx_attacker,
                "collusion_adjustment_latency_s": collusion_adjustment_latency_s,
            }
            write_csv_row("round_metrics_{}.csv".format(per_run), directory_name, round_row,
                          ROUND_METRIC_FIELDS, clear=first_call)
            for latency_i, latency_row in enumerate(client_latency_rows):
                write_csv_row("client_latency_{}.csv".format(per_run), directory_name, latency_row,
                              CLIENT_LATENCY_FIELDS, clear=first_call and latency_i == 0)

            print("Training accuracy: {:.2f}".format(acc_train))
            print("Round {:3d},Testing accuracy: {:.2f}, Average loss: {:.2f}, Time:  {:.2f}s".format(iter, acc_t, loss_avg, t_end - t_start))
            first_call = False
            last_iter_done = iter
            acc_test.append(to_float(acc_t))
