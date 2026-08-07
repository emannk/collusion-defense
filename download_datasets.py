#!/usr/bin/env python3

import argparse

from torchvision import datasets

from utils.data_paths import (
    CIFAR10_ROOT,
    CIFAR100_ROOT,
    FASHION_MNIST_ROOT,
    MNIST_ROOT,
    ensure_data_directories,
)


DOWNLOADERS = {
    "mnist": lambda: (
        datasets.MNIST(str(MNIST_ROOT), train=True, download=True),
        datasets.MNIST(str(MNIST_ROOT), train=False, download=True),
    ),
    "fashion-mnist": lambda: (
        datasets.FashionMNIST(
            str(FASHION_MNIST_ROOT),
            train=True,
            download=True,
        ),
        datasets.FashionMNIST(
            str(FASHION_MNIST_ROOT),
            train=False,
            download=True,
        ),
    ),
    "cifar": lambda: (
        datasets.CIFAR10(str(CIFAR10_ROOT), train=True, download=True),
        datasets.CIFAR10(str(CIFAR10_ROOT), train=False, download=True),
    ),
    "cifar100": lambda: (
        datasets.CIFAR100(str(CIFAR100_ROOT), train=True, download=True),
        datasets.CIFAR100(str(CIFAR100_ROOT), train=False, download=True),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets",
        nargs="+",
        choices=[*DOWNLOADERS, "all"],
    )
    args = parser.parse_args()

    ensure_data_directories()

    requested = list(DOWNLOADERS) if "all" in args.datasets else args.datasets

    for name in requested:
        print(f"Preparing {name}...")
        DOWNLOADERS[name]()
        print(f"{name}: ready")


if __name__ == "__main__":
    main()
