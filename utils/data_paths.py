from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

MNIST_ROOT = DATA_ROOT / "mnist"
FASHION_MNIST_ROOT = DATA_ROOT / "fashion-mnist"
CIFAR10_ROOT = DATA_ROOT / "cifar"
CIFAR100_ROOT = DATA_ROOT / "cifar100"
SENT140_ROOT = DATA_ROOT / "sentiment-140"


def ensure_data_directories() -> None:
    """Create repository-local dataset directories when needed."""
    for directory in (
        MNIST_ROOT,
        FASHION_MNIST_ROOT,
        CIFAR10_ROOT,
        CIFAR100_ROOT,
        SENT140_ROOT,
    ):
        directory.mkdir(parents=True, exist_ok=True)
