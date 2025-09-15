"""
FIRM experiment runner.

Loads configs/firm.yaml and dispatches to src.deephedge.train_firm:main.
"""
from pathlib import Path
from omegaconf import OmegaConf


def load_config() -> dict:
    cfg_path = Path("configs/firm.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError("configs/firm.yaml not found")
    cfg = OmegaConf.load(str(cfg_path))
    return OmegaConf.to_container(cfg, resolve=True)


def main():
    from src.deephedge.train_firm import main as train_main

    config = load_config()
    train_main(config)


if __name__ == "__main__":
    main()

