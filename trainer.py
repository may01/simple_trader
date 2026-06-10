"""trainer.py — CLI entry point. Maps compose argv mode to training.Trainer RUN_TYPE."""

import os
import sys

# compose command argv → Trainer RUN_TYPE
_MODE_MAP = {
    "grab_data": "grab_data",
    "generate_full_ohlc": "prepare_data",
    "simulate": "simulate",
    "group_nn": "prepare_data",   # UNRESOLVED: no dedicated RUN_TYPE for NN grouping;
                                  # plan README "Unresolved" section — revisit with NN target labeling
    "nn_train": "train_nn",
    "simulate_nn": "simulate_nn",
    "full": "full",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in _MODE_MAP:
        valid = ", ".join(sorted(_MODE_MAP))
        print(f"usage: trainer.py <{valid}>", file=sys.stderr)
        sys.exit(2)

    os.environ["RUN_TYPE"] = _MODE_MAP[sys.argv[1]]

    from training.trainer import Trainer
    Trainer(config_path="configs/").run()


if __name__ == "__main__":
    main()
