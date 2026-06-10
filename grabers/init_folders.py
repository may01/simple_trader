"""Initialize dataset folder structure.

Creates all required dataset directories on Docker volumes.
Safe to call multiple times (exist_ok=True).
"""

import os


def init_dataset_folders() -> None:
    """Create all dataset directories. Safe to call multiple times."""
    from helpers import (
        dataset_folder,
        data_folder,
        shared_folder,
        nn_folder,
        action_folder,
        nn_weights_folder,
        stats_folder,
    )

    dirs_to_create = [
        dataset_folder(),
        data_folder(),
        shared_folder(),
        nn_folder(),
        action_folder(),
        nn_weights_folder(),
        stats_folder(),
    ]

    for path in dirs_to_create:
        os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    init_dataset_folders()
