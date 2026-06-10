"""view_training.py — training progress dashboard (Docker path F)."""


def main() -> None:
    from frontend.training_dashboard import TrainingDashboard
    from helpers import shared_folder

    dashboard = TrainingDashboard()
    dashboard.run(state_path=shared_folder() + "training_state.pkl")


if __name__ == "__main__":
    main()
