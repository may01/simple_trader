"""view_onlineB.py — live dashboard viewer, rolling live view (Docker path F)."""


def main() -> None:
    from frontend.live_dashboard import LiveDashboard
    from helpers import shared_folder

    dashboard = LiveDashboard()
    dashboard.run(state_path=shared_folder() + "live_state.pkl")


if __name__ == "__main__":
    main()
