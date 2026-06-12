"""view_full.py — historical data viewer (Docker path F)."""


def main() -> None:
    import pandas as pd

    from frontend.data_viewer import FullData
    from frontend.history_dashboard import HistoryDashboard
    from helpers import wide_df_path

    df = pd.read_pickle(wide_df_path())
    dashboard = HistoryDashboard(FullData(df))
    dashboard.run()


if __name__ == "__main__":
    main()
