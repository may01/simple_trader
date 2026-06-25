"""view_full.py — historical data viewer (Docker path F)."""


def main() -> None:
    import os

    import pandas as pd

    from data import join_nn_results
    from frontend.data_viewer import FullData
    from frontend.history_dashboard import HistoryDashboard
    from helpers import wide_df_path

    path = wide_df_path()
    df = pd.read_pickle(path)
    # Surface NN inference outputs (nn_res_*) in the viewer by left-joining the
    # batch artifact df_with_nn.pkl from the dataset dir. Absence-safe no-op.
    df = join_nn_results(df, os.path.dirname(path))
    dashboard = HistoryDashboard(FullData(df))
    dashboard.run()


if __name__ == "__main__":
    main()
