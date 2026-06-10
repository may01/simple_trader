"""view_full.py — historical data viewer (Docker path F)."""


def main() -> None:
    import pandas as pd

    from frontend.data_viewer import DataViewer, FullData
    from helpers import wide_df_path

    df = pd.read_pickle(wide_df_path())
    viewer = DataViewer(FullData(df))
    viewer.view_full()


if __name__ == "__main__":
    main()
