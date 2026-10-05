"""view_full.py — historical data viewer (Docker path F)."""


def main() -> None:
    import os

    import pandas as pd

    from data import (
        join_cb_overshoot, join_cb_zone_atr, join_ema_slope, join_ext_done,
        join_action_zones, join_candle_bounds, join_candle_bounds_nc,
        join_ev_line, join_ev_reach, join_nn_results, join_zone_profitability,
    )
    from frontend.data_viewer import FullData
    from frontend.history_dashboard import HistoryDashboard
    from helpers import wide_df_path

    path = wide_df_path()
    df = pd.read_pickle(path)
    # Surface NN inference outputs (nn_res_*) in the viewer by left-joining the
    # batch artifact df_with_nn.pkl from the dataset dir. Absence-safe no-op.
    df = join_nn_results(df, os.path.dirname(path))
    # Surface action-zones overlay columns (az_*) in the viewer by left-joining
    # the batch artifact df_with_action_zones.pkl from the dataset dir.
    # Absence-safe no-op.
    df = join_action_zones(df, os.path.dirname(path))
    # Surface predicted candle-bound overlay columns (cb_*) in the viewer by
    # left-joining the batch artifact df_with_candle_bounds.pkl from the dataset
    # dir. Absence-safe no-op.
    df = join_candle_bounds(df, os.path.dirname(path))
    # Non-closed (forming-candle) bound predictions (cbnc_*/cbx_*), from the
    # batch artifact df_with_candle_bounds_nc.pkl. Absence-safe no-op.
    df = join_candle_bounds_nc(df, os.path.dirname(path))
    # Zone-profitability shifted entry/target zones (zp_*), from the batch
    # artifact df_with_zone_profitability.pkl. Absence-safe no-op.
    df = join_zone_profitability(df, os.path.dirname(path))
    # EV-optimal entry levels between the two candle-bound bands (ev_*), from
    # the batch artifact df_with_ev_line.pkl. Absence-safe no-op.
    df = join_ev_line(df, os.path.dirname(path))
    # Empirical-reach entry rule (evr_*), from df_with_ev_reach.pkl.
    # Absence-safe no-op.
    df = join_ev_reach(df, os.path.dirname(path))
    # ATR-shifted entry zones (cbatr_*), from df_with_cb_zone_atr.pkl.
    # Absence-safe no-op.
    df = join_cb_zone_atr(df, os.path.dirname(path))
    # "Extreme done" state + first-trigger markers (extdone_*), from
    # df_with_ext_done.pkl. Absence-safe no-op.
    df = join_ext_done(df, os.path.dirname(path))
    # cb marker overshoot filter (cbover_*), from df_with_cb_overshoot.pkl.
    # Absence-safe no-op.
    df = join_cb_overshoot(df, os.path.dirname(path))
    # EMA-25 slope classes (ema_25_rise/fall/neutral), from
    # df_with_ema_slope.pkl. Absence-safe no-op.
    df = join_ema_slope(df, os.path.dirname(path))
    dashboard = HistoryDashboard(FullData(df))
    dashboard.run()


if __name__ == "__main__":
    main()
