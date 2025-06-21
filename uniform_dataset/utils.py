import pandas as pd


def calc_dist_between_two_points(start_lat, start_lon, end_lat, end_lon):
    return ((start_lat - end_lat) ** 2 + (start_lon - end_lon) ** 2) ** 0.5


def bin_numbers(df, bins, bin_names):
    return_df = pd.DataFrame({})
    df = (df.groupby([df.index, pd.cut(df, bins, labels=bin_names)], observed=True)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=bin_names, fill_value=0)).sum()
    for name in bin_names:
        return_df[name] = [df[name]]
    return return_df
