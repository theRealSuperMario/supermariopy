#!/usr/bin/env python3
import glob
import os
import pandas as pd
import click
import pprint

import ast


class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


@click.command()
@click.argument("files2stack")
@click.option("--range-list", cls=PythonLiteralOption, default="[]")
@click.option("--out-name", "-o", help="output name or output directory", default=".")
def main(files2stack: str, range_list: str, out_name: str):
    """Merges multiple dataframes generated with tflogs2pandas.py into one single dataframe. This useful when a training is continued from a certain checkpoint.
    Then one can easily stack them together. It is also possible to filter the dataframe on the "step" column to only include a certain range. Dataframe files (currently only csv files are supported) must be provided as comma-seperated list.

    # stack df1.csv and df2.csv together into df_final.csv
    merge_tflog_dfs.py df1.csv,df2.csv -o df_final.csv

    # stack df1.csv and df2.csv together into df_final.csv. Take only steps within [0, 3000] from df1.csv
    merge_tflog_dfs.py df1.csv,df2.csv '[[0, 3000], [0, -1]]' -o df_final.csv
    """
    # TODO: support reading multiple file extensions

    files = files2stack.split(",")
    dfs = [pd.read_csv(f) for f in files]  # TODO: also allow other file extensions
    if range_list:
        df_sliced = []
        for (start, end), df in zip(range_list, dfs):
            if end == -1:
                end = df.step.max()
            df_sliced.append(df[df.step.between(start, end)])
        dfs = df_sliced
    df_final = pd.concat(dfs, axis=0)  # type: pd.DataFrame
    if os.path.isdir(out_name):
        out_name = os.path.join(out_name, "df_merge.csv")
    df_final.to_csv(out_name, index=False)


if __name__ == "__main__":
    main()
