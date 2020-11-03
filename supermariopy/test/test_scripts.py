import os

import numpy as np
import pandas as pd


class Test_merge_tflog_dfs:
    def setup_dataframes(self, tmpdir):
        data1 = {
            "metric": ["foo", "foo", "bar", "bar"],
            "value": [1.0, 1.0, 2.0, 3.0],
            "step": [0, 1, 0, 1],
        }
        data2 = {
            "metric": ["foo", "foo", "bar", "bar"],
            "value": [1.0, 1.0, 2.0, 3.0],
            "step": [2, 3, 2, 3],
        }
        df = pd.DataFrame(data1)
        df.to_csv(os.path.join(tmpdir, "data1.csv"), index=False)

        df = pd.DataFrame(data2)
        df.to_csv(os.path.join(tmpdir, "data2.csv"), index=False)

    def test_merging(self, tmpdir):
        self.setup_dataframes(tmpdir)
        files2stack = [os.path.join(tmpdir, "data{}.csv".format(i)) for i in [1, 2]]
        out_name = tmpdir
        out_name = os.path.join(out_name, "df_stacked.csv")
        import subprocess

        subprocess.call(
            [
                "python",
                "scripts/merge_tflog_dfs.py",
                ",".join(files2stack),
                "--out-name",
                out_name,
            ]
        )
        df_created = pd.read_csv(out_name)
        assert [0, 1, 2, 3] == list(np.unique(df_created.step))

        subprocess.call(
            [
                "python",
                "scripts/merge_tflog_dfs.py",
                ",".join(files2stack),
                "--out-name",
                out_name,
                "--range-list",
                "[[0, -1], [0, -1]]",
            ]
        )
        df_created = pd.read_csv(out_name)
        assert [0, 1, 2, 3] == list(np.unique(df_created.step))

    def test_merging_with_ranges(self, tmpdir):
        self.setup_dataframes(tmpdir)
        files2stack = [os.path.join(tmpdir, "data{}.csv".format(i)) for i in [1, 2]]
        out_name = tmpdir
        out_name = os.path.join(out_name, "df_stacked.csv")
        import subprocess

        subprocess.call(
            [
                "python",
                "scripts/merge_tflog_dfs.py",
                ",".join(files2stack),
                "--out-name",
                out_name,
                "--range-list",
                "[[0, 1], [2, 2]]",
            ]
        )
        df_created = pd.read_csv(out_name)
        assert [0, 1, 2] == list(np.unique(df_created.step))
