#!/usr/bin/env python
"""Improved version of https://stackoverflow.com/questions/39416410/python-save-different-sheets-of-an-excel-file-as-individual-excel-files"""  # noqa


import os

import click
import xlrd
from xlutils.copy import copy


@click.command()
@click.argument(
    "filename", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.argument(
    "target-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
def main(filename, target_dir):
    """Save each excel sheet of given excel file `filename` into separate excel files.

    Example:
    - Assume excel file "balance.xlsx" with sheets "2018", "2019", "2020".
    - Run `split_excel_file.py balance.xlsx .`
    - You then have "balance_2018.xls", "balance_2019.xls", "balance_2020.xls" as
    new files.

    Note:
    - Due to limited support of `xlwt` for excel file format compatibility, it is only
     possible to output ".xls" files.
     However, the input files can be ".xlsx" from more recent excel versions.
    - See https://pypi.org/project/xlwt/
    """
    f = filename
    wb = xlrd.open_workbook(f, on_demand=False)
    for sheet in wb.sheets():  # cycles through each sheet in each workbook
        newwb = copy(wb)  # makes a temp copy of that book
        newwb._Workbook__worksheets = [
            worksheet
            for worksheet in newwb._Workbook__worksheets
            if worksheet.name == sheet.name
        ]
        # brute force, strips away all other sheets apart from the sheet being looked at
        basename, extension = os.path.splitext(os.path.basename(f))
        extension = ".xls"
        savename = os.path.join(
            target_dir, "{}_{}{}".format(basename, sheet.name, extension)
        )
        newwb.save(savename)
        # saves each sheet as the original file name plus the sheet name


if __name__ == "__main__":
    main()
