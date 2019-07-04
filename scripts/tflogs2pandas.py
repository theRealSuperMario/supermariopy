#!/usr/bin/env python3

import tensorflow as tf
import glob
import os
import pandas as pd
import traceback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import click


# Extraction function
def sum_log(path):
    DEFAULT_SIZE_GUIDANCE = {
        'compressedHistograms': 1,
        'images': 1, 
        'scalars': 0, # 0 means load all
        'histograms': 1
    }
    runlog = pd.DataFrame({"metric" : [], "value" : [], "step" : []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
    #         print(tag)
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric" : [tag] * len(step), "value" : values, "step" : step}
            r = pd.DataFrame(r)
            runlog = pd.concat([runlog, r])
    #         runlog = runlog.append(r, ignore_index=True)
                # Dirty catch of DataLossError
    except:
        print('Event file possibly corrupt: {}'.format(path))
        traceback.print_exc()    
    return runlog


epilog = '''
This is a enhanced version of https://gist.github.com/ptschandl/ef67bbaa93ec67aba2cab0a7af47700b

This script exctracts variables from all logs from 
tensorflow event files ("event*"), writes them to Pandas 
and finally stores in long-format to a CSV-file or Pickle-file including
all (readable) runs of the logging directory.

Example usage:

tflogs2pandas.py . --csv --no-pkl --o converted # writes everything to csv file into folder "converted"
'''
@click.command()
@click.argument("log_dir")
@click.option("--pkl/--no-pkl", help="save to pickle file or not", default=False)
@click.option("--csv/--no-csv", help="save to csv file or not", default=True)
@click.option("--o", help="output directory", default=".")
def main(log_dir, pkl, csv, o):
    # Get all event* runs from logging_dir subdirectories
    event_paths = glob.glob(os.path.join(log_dir, "event*"))
    # Call & append
    all_log = pd.DataFrame()
    for path in event_paths:
        log = sum_log(path)
        if log is not None:
            if all_log.shape[0] == 0:
                all_log = log
            else:
                all_log = all_log.append(log, ignore_index=True)


    print(all_log.shape)
    all_log.head()    
                
    os.makedirs(o, exist_ok=True)
    if csv:
        print("saving to csv file")
        out_file = os.path.join(o, "all_training_logs_in_one_file.csv")
        print(out_file)        
        all_log.to_csv(out_file, index=None)
    if pkl:
        print("saving to pickle file")
        out_file = os.path.join(o, "all_training_logs_in_one_file.pkl")
        print(out_file)        
        all_log.to_pickle(out_file)

main()