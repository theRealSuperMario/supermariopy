# Scripts

* tflogs2pands.py
    * script that converts all tensorflow logs events from a directory to a pandas dataframe and then saves it as csv or pickle

# Notebooks

* tflogs2pandas
    * example notebook for plotting from csv files generated with tflogs2pands.py


## Runing tests

* with pytest-mpl
```bast
py.test --mpl --mpl-baseline-path=baseline

py.test --mpl-generate-path=baseline
```
