* CI testing is done on [gitlab](https://gitlab.com/theRealSuperMario/supermariopy).


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


# Suggested Imports

```python
from supermariopy import numpyutils as npu
from supermariopy import tfutils as tfu
```


# TODO
 - [ ] Use type hints ore do not use them at all. right now, it is very mixed.
