import pandas as pd
from sklearn.model_selection import train_test_split


def listify_dict(dict_):
    """Takes a dictionary and replaces all non-list values with a list
    with same length as the rest of the list values.
    If not list is present in the dict, every value will be turned
    into a list of length 1.
    This function is useful to convert dicts in pandas dataframe compatible dictionaries

    Parameters
    ----------
    dict_ : dict
        dictionary with list and non-list elements

    Returns
    -------
    dict
        dictionary with new elements. Lists are copied from input dict_.
        Non-list elements are replaced with lists of same length as list elements
        containint the non-list elements duplicated as often as the length of the list
        elements.

    Raises
    ------
    ValueError
        if list values are not of equal length

    Examples
    --------

    from supermariopy import pandaslib
    dict_ = {1 : 2, 3 : [4, 5]}
    pandaslib.listify_dict(dict_)
    >>> {1: [2, 2], 3: [4, 5]}

    dict_ = {1 : 2, 3: 4}
    pandaslib.listify_dict(dict_)
    >>> {1 : [2], 3 : [4]}
    """

    def len_if_list(x):
        if isinstance(x, list):
            return len(x)

    values = dict_.values()
    list_lengths = list(map(len_if_list, values))
    list_lengths = list(filter(lambda x: x is not None, list_lengths))
    list_lengths = set(list_lengths)
    if not len(list_lengths):
        # dict does not contain any list
        list_lengths.add(1)
    elif not len(list_lengths) == 1:
        raise ValueError("list values are not of equal length")
    final_list_length = list_lengths.pop()
    new_dict = {}
    for key, value in dict_.items():
        if isinstance(value, list):
            new_dict[key] = value
        else:
            new_dict[key] = [value] * final_list_length
    return new_dict


def unnest_dict(dict_, join_subkeys=True, subkey_sep="_"):
    """Takes a nested dictionary structure and unpacks the inner dictionaries into the
    topmost dictionary.

    This is useful when working with recursively nested
    dict structures and the goal is to convert them to a DataFrame.

    Parameters
    ----------
    dict_ : dict
        the nested dictionary structure to transform
    join_subkeys : bool, optional
        if the key from the partent level should be joined to the child level key.
        For example : {"test" : {"1" : 2, "3" : 4}} will result in
        {"test_1" : 2, "test_3" : 4} if join_subkeys is True.
        If False, the keys from the children will be taken.
        This could overwrite parrent elements in the worst case.
        By default True.
    subkey_sep : str, optional
        the seperating character to use for joining subkeys. By default "_".

    Returns
    -------
    dict
        new dictionary with unnested structure

    Examples
    --------
        dict_ = {1 : 2, 3 : {4 : 5, 6 : 7}}
        dict_["test"] = dict_.copy()
        new_dict = unnest_dict(dict_)
        dict_, new_dict

        # create dataframe from old dict results in dict becoming a value in the table
        df1 = pd.DataFrame(dict_)

        # creating dataframe from new dict results a nice tabular dataframe
        df = pd.DataFrame(listify_dict(new_dict))
    """

    new_dict = {}
    A = {}  # subdict that contains only non-dict values
    B = {}  # subdict that contains only dict values
    for k, v in dict_.items():
        if isinstance(v, dict):
            B[k] = v
        else:
            A[k] = v

    new_dict.update(A)
    for k, b in B.items():
        if join_subkeys:
            new_b = {subkey_sep.join([str(k), str(bk)]): bv for bk, bv in b.items()}
        else:
            new_b = b
        new_b = unnest_dict(
            new_b, join_subkeys=join_subkeys, subkey_sep=subkey_sep
        )  # recurse into subdictionary
        new_dict.update(new_b)
    return new_dict


def df_empty(columns, dtypes=None, index=None):
    """create empty dataframe from column names and specified dtypes

    Parameters
    ----------
    columns : list
        list of str specifying column names
    dtypes : list of dtypes, optional
        list of dtypes for each column
    index : bool, optional
        [description], by default None

    Returns
    -------
    df
        empty pandas dataframe

    Examples
    --------
        df = df_empty(['a', 'b'], dtypes=[np.int64, np.int64])
        print(list(df.dtypes)) # int64, int64

        df = df_empty(['a', 'b'], dtypes=None)
        print(list(df.dtypes)) # float64, float64

    References
    ----------
        Shamelessly copied from
        https://stackoverflow.com/questions/36462257/create-empty-dataframe-in-pandas-specifying-column-types # noqa
    """
    if dtypes is None:
        dtypes = [None] * len(columns)
    has_consistent_lengths = len(columns) == len(dtypes)
    if not has_consistent_lengths:
        raise ValueError("columns and dtypes have to have same length")
    df = pd.DataFrame(index=index)
    for c, d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df


def instance_level_split(df, column_name, random_seed=None, test_size=0.2):
    """split rows of dataframe on instance level (grouped by colum_name)

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to split
    column_name : str
        column name to group by
    random_seed : int, optional
        random state for splitting, by default None
    test_size : int or float, optional
        test_size in fraction (float within [0, 1]) or absolute value (int)
    Returns
    -------
    pd.DataFrame
        train_groups
    pd.DataFrame
        test_groups
    """
    groups = [df for _, df in df.groupby(column_name)]
    train_groups, test_groups = train_test_split(
        groups, test_size=test_size, random_state=random_seed
    )
    train_groups = pd.concat(train_groups)
    test_groups = pd.concat(test_groups)
    return train_groups, test_groups
