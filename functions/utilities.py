import numpy as np

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    return np.nan

##
def updateAllDataSets(function, datasets, *args):
    for df in datasets:
        function(df, *args)
