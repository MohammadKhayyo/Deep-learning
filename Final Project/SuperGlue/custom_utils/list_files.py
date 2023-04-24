# import OS module
import itertools
import os
import random
from itertools import permutations

import numpy as np


def combination_two_list(list_1, list_2):
    # create empty list to store the
    # combinations
    unique_combinations = []
    # Getting all permutations of list_1
    # with length of list_2
    combination = list(itertools.product(list_1, list_2))


    # for comb in permut:
    #     zipped = zip(comb, list_2)
    #     # unique_combinations = np.concatenate([unique_combinations, list(zipped)])
    #     print(list(zipped))
    #     unique_combinations.append(list(zipped))
    return combination



# Get the list of all files and directories
path = r"./pieces"
dir_list = os.listdir(path)

pair_order_list = combination_two_list(dir_list, random.sample(dir_list, 100))
#
with open('pieces_pairs_names.txt', 'w') as f:
    for item in pair_order_list:
        f.write(item[0])
        f.write(' ')
        f.write(item[1])
        f.write('\n')
