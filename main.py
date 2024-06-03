import argparse

import ComputeImputations
import data_generation
from data_generation import *
from ComputeImputations import *

if __name__ == '__main__':

    # generate synthethic data
    x, z, y, A = data_generation.execute()

    print(A.shape)
    relevant_latents = 0

    # TODO: get the causal structure from the data A and define relevant latents

    # train VAEAC model
    # TODO: the train method does need some passed arguments: data_train, distribution, param_now, path_to_save_model,
    #  one_hot_max_sizes, A, relevant_latents,

    #results = ComputeImputations.train_VAEAC_model()

    #print(results)





