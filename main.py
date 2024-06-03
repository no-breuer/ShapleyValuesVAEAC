import argparse

import ComputeImputations
import data_generation
from data_generation import *
from ComputeImputations import *

if __name__ == '__main__':

    # generate synthethic data
    data_x, data_z, data_y, A = data_generation.execute()

    relevant_latents = A.shape[0]

    # train VAEAC model
    # TODO: the train method does need some passed arguments: data_train, distribution, param_now, path_to_save_model,
    #  one_hot_max_sizes, A, relevant_latents,

    results = ComputeImputations.train_VAEAC_model(data_x, A, relevant_latents)

    #print(results)





