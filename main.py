import argparse

import ComputeImputations
import data_generation
from data_generation import *
from ComputeImputations import *

if __name__ == '__main__':

    # generate synthethic data
    data_x, data_z, data_y, A = data_generation.execute()

    relevant_latents = A.shape[0]

    # get one hot max sizes (because right now we only deal with continous varibales it is an array of ones of feature
    # dimension
    one_hot_max_sizes = np.ones(data_x.shape[1])

    # unecessary variables for now
    distribution = "unknown_distribution"
    param_now = "unkown"
    path_to_save_model = "unknown"

    # train VAEAC model
    print("this is A")
    print(A)
    results = ComputeImputations.train_VAEAC_model(data_x, distribution, param_now, path_to_save_model,
                                                   one_hot_max_sizes, A, relevant_latents)

    #print(results)





