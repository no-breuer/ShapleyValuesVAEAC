import argparse
import numpy as np
import ComputeImputations
import data_generation
from data_generation import *
from ComputeImputations import *

if __name__ == '__main__':

    # generate synthethic data
    obj, A, top_A = data_generation.execute()

    data_x = np.load("/home/breuer/PhD/CAGE with VAE/ShapleyValuesVAEAC/data/datasets/seed_0/intervention/polynomial_latent_scm_dense_poly_degree_2_data_dim_10_latent_dim_4/train_x.npy")
    data_z = np.load("/home/breuer/PhD/CAGE with VAE/ShapleyValuesVAEAC/data/datasets/seed_0/intervention/polynomial_latent_scm_dense_poly_degree_2_data_dim_10_latent_dim_4/train_z.npy")

    relevant_latents = A.shape[0]

    # get one hot max sizes (because right now we only deal with continous varibales it is an array of ones of feature
    # dimension; THIS IS HARD CODED RIGHT NOW CHANGE TO A FUNCTION
    one_hot_max_sizes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # unecessary variables for now
    distribution = "unknown_distribution"
    param_now = "unkown"
    path_to_save_model = "models/"

    # train VAEAC model
    results = ComputeImputations.train_VAEAC_model(data_x, distribution, param_now, path_to_save_model,
                                                   one_hot_max_sizes, top_A, relevant_latents, latent_dim=data_z.shape[1])






