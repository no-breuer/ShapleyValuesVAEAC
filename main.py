if __name__ == '__main__':

    # Input Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0,
                        help='')
    parser.add_argument('--data_dim', type=int, default=200,
                        help='')
    parser.add_argument('--latent_dim', type=int, default=10,
                        help='')
    parser.add_argument('--latent_case', type=str, default='uniform',
                        help='uniform; uniform_corr; gaussian_mixture')
    parser.add_argument('--poly_degree', type=int, default=2,
                        help='')
    parser.add_argument('--train_size', type=int, default=10000,
                        help='')
    parser.add_argument('--test_size', type=int, default=20000,
                        help='')

    print(parser)