import numpy as np

def log_tensor(tensor, log_file):
    np.set_printoptions(linewidth=np.inf)

    for i, x in enumerate(tensor.detach().numpy()):
        log_file.write("Instance " + str(i+1) + ": " + str(x) + "\n")
    log_file.write("\n")