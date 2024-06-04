def log_tensor(tensor, log_file):
    for i, x in enumerate(tensor.detach().numpy()):
        log_file.write("Instance " + str(i+1) + ": " + str(x) + "\n")
    log_file.write("\n")