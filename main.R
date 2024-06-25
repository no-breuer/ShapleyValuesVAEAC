library(shapr)
library(ranger)
library(data.table)
library(yaml)
library(reticulate)

user_config <- yaml.load_file("config/user_config.yaml")
model_config <- yaml.load_file("config/model_config.yaml")

directory_code <- user_config$directory_code
directory_python <- user_config$directory_python

seed <- model_config$seed
data_dim <- model_config$data_dim
latent_dim <- model_config$latent_dim
relevant_latents <- model_config$relevant_latents
latent_case <- model_config$latent_case
poly_degree <- model_config$poly_degree

epochs <- model_config$epochs
batch_size <- model_config$batch_size
use_fixed_A <- model_config$use_fixed_A

n_samples <- model_config$n_samples

use_python(directory_python)

os <- import("os")
os$chdir(directory_code)

# Load the Python functions from the Python file 'data_generation.py'.
source_python(paste(directory_code, 'data_generation.py', sep = '/'))

# Load the R files needed for computing Shapley values using VAEAC.
source("Source_Shapr_VAEAC.R")

# Set seed
set.seed(seed)

data_generation_result <- execute()
A <- data_generation_result[2]
A <- as.matrix(as.data.frame(A))
top_A <- data_generation_result[3]
top_A <- as.matrix(as.data.frame(top_A))

np <- import("numpy")

data_dir <- paste("data/datasets/seed_", seed, "/intervention/polynomial_latent_", latent_case, "_poly_degree_", poly_degree, "_data_dim_", data_dim, "_latent_dim_", latent_dim, sep = "")
data_train_x <- np$load(paste(data_dir, "/train_x.npy", sep = ""))
data_train_y <- np$load(paste(data_dir, "/train_y.npy", sep = ""))
data_train <- np$load(paste(data_dir, "/train.npy", sep = ""))
data_test_x <- np$load(paste(data_dir, "/test_x.npy", sep = ""))

x_column_names <- paste("X", c(1:data_dim), sep = "")
column_names <- c("Y", x_column_names)

data_train_x <- as.data.frame(data_train_x)
colnames(data_train_x) <- x_column_names

data_train <- as.data.frame(data_train)
colnames(data_train) <- column_names

data_test_x <- as.data.frame(data_test_x)
colnames(data_test_x) <- x_column_names

# linear regression
model <- lm(formula(data_train), data_train)

# Specifying the phi_0, i.e. the expected prediction without any features.
phi_0 <- mean(data_train_y)

# Prepare the data for explanation.
explainer <- shapr(data_train_x, model)
#> The specified model provides feature classes that are NA. The classes of data are taken as the truth.

# Train the VAEAC model with specified parameters and add it to the explainer
explainer_added_vaeac = add_vaeac_to_explainer(
  explainer,
  epochs = epochs,
  width = 32L,
  depth = 3L,
  batch_size=batch_size,
  latent_dim = latent_dim,
  lr = 0.002,
  num_different_vaeac_initiate = 2L,
  epochs_initiation_phase = 2L,
  validation_iwae_num_samples = 25L,
  A=top_A,
  relevant_latents=relevant_latents,
  verbose_summary = TRUE)

# Compute the Shapley values with respect to feature dependence using
# the VAEAC_C approach with parameters defined above
explanation = explain.vaeac(data_test_x,
                            approach = "vaeac",
                            explainer = explainer_added_vaeac,
                            n_samples = n_samples,
                            prediction_zero = phi_0,
                            which_vaeac_model = "best")

# Printing the Shapley values for the test data.
# For more information about the interpretation of the values in the table, see ?shapr::explain.
shapley_values = explanation$dt

options(width=200)

print(shapley_values, n=Inf)

result_file_name = paste(seed, "_polynomial_latent_", latent_case, "_poly_degree_", poly_degree, "_data_dim_", data_dim, "_latent_dim_", latent_dim, "_use_fixed_A_", use_fixed_A, "_shapley_values.txt", sep = "")
file.create(result_file_name)
sink(result_file_name)
print(shapley_values, n=Inf)
sink()

# Finally, we plot the resulting explanations.
png("main_results.png", res = 150, height = 1000, width = 1250)
plot(explanation, plot_phi0 = FALSE)
dev.off()