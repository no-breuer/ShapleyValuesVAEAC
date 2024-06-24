library(shapr)
library(ranger)
library(data.table)
library(yaml)
library(reticulate)

user_config <- yaml.load_file("config/user_config.yaml")

directory_code <- user_config$directory_code
directory_python <- user_config$directory_python
use_python(directory_python)

os = import("os")
os$chdir(directory_code)

# Load the Python functions from the Python file 'data_generation.py'.
source_python(paste(directory_code, 'data_generation.py', sep = '/'))

# Load the R files needed for computing Shapley values using VAEAC.
source("Source_Shapr_VAEAC.R")

# Set seed
set.seed(2021)

data_generation_result = execute()
A = data_generation_result[2]
top_A = data_generation_result[3]
relevant_latents = 4L # todo dynamic

np <- import("numpy")

data_train_x = np$load("data/datasets/seed_0/intervention/polynomial_latent_scm_dense_poly_degree_2_data_dim_10_latent_dim_4/train_x.npy") # todo dynamic
data_train_y = np$load("data/datasets/seed_0/intervention/polynomial_latent_scm_dense_poly_degree_2_data_dim_10_latent_dim_4/train_y.npy")
data_train = np$load("data/datasets/seed_0/intervention/polynomial_latent_scm_dense_poly_degree_2_data_dim_10_latent_dim_4/train.npy")

data_train <- as.data.frame(data_train)

# linear regression
model <- lm(formula(data_train), data_train)

# Specifying the phi_0, i.e. the expected prediction without any features.
phi_0 <- mean(data_train_y)

# Prepare the data for explanation. Diameter, ShuckedWeight, and Sex correspond to 3,6,9.
explainer <- shapr(data_train, model)
#> The specified model provides feature classes that are NA. The classes of data are taken as the truth.

# Train the VAEAC model with specified parameters and add it to the explainer
explainer_added_vaeac = add_vaeac_to_explainer(
  explainer,
  epochs = 30L,
  width = 32L,
  depth = 3L,
  latent_dim = 8L,
  lr = 0.002,
  num_different_vaeac_initiate = 2L,
  epochs_initiation_phase = 2L,
  validation_iwae_num_samples = 25L,
  A=top_A,
  relevant_latents=relevant_latents,
  verbose_summary = TRUE)

# Compute the Shapley values with respect to feature dependence using
# the VAEAC_C approach with parameters defined above
explanation = explain.vaeac(data_train,
                            approach = "vaeac",
                            explainer = explainer_added_vaeac,
                            n_samples = 250L,
                            prediction_zero = phi_0,
                            which_vaeac_model = "best")

# Printing the Shapley values for the test data.
# For more information about the interpretation of the values in the table, see ?shapr::explain.
print(explanation$dt)
#>        none   Diameter ShuckedWeight        Sex
#> 1: 9.927152  0.5514675     0.4102614  0.5386242
#> 2: 9.927152 -0.8691068    -0.5059807  1.5084370
#> 3: 9.927152 -1.1324510    -1.0110522 -0.8981503
#> 4: 9.927152  0.4321455     0.5323742 -1.1651909
#> 5: 9.927152 -1.4529236    -0.9864594  1.2636536
#> 6: 9.927152 -0.8819458    -0.5280294  1.5588355
#> 7: 9.927152 -0.2511181     0.2441703 -1.0906742
#> 8: 9.927152  0.4005953     0.2119935  0.7017644

# Finally, we plot the resulting explanations.
png("main_results.png", res = 150, height = 1000, width = 1250)
plot(explanation, plot_phi0 = FALSE)
dev.off()