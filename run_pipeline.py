import yaml
from pathlib import Path
from methylVA.utils.random_data import generate_random_noise_matrix, split_random_data_and_save
from methylVA.training.train_vae import train_vae
from methylVA.utils.set_seed import set_seed
import pandas as pd 
from methylVA.utils.common import load_config


# Generate Random Data Step
def generate_random_data_pipeline(config):
    random_data_config = config['random_data']
    output_dir = random_data_config['output_dir']
    matrix_name = random_data_config['matrix_name']
    labels_name = random_data_config['labels_name']
    rows = random_data_config['rows']
    cols = random_data_config['cols']
    mean = random_data_config['mean']
    variance = random_data_config['variance']
    random_state = config['set_seed']
    regenerate = random_data_config['regenerate']
    zero_cols = random_data_config.get('zero_cols', 0)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    matrix_output_path = Path(output_dir).joinpath(matrix_name)
    label_output_path = Path(output_dir).joinpath(labels_name)

    # Check if data already exists
    if matrix_output_path.exists() and not regenerate:
        print(f"Data already exists at {matrix_output_path}. Skipping generation.")
        return

    random_matrix, random_labels = generate_random_noise_matrix(rows=rows,
                                                                cols=cols,
                                                                mean=mean,
                                                                var=variance,
                                                                seed=random_state
                                                                )
    
    if zero_cols>0:
        random_matrix[:, :zero_cols] = 0

    pd.DataFrame(random_matrix).to_pickle(matrix_output_path)
    pd.DataFrame(random_labels).to_pickle(label_output_path)
    print(f"Random data generated and saved at: {matrix_output_path}, {label_output_path}")

# Split Train/Val/Test Step
def split_random_data_pipeline(config):
    split_config = config['random_train_test_split']
    input_path = split_config['input_path']
    matrix_name = split_config['matrix_name']
    labels_name = split_config['labels_name']
    output_dir = split_config['output_dir']
    test_size = split_config['test_size']
    val_size = split_config['val_size']
    random_state = config['set_seed']
    regenerate = split_config['regenerate']

    test_path = Path(output_dir).joinpath('test/features.pkl')

    # Check if data already exists
    if test_path.exists() and not regenerate:
        print(f"Data already splited at {Path(test_path).parent}. Skipping generation.")
        return

    split_random_data_and_save(input_path,
                               output_dir,
                               matrix_name,
                               labels_name,
                               test_size,
                               val_size,
                               random_state
                               )
    print(f"Train, validation, and test sets created and saved at: {output_dir}")


# # Subset Creation Step
# def create_subset_pipeline(config):
    subset_config = config['subset']
    input_path = subset_config['input_path']
    output_path = subset_config['output_path']
    size = subset_config['size']

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    create_subset(input_path, output_path, size=size)
    print(f"Subset of size {size} created and saved at: {output_path}")

# # Scaling Step
# def scale_data_pipeline(config):
    scale_config = config['scale']
    input_path = scale_config['input_path']
    output_path = scale_config['output_path']

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    scale_data(input_path, output_path)
    print(f"Scaled data saved at: {output_path}")

# # Split Train/Val/Test Step
# def split_data_pipeline(config):
    
    # split_config = config['train_test_split']
    # input_path = split_config['input_path']
    # output_dir = split_config['output_dir']
    # test_size = split_config['test_size']
    # val_size = split_config['val_size']
    # random_state = split_config['random_state']

    # train_val_test_split(input_path,
    #                      output_dir,
    #                      test_size=test_size,
    #                      val_size=val_size,random_state=random_state)
    # print(f"Train, validation, and test sets created and saved at: {output_dir}")
    pass

# Training Step
def train_vae_pipeline(config):
    train_config = config['training_vae']
    train_vae(train_config)
    print(f"VAE model training completed.")

# Run the full pipeline based on the config file provided
def run_pipeline(config_path):
    # Load the pipeline configuration
    config = load_config(config_path)
    set_seed(config['set_seed'])

    # Check and execute each step based on config file
    if 'random_data' in config:
        generate_random_data_pipeline(config)

    if 'random_train_test_split' in config:
        split_random_data_pipeline(config)

    # if 'subset' in config:
    #     create_subset_pipeline(config)

    # if 'scale' in config:
    #     scale_data_pipeline(config)

    # if 'train_test_split' in config:
    #     split_data_pipeline(config)

    if 'training_vae' in config:
        train_vae_pipeline(config)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the pipeline configuration file')
    args = parser.parse_args()

    run_pipeline(args.config)
