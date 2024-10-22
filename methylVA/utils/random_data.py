import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from methylVA.utils.decorators import time_tracker

# Generate a 10000x20000 matrix of random noise from Gaussian distribution with mean=0 and var=1
@time_tracker
def generate_random_noise_matrix(rows=10000, cols=20000, mean=0, var=1, seed=42):
    np.random.seed(seed)
    print(f"Generating random noise matrix and labels of shape ({rows}, {cols}) with mean={mean} and variance={var}...")
    matrix = np.random.normal(loc=mean, scale=np.sqrt(var), size=(rows, cols))
    labels = np.zeros(rows)
    print("Random noise matrix and label generated successfully.")
    return matrix, labels

# # Example usage
# noise_matrix = generate_random_noise_matrix()


def split_random_data_and_save(input_path, output_dir, matrix_name,
                                labels_name, test_size=0.3, val_size=0.4, random_state=42):
    # Split the random data (using dummy labels since this is a null hypothesis dataset)
    matrix_path = Path(input_path).joinpath(matrix_name)
    labels_path = Path(input_path).joinpath(labels_name)

    random_data = pd.read_pickle(matrix_path)
    labels = pd.read_pickle(labels_path)  # No meaningful labels, using 0s as dummy labels

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(random_data, labels, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)
    
    # Save splits
    split_dirs = ['train', 'val', 'test']
    for split, data, label in zip(split_dirs, [X_train, X_val, X_test], [y_train, y_val, y_test]):
        Path(output_dir).joinpath(split).mkdir(parents=True, exist_ok=True)

        with open(Path(output_dir).joinpath(split).joinpath('features.pkl'), 'wb') as f:
            pickle.dump(data, f)
        with open(Path(output_dir).joinpath(split).joinpath('labels.pkl'), 'wb') as f:
            pickle.dump(label, f)
    
    print(f"Random data splits saved at: {output_dir}")

        # data.to_pickle(Path(output_dir).joinpath(split).joinpath('features.pkl'))
        # label.to_pickle(Path(output_dir).joinpath(split).joinpath('labels.pkl'))


    # # Save splits
    # Path(output_dir).mkdir(parents=True, exist_ok=True)
    # with open(f'{output_dir}/train.pkl', 'wb') as f:
    #     pickle.dump((X_train, y_train), f)
    # with open(f'{output_dir}/val.pkl', 'wb') as f:
    #     pickle.dump((X_val, y_val), f)
    # with open(f'{output_dir}/test.pkl', 'wb') as f:
    #     pickle.dump((X_test, y_test), f)
    # print(f"Train, validation, and test sets for random data saved to: {output_dir}")

# # Example usage:
# random_data_output_path = 'data/random_data/random_matrix_10000_20000.pkl'
# random_data = generate_random_noise_matrix()
# split_random_data_and_save(random_data, 'data/random_data/train_test_split')