import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

def generate_random_data(rows=10000, cols=20000, mean=0, var=1, output_path=None):
    # Generate a matrix with random Gaussian noise
    random_matrix = np.random.normal(loc=mean, scale=np.sqrt(var), size=(rows, cols))
    # Convert to DataFrame
    df_random = pd.DataFrame(random_matrix)
    
    if output_path:
        df_random.to_pickle(output_path)
        print(f"Random noise data saved to: {output_path}")
    
    return df_random

def split_random_data_and_save(random_data, output_dir):
    # Split the random data (using dummy labels since this is a null hypothesis dataset)
    labels = np.zeros(len(random_data))  # No meaningful labels, using 0s as dummy labels

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(random_data, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Save splits
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{output_dir}/train.pkl', 'wb') as f:
        pickle.dump((X_train, y_train), f)
    with open(f'{output_dir}/val.pkl', 'wb') as f:
        pickle.dump((X_val, y_val), f)
    with open(f'{output_dir}/test.pkl', 'wb') as f:
        pickle.dump((X_test, y_test), f)
    print(f"Train, validation, and test sets for random data saved to: {output_dir}")

# Example usage:
random_data_output_path = 'data/random_data/random_matrix_10000_20000.pkl'
random_data = generate_random_data(output_path=random_data_output_path)
split_random_data_and_save(random_data, 'data/random_data/train_test_split')
