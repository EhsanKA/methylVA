import numpy as np
from methylVA.utils.decorators import time_tracker
from sklearn.model_selection import StratifiedShuffleSplit

# Generate a 10000x20000 matrix of random noise from Gaussian distribution with mean=0 and var=1
@time_tracker
def generate_random_noise_matrix(rows=10000, cols=20000, mean=0, var=1, seed=42):
    np.random.seed(seed)
    print(f"Generating random noise matrix of shape ({rows}, {cols}) with mean={mean} and variance={var}...")
    matrix = np.random.normal(loc=mean, scale=np.sqrt(var), size=(rows, cols))
    print("Random noise matrix generated successfully.")
    return matrix

# Example usage
noise_matrix = generate_random_noise_matrix()


@time_tracker
def sample_data(numerical_data_filtered, labels_encoded, split_size, random_state=42):
    if split_size == 'shuffled_10000':
        return generate_shuffled_data(numerical_data_filtered, 10000, random_state)
    else:
        size = int(split_size)
        if size < len(numerical_data_filtered):
            print(f"Splitting data with size: {size}")
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=random_state)
            for train_idx, _ in splitter.split(numerical_data_filtered, labels_encoded):
                return numerical_data_filtered.iloc[train_idx], [labels_encoded[i] for i in train_idx]
        else:
            return numerical_data_filtered, labels_encoded

@time_tracker
def generate_shuffled_data(numerical_data_filtered, size, random_state=42):
    print("Generating shuffled data for null hypothesis...")
    flattened_values = numerical_data_filtered.values.flatten()
    np.random.seed(random_state)
    np.random.shuffle(flattened_values)
    shuffled_matrix = flattened_values.reshape(numerical_data_filtered.shape)
    shuffled_data = pd.DataFrame(shuffled_matrix, columns=numerical_data_filtered.columns).sample(n=size, random_state=random_state).reset_index(drop=True)
    shuffled_labels = [0] * size  # Use dummy labels to represent null hypothesis
    return shuffled_data, shuffled_labels