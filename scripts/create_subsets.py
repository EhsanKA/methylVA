from sklearn.model_selection import train_test_split
import pandas as pd
from methylVA.data_processing.utils import sample_data
from methylVA.data_processing.split_data import train_val_test_split
from methylVA.data_processing.scaling import scale_and_save_data
from methylVA.data_processing.utils import generate_random
import yaml


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
# Main execution

set_seed(42)


# Load or prepare data
numerical_data_path = 'data/v2/numerical_data_filtered_subset.csv'
metadata_path = 'data/v2/metadata_with_labels.csv'

print("Checking if preprocessed data exists...")

print("Loading preprocessed data from CSV files...")
numerical_data_filtered = pd.read_csv(numerical_data_path,  low_memory=False, index_col=0)
df_metadata = pd.read_csv(metadata_path,  low_memory=False, index_col=0)


for split_data_value in ['5000', '10000', '20000', '37067', 'shuffled_10000']:
    print("Starting main execution...")
    labels_encoded = df_metadata['labels_encoded']
    data, labels = sample_data(numerical_data_filtered, labels_encoded, split_data_value)
    
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(data, labels)
    scale_and_save_data((X_train, X_val, X_test, y_train, y_val, y_test), output_path='data/v2/m1')
    
    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    print(f"Test set size: {X_test.shape}")
    print("Main execution finished.")