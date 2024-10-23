from methylVA.utils.decorators import time_tracker
from sklearn.model_selection import train_test_split
import pandas as pd

@time_tracker
def train_val_test_split(data, labels, random_state=42, stratify=True):
    print("Splitting data into training, validation, and test sets...")
    # Convert labels list to pandas Series to use value_counts()
    labels_series = pd.Series(labels, index=data.index)
    
    if not stratify:
            
            X_train, X_remaining, y_train, y_remaining = train_test_split(
                  data_filtered, labels_filtered, test_size=0.3, random_state=random_state
                  )
            X_val, X_test, y_val, y_test = train_test_split(
                  X_remaining, y_remaining, test_size=0.4, random_state=random_state
                  )
            
    else:   
         
        # Filter labels to exclude classes with fewer than 20 samples
        label_counts = labels_series.value_counts()
        valid_labels = label_counts[label_counts > 20].index

        # Select only data and labels corresponding to classes with more than 20 samples
        valid_indices = labels_series.isin(valid_labels)
        data_filtered = data.loc[valid_indices]
        labels_filtered = labels_series.loc[valid_indices]

        # Perform the split with stratification on filtered data
        X_train, X_remaining, y_train, y_remaining = train_test_split(
            data_filtered, labels_filtered, test_size=0.3, random_state=random_state, stratify=labels_filtered
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_remaining, y_remaining, test_size=0.4, random_state=random_state, stratify=y_remaining
        )
    
    return X_train, X_val, X_test, y_train, y_val, y_test