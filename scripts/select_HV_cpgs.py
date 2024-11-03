import pandas as pd
import argparse
from methylVA.utils.common import load_config


from sklearn.model_selection import train_test_split

def select_HV_cpgs(config):

    input_dir = config['input_dir']
    thresholds = config['thresholds']
    output_dir = config['output_dir']

    
    print("Loading a sample of raw data for testing, dropping NaNs, and selecting highly variable CpGs...")
    data_files = [f'{input_dir}methyl_scores_v2_HM450k_{i}.pkl' for i in range(1, 12)]


    print("Loading raw data, dropping nans, and select highly variable cpgs ...")
    data_files = [f'{input_dir}methyl_scores_v2_HM450k_{i}.pkl' for i in range(1, 12)]
    dataframes = [pd.read_pickle(file, compression="bz2") for file in data_files]
    df = pd.concat(dataframes, axis=0)
    # sample_size = 100  # Specify the number of rows to load
    # df = pd.read_pickle(data_files[3], compression="bz2").head(sample_size)


    metadata_columns = [
        'id', 'geo_accession', 'title', 'sex', 'age', 'race', 'tissue',
        'geo_platform', 'inferred_age_Hannum', 'inferred_age_SkinBlood',
        'inferred_age_Horvath353'
    ]
    label_column = 'disease'
    sex_condition_column = 'inferred_sex'
    age_condition_column = 'inferred_age_MepiClock'

    numerical_data = df.drop(
        metadata_columns + [label_column, sex_condition_column, age_condition_column],
        axis=1
    )

    # Fix FutureWarning
    df[label_column] = df[label_column].fillna('no_label')

    # Fix PerformanceWarning
    labels_encoded = df[label_column].astype('category').cat.codes
    df = pd.concat([df, labels_encoded.rename('labels_encoded')], axis=1)
    df = df.reset_index()

    nan_percentage = numerical_data.isna().sum() / numerical_data.shape[0] * 100
    selected_columns = nan_percentage[nan_percentage < 10].index.tolist()
    numerical_data_filtered = numerical_data[selected_columns]
    print("Data is processed successfully.")

    print("Creating metadata with labels ...")
    metadata_columns_with_labels = metadata_columns + [label_column, sex_condition_column, age_condition_column, 'labels_encoded']
    df_metadata = df[metadata_columns_with_labels]


    print("Splitting the data to train and test and select the variable features based on the train data.")
    data_train, data_test, meta_data_train, meta_data_test = train_test_split(
        numerical_data_filtered, df_metadata, test_size=0.1, random_state=42, stratify=df_metadata['labels_encoded']
    )

    print("Calculating column variances ...")
    column_variances = data_train.var()


    for threshold in thresholds:
        print(f"Number of columns with variance > {threshold}: {(column_variances > threshold).sum()}")

    print(" ***** Descriptive statistics of column variances *****")
    print(f"Mean: {column_variances.mean()}")
    print(f"Median: {column_variances.median()}")
    print(f"Min: {column_variances.min()}")
    print(f"Max: {column_variances.max()}")
    print(f"Standard deviation: {column_variances.std()}")
    print(f"Variance: {column_variances.var()}")

    for threshold in thresholds:
        print(f"Saving train data with variance > {threshold} ...")
        data_train[column_variances.index[(column_variances>threshold)]].to_csv(f'{output_dir}train_data_filtered_{threshold}.csv')
        print(f"Saving test data with variance > {threshold} ...")
        data_test[column_variances.index[(column_variances>threshold)]].to_csv(f'{output_dir}test_data_filtered_{threshold}.csv')
        
    print("Saving train and test metadata with labels ...")
    meta_data_train.to_csv(f'{output_dir}train_metadata_with_labels.csv')
    meta_data_test.to_csv(f'{output_dir}test_metadata_with_labels.csv')





if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', default="../methylVA/configs/config_hv_cpg_selection.yaml" , type=str, required=True)
    args = argparser.parse_args()
    config = load_config(args.config)
    select_HV_cpgs(config['hvcpg_selection'])

