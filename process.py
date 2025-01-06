import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import pickle

# Define paths
dataset_path = "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
preprocessed_data_path = "data/processed_data.csv"

# Load the dataset
print("Loading dataset...")
data = pd.read_csv(dataset_path)

# List of columns to drop
columns_to_drop = [
    'Unnamed: 0',
    'Flow ID',
    'Src IP', 'Source IP',
    'Dst IP',
    'Source Port',
    'Destination IP',
    'Protocol',
    'Timestamp',
    'SimillarHTTP',
    'Inbound',
]

# Check for columns that exist in the dataset and drop them
columns_to_remove = list(set(columns_to_drop).intersection(set(data.columns)))
data = data.drop(columns=columns_to_remove, errors='ignore')

print("Removing duplicate rows...")
data = data.drop_duplicates()

# Strip column names of whitespace
data.columns = data.columns.str.strip()

# Check for missing 'Label' column
if "Label" not in data.columns:
    raise KeyError("The 'Label' column is missing from the dataset. Please check your data.")

# Clean missing values
data = data.dropna()

# Clean and transform the Label column
def clean_label(x):
    if x == "BENIGN":
        return 0
    else:
        return 1

# Apply clean_label to transform Label
data["Label_cleaned"] = data["Label"].apply(clean_label)

# Drop the original 'Label' and other unwanted columns
if "Destination Port" in data.columns:
    data = data.drop(columns=["Destination Port", "Label"], errors='ignore')

# Replace infinite values
print("Replacing infinite values...")
# data.replace([np.inf, -np.inf], np.nan, inplace=True)
# data.dropna(inplace=True)

###########
numeric_cols = data.select_dtypes(include=[np.number])

# Check for infinite values in numeric columns
inf_values_count = np.isinf(numeric_cols).sum().sum()
print(f"Number of infinite values in the numeric columns: {inf_values_count}")

# If there are infinite values, replace them with NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Check for missing values
missing_values_count = data.isnull().sum().sum()
print(f"Number of missing values: {missing_values_count}")
# #####


label_encoder = LabelEncoder()
data["Label_cleaned"] = label_encoder.fit_transform(data["Label_cleaned"])


# Scale features
print("Scaling features...")
scaler = MinMaxScaler()
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

#resizing the data 
benign_df = data[data["Label_cleaned"] == 0]
DDoS_df = data[data["Label_cleaned"] == 1]

#bengin_df = benign_df.head(1000)
#DDoS_df = DDoS_df.head(1000)

bengin_df = benign_df.head(2000)
DDoS_df = DDoS_df.head(2000)

data = pd.concat([bengin_df, DDoS_df], axis = 0)

data.head(10)

# Save the preprocessed data
print("Saving preprocessed data...")
os.makedirs(os.path.dirname(preprocessed_data_path), exist_ok=True)
data.to_csv(preprocessed_data_path, index=False)

print(f"Preprocessing complete. Preprocessed data saved to {preprocessed_data_path}")
