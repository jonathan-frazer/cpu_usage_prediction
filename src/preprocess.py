import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)

    # Select only the required columns
    features = ['cpu_request','mem_request','cpu_limit','mem_limit','runtime_minutes','controller_kind']
    target = 'cpu_usage'

    df = df[features + [target]]

    # One-hot encode categorical column
    df = pd.get_dummies(df, columns=['controller_kind'], drop_first=True)

    # Scale numeric features
    num_cols = ['cpu_request','mem_request','cpu_limit','mem_limit','runtime_minutes']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Save processed dataset
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    import sys
    preprocess(sys.argv[1], sys.argv[2])