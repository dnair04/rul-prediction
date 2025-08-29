import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_path = os.path.join(DATA_DIR, "train_FD001.txt")
test_path = os.path.join(DATA_DIR, "test_FD001.txt")
rul_path = os.path.join(DATA_DIR, "RUL_FD001.txt")

col_names = ['unit_number', 'time_in_cycles'] + \
            [f'operational_setting_{i}' for i in range(1, 4)] + \
            [f'sensor_{i}' for i in range(1, 22)]

useful_sensors = [
    'sensor_2','sensor_3','sensor_4',
    'sensor_7','sensor_8','sensor_9']

def load_and_preprocess(add_rolling=False):
    train = pd.read_csv(train_path, sep=" ", header=None)
    train = train.dropna(axis=1)
    train.columns = col_names

    test = pd.read_csv(test_path, sep=" ", header=None)
    test = test.dropna(axis=1)
    test.columns = col_names

    y_test_rul = pd.read_csv(rul_path, sep=" ", header=None)
    y_test_rul = y_test_rul.dropna(axis=1)
    y_test_rul.columns = ["RUL"]

    rul = train.groupby('unit_number')['time_in_cycles'].max().reset_index()
    rul.columns = ['unit_number', 'max_cycles']
    train = train.merge(rul, on='unit_number', how='left')
    train['RUL'] = train['max_cycles'] - train['time_in_cycles']
    train.drop('max_cycles', axis=1, inplace=True)
    # Cap RUL at 125 cycles to reduce noise in early cycles
    train['RUL'] = train['RUL'].clip(upper=125)


    train = train[['unit_number', 'time_in_cycles'] + useful_sensors + ['RUL']]
    test  = test[['unit_number', 'time_in_cycles'] + useful_sensors]

    scaler = MinMaxScaler()
    train[useful_sensors] = scaler.fit_transform(train[useful_sensors])
    test[useful_sensors] = scaler.transform(test[useful_sensors])

    if add_rolling:
        for sensor in useful_sensors:
            train[f"{sensor}_roll5_mean"] = train.groupby("unit_number")[sensor].transform(lambda x: x.rolling(5, min_periods=1).mean())
            train[f"{sensor}_roll5_std"]  = train.groupby("unit_number")[sensor].transform(lambda x: x.rolling(5, min_periods=1).std())
            test[f"{sensor}_roll5_mean"] = test.groupby("unit_number")[sensor].transform(lambda x: x.rolling(5, min_periods=1).mean())
            test[f"{sensor}_roll5_std"]  = test.groupby("unit_number")[sensor].transform(lambda x: x.rolling(5, min_periods=1).std())


    train.to_csv(os.path.join(OUTPUT_DIR, "train_FD001_clean.csv"), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR, "test_FD001_clean.csv"), index=False)
    y_test_rul.to_csv(os.path.join(OUTPUT_DIR, "RUL_FD001.csv"), index=False)

    return train, test, y_test_rul


if __name__ == "__main__":
    train, test, y_test_rul = load_and_preprocess(add_rolling=True)  
    print("Preprocessing complete!")
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("RUL test shape:", y_test_rul.shape)
    print("\nSample training data:")
    print(train.head())
