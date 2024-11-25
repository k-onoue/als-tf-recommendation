import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, file_path, test_size=0.2, random_state=42, nrows=None):
        self.file_path = file_path
        self.columns = ["user_id", "item_id", "rating", "timestamp"]
        self.test_size = test_size
        self.random_state = random_state
        self.nrows = nrows
        self.data = None
        self.train_data = None
        self.test_data = None
        self._load_and_split_data()

    def _load_and_split_data(self):
        self.data = pd.read_csv(
            self.file_path,
            sep="\t",
            names=self.columns,
            engine="python",
            nrows=self.nrows,
        )

        # Convert 'timestamp' to a human-readable datetime format
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"], unit="s")

        self.data["time_of_day"] = self.data["timestamp"].dt.hour.apply(
            self._classify_time_of_day
        )

        self.train_data, self.test_data = train_test_split(
            self.data, test_size=self.test_size, random_state=self.random_state
        )

    def _classify_time_of_day(self, hour):
        if 0 <= hour < 5:
            return "Late Night"
        elif 5 <= hour < 10:
            return "Morning"
        elif 10 <= hour < 15:
            return "Afternoon"
        elif 15 <= hour < 19:
            return "Evening"
        elif 19 <= hour < 24:
            return "Night"

    def display_shape(self):
        if self.data is not None:
            print(f"Data shape: {self.data.shape}")
            print(f"Train data shape: {self.train_data.shape}")
            print(f"Test data shape: {self.test_data.shape}")
        else:
            print("Data not loaded yet.")


# if __name__ == "__main__":
#     # Usage
#     file_path = "./data/u.data"
#     loader = DataLoader(file_path, nrows=100)
#     loader.display_shape()
#     print(f"Train data shape: {loader.train_data.shape}")
#     print(f"Test data shape: {loader.test_data.shape}")
