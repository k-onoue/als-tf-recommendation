import numpy as np
import pandas as pd

from loader import DataLoader
# from parafac import 


import pandas as pd
import numpy as np


class TFRecommender:
    def __init__(self, tensor, user_map, item_map, time_map):
        self.tensor = tensor
        self.user_map = user_map
        self.item_map = item_map
        self.time_map = time_map

    def fit(self):
        pass

    def predict(self, user_id, item_id, time_of_day):
        pass




def main():
    file_path = "./data/u.data"
    loader = DataLoader(file_path, nrows=None)

    df = loader.data

    # Pivot the table with a full combination of levels
    users = df['user_id'].unique()
    items = df['item_id'].unique()
    times = df['time_of_day'].unique()

    # Create a MultiIndex to ensure all combinations exist
    full_index = pd.MultiIndex.from_product(
        [items, times], names=['item_id', 'time_of_day']
    )

    # Pivot table with missing values filled
    mul_arr = df.pivot_table(
        index='user_id', 
        columns=['item_id', 'time_of_day'], 
        values='rating', 
        fill_value=np.nan
    ).reindex(columns=full_index, fill_value=np.nan)  # Ensure all combinations exist

    # Create mapping dictionaries for indices and their original values
    user_map = {idx: user for idx, user in enumerate(mul_arr.index)}
    item_map = {idx: item for idx, item in enumerate(items)}
    time_map = {idx: time for idx, time in enumerate(times)}

    # Group by time_of_day and stack each group
    tensor_list = []
    for time in times:
        # Select columns for the specific 'time_of_day'
        subset = mul_arr.xs(key=time, axis=1, level='time_of_day')
        tensor_list.append(subset.to_numpy())

    # Stack the groups along a new axis to form a 3D tensor
    tensor_3d = np.stack(tensor_list, axis=-1)

    # Print mappings
    # print(f"User mapping: {user_map}")
    # print(f"Item mapping: {item_map}")
    # print(f"Time mapping: {time_map}")

    print(f"3D Tensor shape: {tensor_3d.shape}")  # Should be (n_users, n_items, n_times)

    print(tensor_3d.isna().sum())


if __name__ == "__main__":

    data_config = {
        "file_path": "./data/u_transformed.csv",
        "columns": ["user_id", "item_id", "rating", "timestamp", "time_of_day"],
        "user_id": {
            "min": 1,
            "max": 943
        },
        "item_id": {
            "min": 1,
            "max": 1682
        },
        "rating": {
            "min": 1,
            "max": 5
        },
        "time_of_day": ["Late Night", "Morning", "Afternoon", "Evening", "Night"]
    }

    main()