import numpy as np
import pandas as pd
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac

from loader import DataLoader


class TFRecommender:
    def __init__(self, file_path):
        self.file_path = file_path
        self.original_tensor = None
        self.user_map = None
        self.item_map = None
        self.time_map = None
        self.recovered_tensor = None
        self._load_and_preprocess()

    def _load_and_preprocess(self):
        """
        Load the data from the file and preprocess it to create a 3D tensor and mapping dictionaries.
        """
        loader = DataLoader(self.file_path, nrows=None)
        df = loader.data

        # Pivot the table with a full combination of levels
        users = df["user_id"].unique()
        items = df["item_id"].unique()
        times = df["time_of_day"].unique()

        # Create a MultiIndex to ensure all combinations exist
        full_index = pd.MultiIndex.from_product(
            [items, times], names=["item_id", "time_of_day"]
        )

        # Pivot table with missing values filled
        mul_arr = df.pivot_table(
            index="user_id",
            columns=["item_id", "time_of_day"],
            values="rating",
            fill_value=np.nan,
        ).reindex(
            columns=full_index, fill_value=np.nan
        )  # Ensure all combinations exist

        # Create mapping dictionaries for indices and their original values
        self.user_map = {idx: user for idx, user in enumerate(mul_arr.index)}
        self.item_map = {idx: item for idx, item in enumerate(items)}
        self.time_map = {idx: time for idx, time in enumerate(times)}

        # Group by time_of_day and stack each group
        tensor_list = []
        for time in times:
            # Select columns for the specific 'time_of_day'
            subset = mul_arr.xs(key=time, axis=1, level="time_of_day")
            tensor_list.append(subset.to_numpy())

        # Stack the groups along a new axis to form a 3D tensor
        self.original_tensor = np.stack(tensor_list, axis=-1)

    def fit(self, rank=2, n_iter_max=100):
        """
        Perform tensor completion using CP decomposition.
        The completed tensor is stored in self.recovered_tensor.
        """
        # Standardize the tensor (mean 0, variance 1), ignoring NaN values
        mean = np.nanmean(self.original_tensor)
        std = np.nanstd(self.original_tensor)
        standardized_tensor = (self.original_tensor - mean) / std
        
        # Create a mask for missing values
        mask_tensor = ~np.isnan(standardized_tensor)
        
        # Replace NaN values with zeros for the decomposition process
        filled_tensor = np.nan_to_num(standardized_tensor)
        
        # Perform CP decomposition with masking
        factors = parafac(filled_tensor, rank=rank, mask=mask_tensor, n_iter_max=n_iter_max)
        
        # Reconstruct the tensor from the factors
        completed_tensor = cp_to_tensor(factors)
        
        # De-standardize the tensor
        self.recovered_tensor = completed_tensor * std + mean
        
        # Ensure original values are retained in the completed tensor
        condition = ~np.isnan(self.original_tensor)
        self.recovered_tensor[condition] = self.original_tensor[condition]

    def predict_rating(self, user_id, item_id, time_of_day):
        """
        Predict rating for a given user, item, and time.
        Ratings are expected to be a real number from 1 to 5.
        """
        user_idx = list(self.user_map.values()).index(user_id)
        item_idx = list(self.item_map.values()).index(item_id)
        time_idx = list(self.time_map.values()).index(time_of_day)
        rating = self.recovered_tensor[user_idx, item_idx, time_idx]
        return float(np.clip(rating, 1, 5))

    def recommend_items(self, user_id, time_of_day, n=5):
        """
        Recommend top n items for a given user and time.
        """
        user_idx = list(self.user_map.values()).index(user_id)
        time_idx = list(self.time_map.values()).index(time_of_day)
        ratings = self.recovered_tensor[user_idx, :, time_idx]
        top_n_indices = np.argsort(ratings)[::-1][:n]
        top_n_items = [int(self.item_map[idx]) for idx in top_n_indices]
        return top_n_items
    
    def search_best_users(self, item_id, time_of_day, n=5):
        """
        Search for top n users for a given item and time.
        """
        item_idx = list(self.item_map.values()).index(item_id)
        time_idx = list(self.time_map.values()).index(time_of_day)
        ratings = self.recovered_tensor[:, item_idx, time_idx]
        top_n_indices = np.argsort(ratings)[::-1][:n]
        top_n_users = [self.user_map[idx] for idx in top_n_indices]
        return top_n_users
    


# def main():
#     file_path = "./data/u.data"
#     recommender = TFRecommender(file_path)
#     recommender.fit()

#     # Predict the rating for a specific user, item, and time
#     target_user = 1
#     target_item = 1
#     target_time = "Late Night"
#     predicted_rating = recommender.predict_rating(target_user, target_item, target_time)
#     print(f"Predicted rating for (User, Item, Time) = {(target_user, target_item, target_time)}: {predicted_rating}")

#     # Recommend top 5 items for a specific user and time
#     recommended_items = recommender.recommend_items(target_user, target_time, n=5)
#     print(f"Top 5 recommended items for User {target_user} at {target_time}: {recommended_items}")

#     # Search for top 5 users for a specific item and time
#     best_users = recommender.search_best_users(target_item, target_time, n=5)
#     print(f"Top 5 users for Item {target_item} at {target_time}: {best_users}")

# if __name__ == "__main__":
#     main()