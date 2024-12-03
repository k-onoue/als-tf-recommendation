# Recommendation System Using Tensor Factorization 

- Rating prediction

### Data

Before formatting

```
User Item Rating TimeStamp
196	242	3	881250949
186	302	3	891717742
22	377	1	878887116
244	51	2	880606923
166	346	1	886397596
```

After formatting

```
user_id,item_id,rating,timestamp,time_of_day
196,242,3,1997-12-04 15:55:49,Evening
186,302,3,1998-04-04 19:22:22,Night
22,377,1,1997-11-07 07:18:36,Morning
244,51,2,1997-11-27 05:02:03,Morning
166,346,1,1998-02-02 05:33:16,Morning
```

```
| Time of Day | Time          | Description                                    |
|-------------|---------------|------------------------------------------------|
| Late Night  | 0:00 ～ 5:00  | The time after midnight and before sunrise; quiet hours. |
| Morning     | 5:00 ～ 10:00 | The time from sunrise to mid-morning; people start their activities. |
| Afternoon   | 10:00 ～ 15:00| The central part of the day; work and school are in full swing. |
| Evening     | 15:00 ～ 19:00| The time when the sun begins to set; work and school often end. |
| Night       | 19:00 ～ 0:00 | Evening activities begin; time for relaxation and family. |
```

### Example Usage

```
# Load the data and fit the recommender
file_path = "./data/u.data"
recommender = TFRecommender(file_path)
recommender.fit()

# Predict the rating for a specific user, item, and time
target_user = 1
target_item = 1
target_time = "Late Night"
predicted_rating = recommender.predict_rating(target_user, target_item, target_time)
print(f"Predicted rating for (User, Item, Time) = {(target_user, target_item, target_time)}: {predicted_rating}")
# >>> Predicted rating for (User, Item, Time) = (1, 1, 'Late Night'): 4.155220625158162

# Recommend top 5 items for a specific user and time
recommended_items = recommender.recommend_items(target_user, target_time, n=5)
print(f"Top 5 recommended items for User {target_user} at {target_time}: {recommended_items}")
# >>> Top 5 recommended items for User 1 at Late Night: [483, 50, 318, 169, 127]

# Search for top 5 users for a specific item and time
best_users = recommender.search_best_users(target_item, target_time, n=5)
print(f"Top 5 users for Item {target_item} at {target_time}: {best_users}")
# >>> Top 5 users for Item 1 at Late Night: [472, 130, 312, 295, 118]
```