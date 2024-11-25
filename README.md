# als-tf-recommendation

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

###

```
| Time of Day | Time          | Description                                    |
|-------------|---------------|------------------------------------------------|
| Late Night  | 0:00 ～ 5:00  | The time after midnight and before sunrise; quiet hours. |
| Morning     | 5:00 ～ 10:00 | The time from sunrise to mid-morning; people start their activities. |
| Afternoon   | 10:00 ～ 15:00| The central part of the day; work and school are in full swing. |
| Evening     | 15:00 ～ 19:00| The time when the sun begins to set; work and school often end. |
| Night       | 19:00 ～ 0:00 | Evening activities begin; time for relaxation and family. |
```