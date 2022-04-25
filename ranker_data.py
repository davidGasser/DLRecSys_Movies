import numpy as np
from numpy.lib.shape_base import split
import pandas as pd
import tensorflow as tf
from pyzipcode import ZipCodeDatabase
from datetime import datetime
import copy
from ast import literal_eval


#for orientation, no practical use
CSV_HEADER = ["user_id","occupation","gender","age","age_squared","age_power_3","city","state","longitude","latitude","timezone","watched_movies",
            "liked","disliked","watched_genres","released","movie_names","days","months","years","label","knn","top_k"]
CSV_ITEM = ["item_id","movie_released","movie_name","covered_genres"]


#get data
train_data = pd.read_pickle("knn_train_data.pkl")
test_data = pd.read_pickle("knn_test_data.pkl")

item_data = pd.read_pickle("item_data.pkl")

#print("shape train_data:", train_data.shape)
#print("shape test_data:", test_data.shape)

#join the data so there are no distribution problems
data = pd.concat([train_data,test_data])
data = data.reset_index(drop=True)

#for oversampling if there is not enough space on the eikon
#data = data[:int(len(data.index)/1)]

#load item data for later, will be added to the rest
#print("shape data:", data.shape)

#print(train_data.head(10))
#print(test_data.head(10))
#print(data)

#train_data = train_data.drop([x for x in range(49000,49033)])

#coverting the string back to a list and then to a np.array for the tensor

def split_and_multiplicate(data):
    data_repeated = data.drop(columns=["knn","top_k"],axis=1)
    data_repeated = pd.concat([data_repeated]*100)
    data_repeated.sort_index(inplace=True)
    return data_repeated

data_mul = split_and_multiplicate(data)
#print(data_mul.head(101))
#print(data["top_k"].values[0])

#format  the candidates
selected_from = "top_k"
candidate_list = []
for l in data[selected_from].values:
    candidate_list.extend(list(l))
    if len(l) != 100: print(len(l))


data_mul["item_id"] = candidate_list

#print(data_mul.head(101))
#create the new labels
label_list = []
for idx, row in data_mul.iterrows():
    if row["item_id"] == row["label"]:
        label_list.append(1)
    else:
        label_list.append(0)

data_mul["correct_movie_id"] = data_mul["label"]
data_mul["label"] = label_list

data = data_mul.merge(item_data,on="item_id")
data = data.sort_values(by=["user_id","correct_movie_id"]).reset_index(drop=True)
data = data.drop(columns=["correct_movie_id"],axis=1)

in_watched = []
in_year = []
in_genre = []
in_name = []
for idx, row in data.iterrows():
    if row["item_id"] in row["watched_movies"]: in_watched.append(1)
    else: in_watched.append(0)
    if row["movie_released"] in row["released"]: in_year.append(1)
    else: in_year.append(0)
    for g in row["covered_genres"]:
        if g in row["watched_genres"]:
            x = 1
            break
        else: x=0
    in_genre.append(x)
    if row["movie_name"] in row["movie_names"]: in_name.append(1)
    else: in_name.append(0)

data["in_watched"] = in_watched
data["in_year"] = in_year
data["in_genre"] = in_genre
data["in_name"] = in_name


data["item_id"] = data["item_id"].apply(lambda x: [x])
max_id = data["user_id"].max()
percentage_split = 0.8
print(max_id)
train_data = data.loc[data["user_id"]<int(max_id*percentage_split)]
val_data = data.loc[(data["user_id"]>=int(max_id*percentage_split)) & (data["user_id"]<int(max_id*(percentage_split+(1-percentage_split)/2)))]
test_data = data.loc[data["user_id"]>=int(max_id*(percentage_split+(1-percentage_split)/2))]

data = pd.concat([train_data,val_data])
data = data.sample(frac=1).reset_index(drop=True)

pos = data.loc[data["label"] == 1].reset_index(drop=True)
neg = data.loc[data["label"] == 0].reset_index(drop=True)

train_data_pos = pos.loc[pos.index<len(pos.index)*0.9]
print(train_data_pos.head())
train_data_neg = neg.loc[neg.index<len(neg.index)*0.9]
print(train_data_neg.head())
val_data_pos = pos.loc[pos.index>=len(pos.index)*0.9]
print(val_data_pos.head())
val_data_neg = neg.loc[neg.index>=len(neg.index)*0.9]
print(val_data_neg.head())
train_data = pd.concat([train_data_pos,train_data_neg]).sample(frac=1).reset_index(drop=True)
print(train_data.head())
val_data = pd.concat([val_data_pos,val_data_neg]).sample(frac=1).reset_index(drop=True)
print(val_data.head())

print("shape positives",(data.loc[data["label"]==1]).shape)
print("shape negatives",data.loc[data["label"]==0].shape)
#label to the end
def rearange_columns(data):
    data= data[["user_id","occupation","gender","age","age_squared","age_power_3","city","state","longitude","latitude","timezone","watched_movies","liked",
    "disliked","watched_genres","released","movie_names","item_id","movie_released","movie_name","covered_genres","days","months","years","in_watched","in_year",
    "in_genre","in_name","label"]]
    return data

train_data = rearange_columns(train_data)
val_data = rearange_columns(val_data)
test_data = rearange_columns(test_data)


#print(data.shape)
print("train",len(train_data.index))
print("val",len(val_data.index))  
print("test",len(test_data.index))

train_data_file="ranking_train_data.pkl"
val_data_file="ranking_val_data.pkl"
test_data_file="ranking_test_data.pkl"

train_data.to_pickle(train_data_file)
val_data.to_pickle(val_data_file)
test_data.to_pickle(test_data_file)
 