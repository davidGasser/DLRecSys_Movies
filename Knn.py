from collections import UserDict
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from ast import literal_eval
import re
from annoy import AnnoyIndex
from numpy.core.fromnumeric import shape
import random
import time
import pynndescent 
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

CSV_HEADER = ["user_id","occupation","gender","age","age_squared","age_power_3","city","state","longitude","latitude","timezone","watched_movies","liked","disliked",
              "watched_genres","released","movie_names","days","months","years","label"]
TARGET_FEATURE_NAME = "label"


#get data
train_data = pd.read_pickle("candidate_train_data.pkl")
test_data = pd.read_pickle("candidate_test_data.pkl")

try: 
    train_data = train_data.drop(columns=["top_k","knn"])
    test_data = test_data.drop(columns=["top_k","knn"])
except:
    pass

#print(train_data.head(10))
# print(test_data.head(10))

#train_data = train_data.drop([x for x in range(49000,49033)])

#coverting the string back to a list and then to a np.array for the tensor

def find_words(data):
    data["movie_words"] = data["movie_names"].apply(lambda x: " ".join(x))
    data["movie_words"] = data["movie_words"].apply(lambda x: re.findall(r"[\w'.]+",x)) 

find_words(train_data)
find_words(test_data)
#print(train_data["movie_words"])
# zip code from object to int
# train_data["zip_code"] = train_data["zip_code"].apply(lambda x: int(x))
# test_data["zip_code"] = test_data["zip_code"].apply(lambda x: int(x))

# print(train_data.dtypes)
# print(test_data.dtypes)

all_movie_list =[]
all_words_list =[]
def get_all_movie_names(data):
    for movie_list in data["movie_names"]:
        for name in movie_list:
            if not(name in all_movie_list):
                all_movie_list.append(name)
                all_words_list.extend(re.findall(r"[\w'.]+", name))

get_all_movie_names(train_data)
get_all_movie_names(test_data)
all_words_list = list(set(all_words_list))

city_list = []
state_list = []
def get_city_and_state(data):
    for city, state in zip(data["city"],data["state"]):
        if city not in city_list:
            city_list.append(city)
        if state not in state_list:
            state_list.append(state)
get_city_and_state(train_data)
get_city_and_state(test_data)

candidate_model = tf.keras.models.load_model("model_candidate")

def get_user_output():
    user_layer = candidate_model.layers[-2]
    user_output = user_layer.output
    #print(shape(user_output))
    model = keras.Model(inputs=candidate_model.input,outputs=user_output)
    return model

user_output = get_user_output()

movie_embeddings = candidate_model.layers[-1].get_weights()[0]
#print(shape(movie_embeddings))
#print(type(movie_embeddings))
movie_embeddings = movie_embeddings.transpose()
#print(shape(movie_embeddings))
#print(type(movie_embeddings))

train_input = {"age": train_data["age"].values,
            "age_squared":train_data["age_squared"].values,
            "age_power_3":train_data["age_power_3"].values,
            "longitude": train_data["longitude"].values,
            "latitude": train_data["latitude"].values,
            "timezone": train_data["timezone"].values,
            "city": train_data["city"].values,
            "state": train_data["state"].values,
            "gender": train_data["gender"].values,
            "occupation": train_data["occupation"].values,
            "watched_movies": tf.keras.preprocessing.sequence.pad_sequences(train_data["watched_movies"]),
            "liked": tf.keras.preprocessing.sequence.pad_sequences(train_data["liked"]),
            "disliked": tf.keras.preprocessing.sequence.pad_sequences(train_data["disliked"]),
            "watched_genres": tf.keras.preprocessing.sequence.pad_sequences(train_data["watched_genres"]),
            "released": tf.keras.preprocessing.sequence.pad_sequences(train_data["released"]),
            "days": train_data["days"].values,
            "months": train_data["months"].values,
            "years": train_data["years"].values,
            "movie_names": tf.keras.preprocessing.sequence.pad_sequences(train_data["movie_names"],dtype=object,value="yyy"),    
            "movie_words": tf.keras.preprocessing.sequence.pad_sequences(train_data["movie_words"],dtype=object,value="yyy"),      
        }

test_input = {"age": test_data["age"].values,
            "age_squared":test_data["age_squared"].values,
            "age_power_3":test_data["age_power_3"].values,
            "longitude": test_data["longitude"].values,
            "latitude": test_data["latitude"].values,
            "timezone": test_data["timezone"].values,
            "city": test_data["city"].values,
            "state": test_data["state"].values,
            "gender": test_data["gender"].values,
            "occupation": test_data["occupation"].values,
            "watched_movies": tf.keras.preprocessing.sequence.pad_sequences(test_data["watched_movies"]),
            "liked": tf.keras.preprocessing.sequence.pad_sequences(test_data["liked"]),
            "disliked": tf.keras.preprocessing.sequence.pad_sequences(test_data["disliked"]),
            "watched_genres": tf.keras.preprocessing.sequence.pad_sequences(test_data["watched_genres"]),
            "released": tf.keras.preprocessing.sequence.pad_sequences(test_data["released"]),
            "days": test_data["days"].values,
            "months": test_data["months"].values,
            "years": test_data["years"].values,
            "movie_names": tf.keras.preprocessing.sequence.pad_sequences(test_data["movie_names"],dtype=object,value="yyy"),    
            "movie_words": tf.keras.preprocessing.sequence.pad_sequences(test_data["movie_words"],dtype=object,value="yyy"),      
        }


num_trees = 350
k = 100

# #for train data
# user = user_output.predict(x=train_input)
# movie = candidate_model.predict(x=train_input)
# print(shape(movie))
# #movie 0 is not an option, hence delete
# movie = np.delete(movie,0,1)

# #set up the KNN search with annoy
# knn = AnnoyIndex(len(movie_embeddings[0,:]),"dot")
# #print(movie_embeddings.shape[0])

# #load in the vectors
# for idx in range(1,movie_embeddings.shape[0]):
#     knn.add_item(idx,movie_embeddings[idx,:])

# knn.build(num_trees)
# #knn.save("knn.ann")
# knn_list = []

# #get the closest 100 for all users
# for u in user:   
#     knn_list.append(knn.get_nns_by_vector(u,k))

# #+1 necessary because of index shift
# predicted_label = [np.argmax(i)+1 for i in movie]
# predict_data = pd.DataFrame(data=train_data["label"],columns=["label"])
# predict_data["predicted"] = predicted_label

# #get top_k rated movies form the softmax output
# values,indices = tf.math.top_k(movie,k=k)
# top_k = list(indices.numpy())
# predict_data["top_k"] = top_k
# predict_data["knn"] = knn_list
# predict_data["top_k"] = predict_data["top_k"].apply(lambda x: [i+1 for i in x])

# # calculate accuracy and prepare the data for the next step
# counter1 = 0
# counter2 = 0
# counterknn = 0

# for idx, row in predict_data.iterrows():
#     counter2 += 1
#     if row["label"] in row["top_k"]:
#         counter1 += 1
#     else:
#         row["top_k"][random.randint(0,len(row["top_k"])-1)] = row["label"]

#     if row["label"] in row["knn"]:
#         counterknn += 1
#     else:
#          row["knn"][random.randint(0,len(row["knn"])-1)] = row["label"]

# train_data["knn"] = predict_data["knn"]
# train_data["top_k"] = predict_data["top_k"]

# train_data = train_data.drop(columns=["movie_words"],axis=1)
# train_data.to_pickle("knn_train_data.pkl")

# print("accuracy topk train: ",float(counter1)/float(counter2)*100,"%")
# print("accuracy knn train: ",float(counterknn)/float(counter2)*100,"%")
# del predict_data
# del train_data
# del user
# del movie


#for the test data 
print("length of test input:", len(test_data.index))
user_start = time.time()
user = user_output.predict(x=test_input)
user_end = time.time()
print("predictions for KNN:",user_end-user_start)
movie_start = time.time()
movie = candidate_model.predict(x=test_input)
movie_end = time.time()
print("top-k prediction time:", movie_end-movie_start)
print(len(movie))
movie = np.delete(movie,0,1)

knn = AnnoyIndex(len(movie_embeddings[0,:]),"dot")
#print(movie_embeddings.shape[0])

#load in the vectors
for idx in range(1,movie_embeddings.shape[0]):
    knn.add_item(idx,movie_embeddings[idx,:])

knn.build(num_trees)
#knn.save("knn.ann")
knn_list = []

knn_get_time_start = time.time()
for u in user: 
    knn_list.append(knn.get_nns_by_vector(u,k))
knn_get_time_end = time.time()

print("knn get time:", knn_get_time_end-knn_get_time_start)

knn_list = []

knn_get_time_start = time.time()
for u in user:   
    knn_list.append(knn.get_nns_by_vector(u,k))
knn_get_time_end = time.time()

print("knn get time:", knn_get_time_end-knn_get_time_start)

predicted_label = [np.argmax(i)+1 for i in movie]
predict_data = pd.DataFrame(data=test_data["label"],columns=["label"])
predict_data["predicted"] = predicted_label

top_k_get_time_start = time.time()
values,indices = tf.math.top_k(movie,k=k)
top_k = list(indices.numpy())
top_k_get_time_end = time.time()
print("top_k_get_time:", top_k_get_time_end-top_k_get_time_start)

#print(top_k)
predict_data["top_k"] = top_k
predict_data["knn"] = knn_list
predict_data["top_k"] = predict_data["top_k"].apply(lambda x: [i+1 for i in x])
#predict_data["pynn_eucl"] = neighbors_eucl.tolist()
#predict_data["pynn_dot"] = neighbors_dot.tolist()


countertopk = 0
counter2 = 0
counterknn = 0

for idx, row in predict_data.iterrows():
    counter2 += 1
    if row["label"] in row["top_k"]:
        countertopk += 1
    else:
        row["top_k"][random.randint(0,len(row["top_k"])-1)] = row["label"]
    if row["label"] in row["knn"]:
        counterknn += 1
    else:
       row["knn"][random.randint(0,len(row["knn"])-1)] = row["label"]

test_data["knn"] = predict_data["knn"]
test_data["top_k"] = predict_data["top_k"]

test_data = test_data.drop(columns=["movie_words"],axis=1)
test_data.to_pickle("knn_test_data.pkl")

print("accuracy topk test: ",float(countertopk)/float(counter2)*100,"%")
print("accuracy knn test: ",float(counterknn)/float(counter2)*100,"%")


