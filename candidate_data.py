import numpy as np
import pandas as pd
import tensorflow as tf
from pyzipcode import ZipCodeDatabase
from datetime import datetime
import copy

user_columns = ["user_id","age","gender","occupation","zip_code"]
user_data = pd.read_csv("ml-100k/u.user",header=None,sep="|",names=user_columns)
#print(user_data.head())

item_columns = ["item_id","movie_title","release_date","video_release_date","IMDb_URL","unkown","Action","Adventure","Animation" ,
              "Children's","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi",
              "Thriller","War","Western"]
genre_columns = ["unkown","Action","Adventure","Animation" ,"Children's","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery",
                "Romance","Sci-Fi","Thriller","War","Western"]
item_data = pd.read_csv("ml-100k/u.item",header=None,sep="|",names=item_columns,encoding="latin-1")
#print(item_data.head())

rating_columns = ["user_id","item_id","rating","timestamp"]
rating_data = pd.read_csv("ml-100k/u.data",header=None,sep="\t",names=rating_columns,encoding='latin-1')
#print(rating_data.head())

#change index
user_data["user_id"] = user_data["user_id"].apply(lambda x: x-1)
#item_data["item_id"] = item_data["item_id"].apply(lambda x: x-1)
rating_data["user_id"] = rating_data["user_id"].apply(lambda x: x-1)
#rating_data["item_id"] = rating_data["item_id"].apply(lambda x: x-1)

#add column
user_data["age"] = user_data["age"].apply(lambda x: x/100)
user_data["age_squared"] = user_data["age"].apply(lambda x: x*x)
user_data["age_power_3"] = user_data["age"].apply(lambda x: x*x*x)

def get_zip_info():
    z = ZipCodeDatabase()
    city = []
    state = []
    longitude = []
    latitude = []
    timezone = []
    for idx, row in user_data.iterrows():
        zip = row["zip_code"]
        try: 
            info = z[f"{zip}"]
            city.append(info.city)
            state.append(info.state)
            longitude.append(info.longitude)
            latitude.append(info.latitude)
            timezone.append(info.timezone)
        except: 
            city.append("unknown")
            state.append("unknown")
            longitude.append(0)
            latitude.append(0)
            timezone.append(0)
    user_data["city"] = city
    user_data["state"] = state
    user_data["longitude"] = longitude
    user_data["latitude"] = latitude
    user_data["timezone"] = timezone 
get_zip_info()

#norm longitude and latitude
max_long = (user_data["longitude"].abs()).max()
max_lat = (user_data["latitude"].abs()).max()
max_tz = (user_data["timezone"].abs()).max()
user_data["longitude"] = user_data["longitude"].apply(lambda x: abs(x/max_long))
user_data["latitude"] = user_data["latitude"].apply(lambda x: abs(x/max_lat))
user_data["timezone"] = user_data["timezone"].apply(lambda x: abs(x/max_tz))

#exceptions for unkown needed
def get_date(x):
    if x != "unknown":
        return int(x[-5:-1])
    else: return 1
def get_name(x):
    if x != "unknown":
        return x[:-6]
    else: return "unknown"
item_data["movie_released"] = item_data["movie_title"].apply(lambda x: get_date(x))
item_data["movie_name"] = item_data["movie_title"].apply(lambda x: get_name(x))

rating_data["date"] = rating_data["timestamp"].apply(lambda x: datetime.utcfromtimestamp(x).strftime("%d-%m-%Y").split("-"))
print(rating_data.head())
def all_genre_func():
    genre_list = []
    for idx, row in item_data[genre_columns].iterrows():
        user_genre_list = []
        for idx,genre in enumerate(genre_columns):
            if row[genre] != 0:
                user_genre_list.append(idx*row[genre])
        genre_list.append(user_genre_list)
    item_data["covered_genres"] = genre_list
all_genre_func()

#drop unnecessary column, for URL links are not working
item_data = item_data.drop(columns=["IMDb_URL","movie_title"],axis=1)


item_data[["item_id","movie_released","movie_name","covered_genres"]].to_pickle("item_data.pkl")
# print(user_data.head())
#print(item_data.head())
# print(rating_data.head())

#combine the data and sort it 
combined_data = rating_data.merge(item_data, on="item_id").merge(user_data, on="user_id")
combined_data = combined_data.sort_values(by=["user_id","timestamp"], ignore_index=True)

combined_data["bool_rating"] = np.where(combined_data["rating"] >=4,"liked","disliked")


print(combined_data.head(20))

user_ids = combined_data["user_id"].unique().tolist()

max_rating_num = 50

def gather_user_info():
    
    all_user_genres = []
    all_user_movies = []
    all_user_liked = []
    all_user_disliked = []
    all_user_labels = []
    all_user_released = []
    all_user_name = []
    all_user_days = []
    all_user_months = []
    all_user_years = []

    user_ids_mul = []
    mul_vector = []
    for user in user_ids:
        
        user_all = combined_data.loc[combined_data["user_id"]==user]
        #find the number of positive ratings per user
        num_pos_rating = len(user_all.loc[user_all["bool_rating"]=="liked"].index)
        if num_pos_rating <= 2: 
            mul_vector.append(0)
            continue
        #create list with the number of training data per user
        user_ids_mul = user_ids_mul+[user]*(num_pos_rating-1)
        mul_vector.append((num_pos_rating-1))
        
        user_all_genres = []
        user_all_movies = []
        user_all_liked = []
        user_all_disliked = []   
        user_all_released = []
        user_all_name = []   
        user_all_days = []  
        user_all_months = []
        user_all_years = []
        
        counter = 0
        for idx, row in user_all.iterrows():
            #keeps the max length of watched movies at 20
            if len(user_all_movies)>max_rating_num:
                    movie = user_all_movies[0]
                    user_all_movies.pop(0)
                    if movie in user_all_liked:
                        index = user_all_liked.index(movie)
                        user_all_liked.remove(movie)
                        user_all_released.pop(index)
                        user_all_name.pop(index)
                    else: user_all_disliked.remove(movie)

            if counter == 0 or row["bool_rating"] == "disliked":
                
                user_all_movies.append(row["item_id"])
                
                if row["bool_rating"] == "liked":                       
                    user_all_liked.append(row["item_id"])
                    user_all_genres = (user_all_genres + row["covered_genres"])
                    user_all_released.append(row["movie_released"])
                    user_all_name.append(row["movie_name"])
                    user_all_days.append(int(row["date"][0]))
                    user_all_months.append(int(row["date"][1]))
                    user_all_years.append(int(row["date"][2]))
                    counter = counter + 1
                    
                else:
                    user_all_disliked.append(row["item_id"])
                    
               
            else:
                #represents a row of training data
                #count genres and pick the most popular
                count_genre = [user_all_genres.count(a) for a in range(18)]
                if count_genre.count(0)>13: N = 18-count_genre.count(0)
                else: N=5
                res = sorted(range(len(count_genre)), key = lambda sub: count_genre[sub])[-N:]
                res.sort()       

                all_user_genres.append(res)
                all_user_movies.append(copy.deepcopy(user_all_movies))
                all_user_liked.append(copy.deepcopy(user_all_liked))
                all_user_disliked.append(copy.deepcopy(user_all_disliked))
                all_user_released.append(copy.deepcopy(user_all_released))
                all_user_name.append(copy.deepcopy(user_all_name))
                all_user_days.append(copy.deepcopy(sum(user_all_days)/len(user_all_days)))
                all_user_months.append(copy.deepcopy(sum(user_all_months)/len(user_all_months)))
                all_user_years.append(copy.deepcopy(sum(user_all_years)/len(user_all_years)))
                all_user_labels.append([row["item_id"]])
                
                # print(user_all_movies)
                # print(user_all_liked)
                # print(row["item_id"])
                # print(list(set(user_all_genres)))

                #now the values are updated for the next iteration
                user_all_movies.append(row["item_id"])
                user_all_genres = (user_all_genres + row["covered_genres"])
                user_all_liked.append(row["item_id"])   
                user_all_released.append(row["movie_released"])
                user_all_name.append(row["movie_name"])
                user_all_days.append(int(row["date"][0]))
                user_all_months.append(int(row["date"][1]))
                user_all_years.append(int(row["date"][2]))
                counter = counter + 1             
    #print(all_user_labels)
    #multiply the data from the original dataset times the infered trainingsets per user
    user_gender_mul = np.repeat(user_data["gender"].tolist(), mul_vector)
    user_occupation_mul = np.repeat(user_data["occupation"].tolist(),mul_vector)
    user_age_mul = np.repeat(user_data["age"].tolist(),mul_vector)
    user_age_squared_mul = np.repeat(user_data["age_squared"].tolist(),mul_vector)
    user_age_power_3_mul = np.repeat(user_data["age_power_3"].tolist(),mul_vector)
    user_city_mul = np.repeat(user_data["city"].tolist(),mul_vector)
    user_state_mul = np.repeat(user_data["state"].tolist(),mul_vector)
    user_long_mul = np.repeat(user_data["longitude"].tolist(),mul_vector)
    user_lat_mul = np.repeat(user_data["latitude"].tolist(),mul_vector)
    user_timezone_mul = np.repeat(user_data["timezone"].tolist(),mul_vector)

    #create dataframe with the user id as inizialization
    input_data = pd.DataFrame(data=user_ids_mul,columns=["user_id"])
    input_data["occupation"] = user_occupation_mul
    input_data["gender"] = user_gender_mul
    input_data["age"] = user_age_mul
    input_data["age_squared"] = user_age_squared_mul
    input_data["age_power_3"] = user_age_power_3_mul
    input_data["city"] = user_city_mul
    input_data["state"] = user_state_mul
    input_data["longitude"] = user_long_mul
    input_data["latitude"] = user_lat_mul
    input_data["timezone"] = user_timezone_mul

    #print(input_data.head(20))

    #fill the gathered information into columns
    input_data["watched_movies"] = all_user_movies
    input_data["liked"] = all_user_liked
    input_data["disliked"] = all_user_disliked
    input_data["watched_genres"] = all_user_genres
    input_data["released"] = all_user_released
    input_data["movie_names"] = all_user_name
    input_data["days"] = all_user_days
    input_data["months"] = all_user_months
    input_data["years"] = all_user_years
    input_data["label"] = all_user_labels
    #add further info

    return input_data

input_data = gather_user_info()

#covert label list to int
input_data["label"] = input_data["label"].apply(lambda x: x[0])
print(input_data.head(20))

train_splits = []
val_splits = []
test_splits = []

for _, group_data in input_data.groupby("user_id"):
    #print(group_data,"||",_)
    rand_num = np.random.rand(len(group_data.index)) 
    selection = rand_num <=0.8 #if ran <= 0.85 True else False
    train_splits.append(group_data[selection])
    selection = (0.8<rand_num) & (rand_num<=0.9)
    val_splits.append(group_data[selection])
    selection = rand_num>0.9
    test_splits.append(group_data[selection])
#print(test_splits)

#shuffle entries
train_data = pd.concat(train_splits).sample(frac=1).reset_index(drop=True)
val_data = pd.concat(val_splits).sample(frac=1).reset_index(drop=True)
test_data = pd.concat(test_splits).sample(frac=1).reset_index(drop=True)

#print(f"Training Split Size: {len(train_data.index)}")
#print(f"Test Split Size:{len(test_data.index)}")
def format_data(data):
    data["longitude"] = data["longitude"].abs()
    data["timezone"] = data["timezone"].abs()
    data["days"] = data["days"].apply(lambda x: x/31)
    data["months"] = data["months"].apply(lambda x: x/12)
    data["years"] = data["years"].apply(lambda x: x/2000)

    return data

train_data = format_data(train_data)
val_data = format_data(val_data)
test_data = format_data(test_data)

train_data_file="candidate_train_data.pkl"
val_data_file="candidate_val_data.pkl"
test_data_file="candidate_test_data.pkl"

train_data.to_pickle(train_data_file)
val_data.to_pickle(val_data_file)
test_data.to_pickle(test_data_file)