import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
import time
import re
import seaborn as sns
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

CSV_HEADER = ["user_id","occupation","gender","age","age_squared","age_power_3","city","state","longitude","latitude","timezone","watched_movies","liked","disliked",
              "watched_genres","released","movie_names","item_id","movie_released","movie_name","covered_genres","days","months","years","in_watched","in_year",
              "in_genre","in_name","label"]
TARGET_FEATURE_NAME = "label"


#get data
train_data = pd.read_pickle("candidate_train_data.pkl")
val_data = pd.read_pickle("candidate_val_data.pkl")
test_data = pd.read_pickle("candidate_test_data.pkl")
item_data = pd.read_pickle("item_data.pkl")
print("data reading finished")
#print(train_data.head(10))
# print(test_data.head(10))

#train_data = train_data.drop([x for x in range(49000,49033)])

#coverting the string back to a list and then to a np.array for the tensor

def find_words(data):
    data["movie_words"] = data["movie_names"].apply(lambda x: " ".join(x))
    data["movie_words"] = data["movie_words"].apply(lambda x: re.findall(r"[\w'.]+",x))

find_words(train_data)
find_words(val_data)
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
    for name in data["movie_name"]:
        if name not in all_movie_list:
            all_movie_list.append(name)
            all_words_list.extend(re.findall(r"[\w'.]+", name))

get_all_movie_names(item_data)
print("length movie_list:", len(all_movie_list))
print("length word_list:", len(all_words_list))
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
get_city_and_state(val_data)
get_city_and_state(test_data)
print("length city:",len(city_list))
print("length state:",len(state_list))

del train_data
del val_data
del test_data

TARGET_FEATURE_LABELS = [0,1]
NUMERIC_FEATURE_NAMES = [#"user_id",
                         "age",
                         "age_squared",
                         "age_power_3",
                         "timezone",
                         "longitude",
                         "latitude",
                         "movie_released",
                         "days",
                         "months",
                         "years",
                         "in_watched",
                         "in_year",
                         "in_genre",
                         "in_name",
                        ]
NUM_LIST_FEATURE_NAMES = ["watched_movies",
                          "liked",
                          "disliked",
                          "watched_genres",
                          "released",
                          "covered_genres",
                          "item_id",
                         ]

CAT_LIST_FEATURE_WITH_VOCABULARY = {
    "movie_names": all_movie_list,
    #"movie_words": all_words_list,
}
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    "gender": ["M","F"],
    "occupation": ["administrator","artist","doctor","educator","engineer","entertainment","executive","healthcare","homemaker","lawyer","librarian","marketing",
                   "none","other","programmer","retired","salesman","scientist","student","technician","writer"],
    "city": city_list,
    "state": state_list,
    "movie_name": all_movie_list,
}

CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())
CAT_LIST_FEATURE_NAMES = list(CAT_LIST_FEATURE_WITH_VOCABULARY.keys())
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES + NUM_LIST_FEATURE_NAMES + CAT_LIST_FEATURE_NAMES

#+1 for the embeddings
NUM_MOVIES = 1682+1

learning_rate = 0.0005
#learning_rate = 0.0005
dropout_rate = 0.05
num_batch_size = 3000
num_epochs = 20
embedding_tall = 110
embedding_middle = 16
embedding_small = 8

frac_num = 2
k = 11
#init_learning_rate = 0.006

hidden_units_deep = [256,128,64]
hidden_units_cross = [1,2,3]
hidden_units_dlrm_top = [2048,1024]
hidden_units_dlrm_bottom = [2048,1024]

pos = 54433
neg = 5388867
total = 5443300

initial_bias = np.log([pos/neg])

weights_0 = (1/neg)*(total/2.0)
weights_1 = (1/pos)*(total/2.0)
class_weights = {0 : weights_0, 1 : weights_1}


def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERIC_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.float32
            )
        elif feature_name in NUM_LIST_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(None, ), dtype=tf.float32
            )
        elif feature_name in CAT_LIST_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(None, ), dtype=tf.string
            )
        elif feature_name in CATEGORICAL_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(),dtype=tf.string
            )
    return inputs

def encode_inputs(inputs, style, use_embedding=False):
    encoded_features = []
    #print(inputs)
    for feature_name in inputs:
        #print(feature_name)
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Create a lookup to convert string values to an integer indices.
            lookup = StringLookup(
                output_mode="int" if use_embedding else "binary",
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=0,
            )
            if use_embedding:
                # Covert the string to integer indices
                encoded_feature = lookup(inputs[feature_name])
                #print(encoded_feature)
                embedding = layers.Embedding(
                    input_dim=len(vocabulary), output_dim=embedding_small, input_length=1
                )
                encoded_feature = embedding(encoded_feature)
                #print(encoded_feature)
                if style == "mean":
                    encoded_feature = tf.expand_dims(encoded_feature, -1)
                    encoded_feature = tf.math.reduce_mean(encoded_feature,axis=1,name="mean")
                    #print(encoded_feature)

                if style == "sum":
                    encoded_feature = tf.expand_dims(encoded_feature, -1)
                    encoded_feature = tf.math.reduce_sum(encoded_feature,axis=1,name="sum")

            else:
                # use lookup as is
                encoded_feature = lookup(tf.expand_dims(inputs[feature_name], -1))

        elif feature_name in NUM_LIST_FEATURE_NAMES:
            #list already padded
            if feature_name == "watched_genres" or feature_name == "covered_genres":
                embedding = layers.Embedding(
                    input_dim=19, output_dim=embedding_tall, mask_zero=True
                )
            if feature_name == "released":
                embedding = layers.Embedding(
                    input_dim=2000, output_dim=embedding_tall, mask_zero=True
                )
            else:
                embedding = layers.Embedding(
                    input_dim=NUM_MOVIES, output_dim=embedding_tall, mask_zero=True
                )

            if style == "mean":
                encoded_feature = tf.math.reduce_mean(embedding(inputs[feature_name]),axis=1,name="mean")
                #print(encoded_feature)
                #encoded_feature = tf.expand_dims(encoded_feature, -1)

            if style == "sum":
                encoded_feature = tf.math.reduce_sum(embedding(inputs[feature_name]),axis=1,name="sum")
                #encoded_feature = tf.expand_dims(encoded_feature, -1)

        # elif feature_name in TIME_LIST:
        #     if feature_name == "days":
        #         embedding = layers.Embedding(
        #             input_dim=32, output_dim = 2, mask_zero=True
        #         )
        #     elif feature_name == "months":
        #         embedding = layers.Embedding(
        #             input_dim=13, output_dim = 2, mask_zero=True
        #         )
        #     elif feature_name == "years":
        #         embedding = layers.Embedding(
        #             input_dim=2000, output_dim = 4, mask_zero=True
        #         )
        #     if style == "mean":
        #         encoded_feature = tf.math.reduce_mean(embedding(inputs[feature_name]),axis=1,name="mean")
        #         #print(encoded_feature)
        #         #encoded_feature = tf.expand_dims(encoded_feature, -1)

        #     if style == "sum":
        #         encoded_feature = tf.math.reduce_sum(embedding(inputs[feature_name]),axis=1,name="sum")
        #         #encoded_feature = tf.expand_dims(encoded_feature, -1)

        elif feature_name in CAT_LIST_FEATURE_NAMES:

            vocabulary = CAT_LIST_FEATURE_WITH_VOCABULARY[feature_name]
            lookup = StringLookup(
                output_mode="int" if use_embedding else "binary",
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=0,
                #oov_token= 0,
            )
            encoded_feature = lookup(inputs[feature_name])
            encoded_feature = tf.where(tf.equal(encoded_feature,-1),0,tf.cast(encoded_feature,dtype=tf.int32))
            if feature_name == "movie_names":
                embedding = layers.Embedding(
                    input_dim=NUM_MOVIES, output_dim=embedding_middle, mask_zero=True
                )
            else:
                embedding = layers.Embedding(
                    input_dim=len(vocabulary)+1, output_dim=embedding_middle, mask_zero=True
                )
            if style == "mean":
                encoded_feature = tf.math.reduce_mean(embedding(encoded_feature),axis=1,name="mean")

            if style == "sum":
                encoded_feature = tf.math.reduce_sum(embedding(encoded_feature),axis=1,name="sum")

        else:
            # Use the numerical features as-is.
            encoded_feature = tf.expand_dims(tf.cast(inputs[feature_name], tf.float32), -1)
        #print(encoded_feature)
        encoded_features.append(encoded_feature)
        #print(encoded_features)

    all_features = layers.concatenate(encoded_features)
    #print(all_features)
    return all_features


def compile_model(model):

    METRICS = [
    #   keras.metrics.TruePositives(name='tp'),
    #   keras.metrics.FalsePositives(name='fp'),
    #   keras.metrics.TrueNegatives(name='tn'),
    #   keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=METRICS,
    )


def fit_model(model,train_generator,val_generator,name):

    class TerminateOnBaseline(keras.callbacks.Callback):
        """Callback that terminates training when either acc or val_acc reaches a specified baseline
        """
        def __init__(self, monitor='val_auc', baseline=0.705):
            super(TerminateOnBaseline, self).__init__()
            self.monitor = monitor
            self.baseline = baseline

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            acc = logs.get(self.monitor)
            if acc is not None:
                if acc >= self.baseline:
                    print('Epoch %d: Reached baseline, terminating training' % (epoch))
                    self.model.stop_training = True

    history = model.fit(train_generator,validation_data=val_generator,epochs=num_epochs,class_weight=class_weights,
                        callbacks=TerminateOnBaseline("val_auc",0.705),shuffle=True)

    def plot_metrics(history):
        metrics = ['loss', 'prc', 'precision', 'recall']
        for n, metric in enumerate(metrics):
            name = metric.replace("_"," ").capitalize()
            plt.subplot(2,2,n+1)
            plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
            plt.plot(history.epoch, history.history['val_'+metric],
                    color=colors[0], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc':
                plt.ylim([0.8,1])
            else:
                plt.ylim([0,1])

            plt.legend()

    plot_metrics(history)

    model.save(name)


def evaluate_model(model,data):

    test_input = {"age": data["age"].values,
            "user_id": data["user_id"].values/942,
            "age_squared":data["age_squared"].values,
            "age_power_3":data["age_power_3"].values,
            "longitude": data["longitude"].values,
            "latitude": data["latitude"].values,
            "timezone": data["timezone"].values,
            "days": data["days"].values,
            "months": data["months"].values,
            "years": data["years"].values,
            "in_watched": data["in_watched"].values,
            "in_year": data["in_year"].values,
            "in_genre": data["in_genre"].values,
            "in_name": data["in_name"].values,
            "city": data["city"].values,
            "state": data["state"].values,
            "gender": data["gender"].values,
            "occupation": data["occupation"].values,
            "watched_movies": tf.keras.preprocessing.sequence.pad_sequences(data["watched_movies"]),
            "liked": tf.keras.preprocessing.sequence.pad_sequences(data["liked"]),
            "disliked": tf.keras.preprocessing.sequence.pad_sequences(data["disliked"]),
            "watched_genres": tf.keras.preprocessing.sequence.pad_sequences(data["watched_genres"]),
            "released": tf.keras.preprocessing.sequence.pad_sequences(data["released"]),
            "movie_names": tf.keras.preprocessing.sequence.pad_sequences(data["movie_names"],dtype=object,value="yyy"),
            #"movie_words": tf.keras.preprocessing.sequence.pad_sequences(data["movie_words"],dtype=object,value="yyy"),
            "item_id": tf.keras.preprocessing.sequence.pad_sequences(data["item_id"]),
            "movie_released": data["movie_released"].values,
            "movie_name": data["movie_name"].values,
            "covered_genres": tf.keras.preprocessing.sequence.pad_sequences(data["covered_genres"]),
            }
    test_label = data["label"].values

    results = model.evaluate(test_input,test_label,batch_size=num_batch_size)

    for name, value in zip(model.metrics_names, results):
        print(name, ': ',value)
    print()
    start = time.time()
    predictions = model.predict(test_input,batch_size= num_batch_size)

    user_predictions = []
    user_labels = []
    predictions = np.reshape(predictions,(1,-1))[0]
    #print(predictions)
    for i in range(1,int(len(predictions)/100)):
        user_predictions.append(predictions[(i-1)*100:i*100])
        user_labels.append(test_label[(i-1)*100:i*100])
    #print(user_predictions)
    print("len user labels: ",len(user_labels[0]))
    print("len user prediction:", len(user_predictions[0]))

    values,indices = tf.math.top_k(user_predictions,k=k)
    top_k = list(indices.numpy())
    end = time.time()
    print(len(user_predictions))
    print("prediction time:",end-start)

    def plot_cm(labels, predictions, p=0.5):
        cm = confusion_matrix(labels, predictions > p)
        plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion matrix @{:.2f}'.format(p))
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()
        print('Movie rightfully not recommended (True Negatives): ', cm[0][0])
        print('Movie wrongfully recommended (False Positives): ', cm[0][1])
        print('Movie wrongfully not recommended (False Negatives): ', cm[1][0])
        print('Movie rightfully recommended (True Positives): ', cm[1][1])
        print('Ideal number of movie recommendations: ', np.sum(cm[1]))
    plot_cm(test_label,predictions)

    print("learning rate:", learning_rate)
    print("batch size:", num_batch_size)
    counterpos = 0
    counterneg = 0
    print("length top_k:",len(top_k[0]))
    for i, l in enumerate(top_k):
        x = 0
        for idx in l:
            if user_labels[i][idx] == 1:
                x += 1
        if x == 1: counterpos +=1
        else: counterneg +=1

    print("top_k accuracy: ", counterpos/(counterpos+counterneg))



def create_dlrm_inputs():
    num_inputs = {}
    cat_inputs = {}
    for feature_name in NUMERIC_FEATURE_NAMES:
        num_inputs[feature_name] = layers.Input(name=feature_name, shape=(), dtype=tf.float32)
    for feature_name in CATEGORICAL_FEATURE_NAMES:
        cat_inputs[feature_name] = layers.Input(name=feature_name, shape = (), dtype = tf.string)
    for feature_name in NUM_LIST_FEATURE_NAMES:
        cat_inputs[feature_name] = layers.Input(name=feature_name, shape = (None,), dtype = tf.float32)
    for feature_name in CAT_LIST_FEATURE_NAMES:
        cat_inputs[feature_name] = layers.Input(name=feature_name, shape = (None,), dtype = tf.string)
    return num_inputs, cat_inputs

def create_wide_and_deep_model(output_bias=None):

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    inputs = create_model_inputs()
    wide = encode_inputs(inputs, style="mean", use_embedding=False)
    wide = layers.BatchNormalization()(wide)

    deep = encode_inputs(inputs, style="mean", use_embedding=True)
    deep = layers.BatchNormalization()(deep)
    for units in hidden_units_deep:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(dropout_rate)(deep)

    merged = layers.concatenate([wide,deep])
    outputs = layers.Dense(units=1, activation="sigmoid", bias_initializer=output_bias)(merged)
    #outputs = layers.Dense(units=1, activation="sigmoid")(merged)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_deep_and_cross_model(output_bias=None):

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    inputs = create_model_inputs()
    x0 = encode_inputs(inputs, style="mean", use_embedding=True)

    cross = x0
    for _ in hidden_units_cross:
        units = cross.shape[-1]
        x = layers.Dense(units)(cross)
        cross = x0 * x + cross
    cross = layers.BatchNormalization()(cross)

    deep = x0
    for units in hidden_units_deep:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(dropout_rate)(deep)

    merged = layers.concatenate([cross, deep])
    outputs = layers.Dense(units=1, activation="sigmoid", bias_initializer=output_bias)(merged)
    #outputs = layers.Dense(units=1, activation="sigmoid")(merged)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_deep_and_cross_model_stacked(output_bias=None):

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)


    inputs = create_model_inputs()
    x0 = encode_inputs(inputs, style="mean", use_embedding=True)

    cross = x0
    for _ in hidden_units_cross:
        units = cross.shape[-1]
        x = layers.Dense(units)(cross)
        cross = x0 * x + cross
    cross = layers.BatchNormalization()(cross)
    cross = layers.ReLU()(cross)

    deep = cross
    for units in hidden_units_deep:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(dropout_rate)(deep)
    stack = deep
    outputs = layers.Dense(units=1, activation="sigmoid", bias_initializer=output_bias)(stack)
    #outputs = layers.Dense(units=1, activation="sigmoid")(stack)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

def create_dlrm_model(output_bias=None):

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    num_inputs, cat_inputs = create_dlrm_inputs()
    embedding_outputs_list = []

    for key in CATEGORICAL_FEATURE_NAMES:
        embedding_outputs_list.append(encode_inputs({key: cat_inputs[key]},"mean",True))
    num_encoded_inputs = encode_inputs(num_inputs,"mean",True)

    interaction_inputs = []

    for deep in embedding_outputs_list:
        for units in hidden_units_dlrm_bottom:
            deep = layers.Dense(units)(deep)
            deep = layers.BatchNormalization()(deep)
            deep = layers.ReLU()(deep)
            deep = layers.Dropout(dropout_rate)(deep)
        interaction_inputs.append(deep)

    deep = num_encoded_inputs
    for units in hidden_units_dlrm_bottom:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(dropout_rate)(deep)
    interaction_inputs.append(deep)

    def matrix_mul_func(inputs):
        matrix_mul = []

        for idx1,one in enumerate(inputs):
            one = tf.expand_dims(one,1)
            for idx2,two in enumerate(inputs):
                if idx1 <= idx2: continue
                two = tf.expand_dims(two,2)
                matrix_mul.append(tf.reshape(one @ two ,[-1,1]))
        matrix_mul = layers.concatenate(matrix_mul)

        return matrix_mul

    matrix_mul = matrix_mul_func(interaction_inputs)
    matrix_mul = layers.BatchNormalization()(matrix_mul)
    matrix_mul = layers.concatenate([matrix_mul,deep])

    deep = matrix_mul

    for units in hidden_units_dlrm_top:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(dropout_rate)(deep)

    #outputs = layers.Dense(units=1, activation="sigmoid")(deep)
    outputs = layers.Dense(units=1, activation="sigmoid", bias_initializer=output_bias)(deep)
    model = keras.Model(inputs={**num_inputs, **cat_inputs}, outputs=outputs)
    return model

def create_dlrm_parallel_model(output_bias=None):

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    num_inputs, cat_inputs = create_dlrm_inputs()
    embedding_outputs_list = []

    for key in CATEGORICAL_FEATURE_NAMES:
        embedding_outputs_list.append(encode_inputs({key: cat_inputs[key]},"mean",True))
    num_encoded_inputs = encode_inputs(num_inputs,"mean",True)

    interaction_inputs = []

    for deep in embedding_outputs_list:
        for units in hidden_units_dlrm_bottom:
            deep = layers.Dense(units)(deep)
            deep = layers.BatchNormalization()(deep)
            deep = layers.ReLU()(deep)
            deep = layers.Dropout(dropout_rate)(deep)
        interaction_inputs.append(deep)

    deep = num_encoded_inputs
    for units in hidden_units_dlrm_bottom:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(dropout_rate)(deep)
    interaction_inputs.append(deep)

    def matrix_mul_func(inputs):
        matrix_mul = []

        for idx1,one in enumerate(inputs):
            one = tf.expand_dims(one,1)
            for idx2,two in enumerate(inputs):
                if idx1 <= idx2: continue
                two = tf.expand_dims(two,2)
                matrix_mul.append(tf.reshape(one @ two ,[-1,1]))
        matrix_mul = layers.concatenate(matrix_mul)

        return matrix_mul

    matrix_mul = matrix_mul_func(interaction_inputs)
    matrix_mul = layers.BatchNormalization()(matrix_mul)
    matrix_mul = layers.concatenate([matrix_mul,deep])

    deep = layers.concatenate(interaction_inputs)

    for units in hidden_units_dlrm_top:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(dropout_rate)(deep)

    merged = layers.concatenate([matrix_mul,deep])
    outputs = layers.Dense(units=1, activation="sigmoid",bias_initializer=output_bias)(merged)
    model = keras.Model(inputs={**num_inputs, **cat_inputs}, outputs=outputs)
    return model

def create_candidate_generator(output_bias=None):

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    inputs = create_model_inputs()
    #print(inputs)
    encoded_inputs = encode_inputs(inputs,"mean",True)
    #print(encoded_inputs)
    deep = encoded_inputs
    for units in hidden_units_deep:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(dropout_rate)(deep)

    outputs = layers.Dense(units=1, activation="sigmoid", bias_initializer=output_bias)(deep)
    #outputs = layers.Dense(units=1, activation="sigmoid")(deep)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model






#wide_and_deep_model = create_wide_and_deep_model(initial_bias)
#deep_and_cross_model = create_deep_and_cross_model(initial_bias) #"DCN_parallel"
# deep_and_cross_stacked = create_deep_and_cross_model_stacked(initial_bias) #"DCN_stacked"
dlrm_model = create_dlrm_model(initial_bias) #"DLRM_stacked"
#dlrm_parallel_model = create_dlrm_parallel_model(initial_bias) #"DLRM_parallel"
# dlrm_modified_model = create_modified_dlrm_model(initial_bias)
# candidate_generator = create_candidate_generator(initial_bias)

def run_experiment(model,name):

    compile_model(model)
    # def loss_function(y_true,y_pred):
    #     print(y_true)
    #     output_layer = model.layers[-1]
    #     loss =  tf.nn.sampled_softmax_loss(
    #         weights = tf.transpose(output_layer.weights[0]),
    #         biases = output_layer.weights[1],
    #         labels = y_true,
    #         inputs = output_layer.input,
    #         num_sampled=1400,
    #         NUM_MOVIES= 1682,
    #         num_true= 1,
    #     )
    #     return loss

    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     init_learning_rate,
    #     decay_steps=30,
    #     decay_rate=0.95,
    #     staircase=True
    # )

    # early_stopping = tf.keras.callbacks.EarlyStopping(
    # monitor='val_prc',
    # verbose=1,
    # patience=10,
    # mode='max',
    # restore_best_weights=True)
    class Data_Generator(keras.utils.Sequence):

        def __init__(self,data,batch_size,shuffle=True):
            self.data = data
            self.shuffle = shuffle
            self.batch_size = batch_size
            self.on_epoch_end()

        def __len__(self):
            return int((len(self.data["user_id"].index)/(frac_num*self.batch_size)))

        def __getitem__(self, idx):

            input_data = self.data[idx*self.batch_size:(idx+1)*self.batch_size]
            train_input ={"age":input_data["age"].values,
                        "user_id": input_data["user_id"].values/942,
                        "age_squared":input_data["age_squared"].values,
                        "age_power_3":input_data["age_power_3"].values,
                        "longitude": input_data["longitude"].values,
                        "latitude": input_data["latitude"].values,
                        "timezone": input_data["timezone"].values,
                        "days": input_data["days"].values,
                        "months": input_data["months"].values,
                        "years": input_data["years"].values,
                        "in_watched": input_data["in_watched"].values,
                        "in_year": input_data["in_year"].values,
                        "in_genre": input_data["in_genre"].values,
                        "in_name": input_data["in_name"].values,
                        "city": input_data["city"].values,
                        "state": input_data["state"].values,
                        "gender": input_data["gender"].values,
                        "occupation": input_data["occupation"].values,
                        "watched_movies": tf.keras.preprocessing.sequence.pad_sequences(input_data["watched_movies"]),
                        "liked": tf.keras.preprocessing.sequence.pad_sequences(input_data["liked"]),
                        "disliked": tf.keras.preprocessing.sequence.pad_sequences(input_data["disliked"]),
                        "watched_genres": tf.keras.preprocessing.sequence.pad_sequences(input_data["watched_genres"]),
                        "released": tf.keras.preprocessing.sequence.pad_sequences(input_data["released"]),
                        "movie_names": tf.keras.preprocessing.sequence.pad_sequences(input_data["movie_names"],dtype=object,value="yyy"),
                        #"movie_words": tf.keras.preprocessing.sequence.pad_sequences(input_data["movie_words"],dtype=object,value="yyy"),
                        "item_id": tf.keras.preprocessing.sequence.pad_sequences(input_data["item_id"]),
                        "movie_released": input_data["movie_released"].values,
                        "movie_name": input_data["movie_name"].values,
                        "covered_genres": tf.keras.preprocessing.sequence.pad_sequences(input_data["covered_genres"]),
                    }
            train_label = input_data["label"].values
            return train_input,train_label

        def on_epoch_end(self):
            if self.shuffle == True:
                self.data = self.data.sample(frac=1).reset_index(drop=True)

    train_data = pd.read_pickle("ranking_train_data.pkl")
    train_data_generator = Data_Generator(train_data,num_batch_size)
    del train_data
    val_data = pd.read_pickle("ranking_val_data.pkl")
    val_data_generator = Data_Generator(val_data,num_batch_size)
    del val_data
    #fit_model(model,train_data_generator,val_data_generator,name)


    test_data = pd.read_pickle("ranking_test_data.pkl")
    evaluate_model(model,test_data)

run_experiment(dlrm_model,"DLRM_stacked")
