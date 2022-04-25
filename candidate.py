import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from ast import literal_eval
import re
from tensorflow.python.keras.layers.advanced_activations import ReLU
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

CSV_HEADER = ["user_id","occupation","gender","age","age_squared","age_power_3","city","state","longitude","latitude","timezone","watched_movies","liked","disliked",
              "watched_genres","released","movie_names","days","months","years","label"]
TARGET_FEATURE_NAME = "label"

#get data
train_data = pd.read_pickle("candidate_train_data.pkl")
val_data = pd.read_pickle("candidate_val_data.pkl")
test_data = pd.read_pickle("candidate_test_data.pkl")

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
    for movie_list in data["movie_names"]:
        for name in movie_list:
            if not(name in all_movie_list):
                all_movie_list.append(name)
                all_words_list.extend(re.findall(r"[\w'.]+", name))

get_all_movie_names(train_data)
get_all_movie_names(val_data)
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
get_city_and_state(val_data)
get_city_and_state(test_data)

#print(all_movie_list)
#print("length of all_movie_list: ", len(all_movie_list))
#number of movies
TARGET_FEATURE_LABELS = [x for x in range(1682)]
NUMERIC_FEATURE_NAMES = ["age",
                        "age_squared",
                        "age_power_3",
                        "timezone",
                        "longitude",
                        "latitude",
                        "days",
                        "months",
                        "years",
                        ]
NUM_LIST_FEATURE_NAMES = ["watched_movies", "liked",
                         "disliked",
                         "watched_genres",
                         "released"
                         ]
CAT_LIST_FEATURE_WITH_VOCABULARY = {
    "movie_names": all_movie_list,
    "movie_words": all_words_list,
}
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    "gender": ["M","F"],
    "occupation": ["administrator","artist","doctor","educator","engineer","entertainment","executive","healthcare","homemaker","lawyer","librarian","marketing",
                  "none","other","programmer","retired","salesman","scientist","student","technician","writer"], 
    "city": city_list,
    "state": state_list,
}


CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())
CAT_LIST_FEATURE_NAMES = list(CAT_LIST_FEATURE_WITH_VOCABULARY.keys())
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES + NUM_LIST_FEATURE_NAMES + CAT_LIST_FEATURE_NAMES #+ TIME_LIST

NUM_CLASSES = len(TARGET_FEATURE_LABELS)+1

#learning_rate = 0.0015
learning_rate = 0.00004
dropout_rate = 0.05
num_batch_size = 8000
#batch_size = 265
num_epochs = 800
embedding_tall = 128
embedding_medium = 16
embedding_small = 8

#init_learning_rate = 0.006

hidden_units_deep = [256,128,64]
hidden_units_cross = [1,2,3]
hidden_units_dlrm_top = [1024,512,256]
hidden_units_dlrm_bottom = [1024,512,256]

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
        # elif feature_name in TIME_LIST:
        #     inputs[feature_name] = layers.Input(
        #         name=feature_name, shape=(None, ), dtype=tf.float32
        #     )    
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
            if feature_name == "watched_genres":
                embedding = layers.Embedding(
                    input_dim=19, output_dim=embedding_tall, mask_zero=True
                )
            if feature_name == "released":
                embedding = layers.Embedding(
                    input_dim=2000, output_dim=embedding_tall, mask_zero=True
                )
            else:
                embedding = layers.Embedding(
                    input_dim=NUM_CLASSES, output_dim=embedding_tall, mask_zero=True
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
                    input_dim=NUM_CLASSES, output_dim=embedding_medium, mask_zero=True
                )
            else: 
                embedding = layers.Embedding(
                    input_dim=len(vocabulary)+1, output_dim=embedding_medium, mask_zero=True
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

def run_experiment(model):

    # def loss_function(y_true,y_pred):
    #     print(y_true)
    #     output_layer = model.layers[-1]
    #     loss =  tf.nn.sampled_softmax_loss(
    #         weights = tf.transpose(output_layer.weights[0]),
    #         biases = output_layer.weights[1],
    #         labels = y_true, 
    #         inputs = output_layer.input,
    #         num_sampled=1400,
    #         num_classes= 1682,
    #         num_true= 1, 
    #     )
    #     return loss 
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     init_learning_rate,
    #     decay_steps=30,
    #     decay_rate=0.95,
    #     staircase=True
    # )

    class TerminateOnBaseline(keras.callbacks.Callback):
        """Callback that terminates training when either acc or val_acc reaches a specified baseline
        """
        def __init__(self, monitor='val_sparse_categorical_accuracy', baseline=0.022):
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

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        experimental_run_tf_function = False,
    )
    # print(np.shape(train_data["age"].values))
    #print(np.shape(train_data["zip_code"].values))
    # print(np.shape(train_data["gender"].values))
    # print(np.shape(train_data["occupation"].values))
    # print(np.shape(tf.keras.preprocessing.sequence.pad_sequences(train_data["watched_movies"])))
    # print(np.shape(tf.keras.preprocessing.sequence.pad_sequences(train_data["liked"])))
    # print(np.shape(tf.keras.preprocessing.sequence.pad_sequences(train_data["disliked"])))
    # print(np.shape(tf.keras.preprocessing.sequence.pad_sequences(train_data["watched_genres"])))
    

    input = {"age": train_data["age"].values,
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
    
    target = train_data["label"].values

    val_input = {"age": val_data["age"].values,
            "age_squared":val_data["age_squared"].values,
            "age_power_3":val_data["age_power_3"].values,
            "longitude": val_data["longitude"].values,
            "latitude": val_data["latitude"].values,
            "timezone": val_data["timezone"].values,
            "city": val_data["city"].values,
            "state": val_data["state"].values,
            "gender": val_data["gender"].values,
            "occupation": val_data["occupation"].values,
            "watched_movies": tf.keras.preprocessing.sequence.pad_sequences(val_data["watched_movies"]),
            "liked": tf.keras.preprocessing.sequence.pad_sequences(val_data["liked"]),
            "disliked": tf.keras.preprocessing.sequence.pad_sequences(val_data["disliked"]),
            "watched_genres": tf.keras.preprocessing.sequence.pad_sequences(val_data["watched_genres"]),
            "released": tf.keras.preprocessing.sequence.pad_sequences(val_data["released"]),
            "days": val_data["days"].values,
            "months": val_data["months"].values,
            "years": val_data["years"].values,
            "movie_names": tf.keras.preprocessing.sequence.pad_sequences(val_data["movie_names"],dtype=object,value="yyy"),    
            "movie_words": tf.keras.preprocessing.sequence.pad_sequences(val_data["movie_words"],dtype=object,value="yyy"),      
            }
    
    val_target = val_data["label"].values
    print("Start training the model...")
    history = model.fit(x=input,y=target,epochs=num_epochs,validation_data=(val_input,val_target),batch_size=num_batch_size,
                        callbacks=TerminateOnBaseline("val_sparse_categorical_accuracy",0.022),
                        shuffle=True)
    print("Model training finished")

    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.plot(history.history["sparse_categorical_accuracy"],label="train")
    ax1.plot(history.history["val_sparse_categorical_accuracy"],label="validation")
    ax1.legend(loc="upper left")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("sparse categortical accuracy")
    ax1.set_title("model accuracy")


    ax2.plot(history.history["loss"],label="train")
    ax2.plot(history.history["val_loss"],label="validation")
    ax2.legend(loc="upper right")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.set_title("model loss")
    #fig.set_legend(["training","validation"])
    plt.show()
    
    input = {"age": test_data["age"].values,
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

    target = test_data["label"].values

    _, accuracy = model.evaluate(x=input,y=target, batch_size = 20, verbose = 2)
    print(f"Test accuracy: {round(accuracy*100, 2)}%")
    print("learning rate:", learning_rate)
    print("batch size:", num_batch_size)

    model.save("model_candidate")


def create_dlrm_inputs():
    num_inputs = {}
    cat_inputs = {}
    for feature_name in NUMERIC_FEATURE_NAMES:
        num_inputs[feature_name] = layers.Input(name=feature_name, shape=(), dtype=tf.float32)
    for feature_name in CATEGORICAL_FEATURE_NAMES:
        cat_inputs[feature_name] = layers.Input(name=feature_name, shape = (), dtype = tf.string)
    return num_inputs, cat_inputs

def create_wide_and_deep_model():
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
    outputs = layers.Dense(units=NUM_CLASSES, activation="softmax")(merged)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_deep_and_cross_model():

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
    outputs = layers.Dense(units=NUM_CLASSES, activation="softmax")(merged)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_deep_and_cross_model_stacked():

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
    outputs = layers.Dense(units=NUM_CLASSES, activation="softmax")(stack)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def create_dlrm_model():
    num_inputs, cat_inputs = create_dlrm_inputs()
    embedding_outputs_list = []

    for key in CATEGORICAL_FEATURE_NAMES:
        embedding_outputs_list.append(encode_inputs({key: cat_inputs[key]}, True))
    num_encoded_inputs = encode_inputs(num_inputs)

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

    def dot_product_func(inputs):
        dot_product = inputs[0]
        for idx in range(1,len(inputs)):
            dot_product*= inputs[idx]
            dot_product = layers.BatchNormalization()(dot_product)
        return dot_product

    def matrix_mul_func(inputs):
        matrix_mul = []

        # print(inputs)
        # for idx, inp in enumerate(inputs):
        #     print(inp)
        #     print(tf.expand_dims(inp,1))
        #     print(tf.expand_dims(inp,2))
        #     print(tf.reshape(tf.expand_dims(inp,1)@tf.expand_dims(inp,2),[-1,1]))

        for one in inputs:
            one = tf.expand_dims(one,1)
            for two in inputs:
                two = tf.expand_dims(two,2)
                matrix_mul.append(tf.reshape(one @ two ,[-1,1]))
        matrix_mul = layers.concatenate(matrix_mul)

        return matrix_mul

    #dot_product = dot_product_func(interaction_inputs)
    #input_top = layers.concatenate([deep, dot_product])
    #deep = input_top

    matrix_mul = matrix_mul_func(interaction_inputs)
    deep = matrix_mul

    for units in hidden_units_dlrm_top:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(dropout_rate)(deep)

    outputs = layers.Dense(units=NUM_CLASSES, activation="softmax")(deep)
    model = keras.Model(inputs={**num_inputs, **cat_inputs}, outputs=outputs)
    return model

def create_dlrm_parallel_model():
    num_inputs, cat_inputs = create_dlrm_inputs()
    embedding_outputs_list = []

    for key in CATEGORICAL_FEATURE_NAMES:
        embedding_outputs_list.append(encode_inputs({key: cat_inputs[key]}, True))
    num_encoded_inputs = encode_inputs(num_inputs)
    cat_embedded_outputs = layers.concatenate(embedding_outputs_list)

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

        # print(inputs)
        # for idx, inp in enumerate(inputs):
        #     print(inp)
        #     print(tf.expand_dims(inp,1))
        #     print(tf.expand_dims(inp,2))
        #     print(tf.reshape(tf.expand_dims(inp,1)@tf.expand_dims(inp,2),[-1,1]))

        for one in inputs:
            one = tf.expand_dims(one,1)
            for two in inputs:
                two = tf.expand_dims(two,2)
                matrix_mul.append(tf.reshape(one @ two ,[-1,1]))
        matrix_mul = layers.concatenate(matrix_mul)

        return matrix_mul

    #dot_product = dot_product_func(interaction_inputs)
    #input_top = layers.concatenate([deep, dot_product])
    #deep = input_top

    matrix_mul = matrix_mul_func(interaction_inputs)

    inputs = layers.concatenate([num_encoded_inputs,cat_embedded_outputs])
    deep = inputs

    for units in hidden_units_dlrm_top:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(dropout_rate)(deep)

    merged = layers.concatenate([matrix_mul,deep])
    outputs = layers.Dense(units=NUM_CLASSES, activation="softmax")(merged)
    model = keras.Model(inputs={**num_inputs, **cat_inputs}, outputs=outputs)
    return model

def create_modified_dlrm_model():
    num_inputs, cat_inputs = create_dlrm_inputs()
    interaction_inputs = []

    for key in CATEGORICAL_FEATURE_NAMES:
        interaction_inputs.append(encode_inputs({key: cat_inputs[key]}, True))
    num_encoded_inputs = encode_inputs(num_inputs)

    hidden_units_dlrm_bottom = [32,16]
    deep = num_encoded_inputs
    for units in hidden_units_dlrm_bottom:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(dropout_rate)(deep)
    interaction_inputs.append(deep)

    # dot_product = [1]*max_embedding_length
    # for input in interaction_inputs:
    #     dot_product = np.dot(input, dot_product)

    def dot_product_func(inputs):
        dot_product = inputs[0]
        for idx in range(1,len(inputs)):
            dot_product*= inputs[idx]
        return dot_product
    dot_product = dot_product_func(interaction_inputs)

    print(dot_product)
    print(deep)

    input_top = deep
    for input in interaction_inputs:
        input_top = layers.concatenate([input_top, input])
    deep = input_top
    for units in hidden_units_dlrm_top:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(dropout_rate)(deep)

    merged = layers.concatenate([deep, dot_product])
    outputs = layers.Dense(units=NUM_CLASSES, activation="softmax")(merged)
    model = keras.Model(inputs={**num_inputs, **cat_inputs}, outputs=outputs)
    return model

def create_candidate_generator():
    
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
    #deep = layers.Dense(units=NUM_CLASSES, activation="linear")(deep)
    outputs = layers.Dense(units=NUM_CLASSES, activation="softmax")(deep)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


#wide_and_deep_model = create_wide_and_deep_model()
#wide_and_cross_model = create_deep_and_cross_model()
#wide_and_cross_stacked = create_deep_and_cross_model_stacked()
#dlrm_model = create_dlrm_model()
#dlrm_parallel_model = create_dlrm_parallel_model()
candidate_generator = create_candidate_generator()

run_experiment(candidate_generator)

