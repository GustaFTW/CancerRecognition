# import Deep learning Libraries
import tensorflow as tf
import sys
import numpy as np
sys.path.append("C:\\Projects\\Project1\\CancerRecognition\\Data\\")
sys.path.append("C:\\Projects\\Project1\\CancerRecognition\\Helper_Functions\\")
from helper_functions import MyCallback
from data_prep import train_gen, valid_gen, test_gen, test_df
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# Create Model Structure
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)
class_count = len(list(train_gen.class_indices.keys())) # to define number of classes in dense layer

# create pre-trained model (you can built on pretrained model such as :  efficientnet, VGG , Resnet )
# we will use efficientnetb3 from EfficientNet family.
base_model = tf.keras.applications.efficientnet.EfficientNetB5(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
    tf.keras.layers.Dense(256, kernel_regularizer= tf.keras.regularizers.l2(l= 0.016), activity_regularizer= tf.keras.regularizers.l1(0.006),
                bias_regularizer= tf.keras.regularizers.l1(0.006), activation= 'relu'),
    tf.keras.layers.Dropout(rate= 0.45, seed= 123),
    tf.keras.layers.Dense(class_count, activation= 'softmax')
])

model.compile(tf.keras.optimizers.Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

model.summary()


batch_size = 32  # set batch size for training
epochs = 50   # number of all epochs in training
patience = 1   #number of epochs to wait to adjust lr if monitored value does not improve
stop_patience = 3   # number of epochs to wait before stopping training if monitored value does not improve
threshold = 0.9   # if train accuracy is < threshold adjust monitor accuracy, else monitor validation loss
factor = 0.5   # factor to reduce lr by
ask_epoch = 5   # number of epochs to run before asking if you want to halt training
batches = int(np.ceil(len(train_gen.labels) / batch_size))    # number of training batch to run per epoch

callbacks = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                             patience=3,
                                             restore_best_weights=True,
                                             verbose=1)
history = model.fit(x= train_gen, epochs= epochs, verbose= 1, callbacks=[callbacks],
                    validation_data= valid_gen, validation_steps= int(len(valid_gen) * 0.2), 
                    shuffle= False)

model.save("C:\Projects\Project1\CancerRecognition\SavedModel\mymodel.h5", overwrite=True)
