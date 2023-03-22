import sys
import tensorflow as tf
sys.path.append("C:\\Projects\\Project1\\CancerRecognition\\Helper_Functions\\")
from helper_functions import plot_label_count, show_images, split_data, create_model_data
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

Data_dir = 'C:\\Projects\\Project1\\CancerRecognition\\Data\\dataset'
train_dir = 'C:\\Projects\\Project1\\CancerRecognition\\Data\\dataset'
valid_dir = '' 
test_dir = ''


try:
    # Get splitted data
    train_df, valid_df, test_df = split_data(train_dir, valid_dir, test_dir)

    # Get Generators
    batch_size = 32
    train_gen, valid_gen, test_gen = create_model_data(train_df, valid_df, test_df, batch_size)

except:
    print('Invalid Input')

show_images(train_gen)

plot_label_count(train_df, 'train')

g_dict = train_gen.class_indices        # defines dictionary {'class': index}
classes = list(g_dict.keys())  
print(classes)