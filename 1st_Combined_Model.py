#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import keras_tuner as kt
import seaborn as sns
import sys, os
import sklearn

#limit tensorflow to only one gpu
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#prevent tensorflow from pre-allocating GPU memory
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import tensorflow as tf

TM_MODE = 0
TE_MODE = 1
batch_size = 512

norm_layer = keras.layers.Normalization(dtype='float32', axis=-1, mean=[200e-9, 200e-9, 200e-9, 200e-9, 2.3, 1.55e-6], variance=[76800e-18, 76800e-18, 76800e-18, 76800e-18, 0.1633, 3.3333e-15])

policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)
print("Set Keras Policy")


# In[2]:


# For Dataset 1


# In[3]:


training_data_file_paths = [
    "ANT1SWG_production_data_nov_26_22.dat",
    "ANT1SWG_production_data_nov_27_22_joshmachine.dat",
    "ANT1SWG_production_data_dec_3_22_3000_2.dat",
    "ANT1SWG_production_data_dec_4_22_3000_1.dat",
    "ANT1SWG_production_data_dec_4_22_3000_2.dat",
    "DatagenANT1SWG_pList_dec_9_22_longLvalues_2_data.dat",
    "DatagenANT1SWG_pList_dec_9_22_longLvalues_data.dat",
]


testing_data_file_paths = [
    "Testing/ANT1SWG_production_data_dec_3_22_3000_1.dat",
    "Testing/ANT1SWG_production_data_nov_28_22_joshmachine.dat",
]


# In[4]:


training_data_file_paths


# In[5]:


testing_data_file_paths


# In[6]:


data1 = pd.read_csv('ANT1SWG_production_data_nov_26_22.dat', delimiter='\t')
random_samples_1 = data1.sample(n=1759, random_state=42)  
random_samples_1.to_csv('random_samples_1.dat', sep='\t', index=False)
data1 = np.loadtxt('random_samples_1.dat')
print(data1.shape)


data2 = pd.read_csv('ANT1SWG_production_data_nov_27_22_joshmachine.dat', delimiter='\t')
random_samples_2 = data2.sample(n=444, random_state=42)  
random_samples_2.to_csv('random_samples_2.dat', sep='\t', index=False)
data2 = np.loadtxt('random_samples_2.dat')
print(data2.shape)

data3 = pd.read_csv('ANT1SWG_production_data_dec_3_22_3000_2.dat', delimiter='\t')
random_samples_3 = data3.sample(n=1318, random_state=42)  
random_samples_3.to_csv('random_samples_3.dat', sep='\t', index=False)
data3 = np.loadtxt('random_samples_3.dat')
print(data3.shape)

data4 = pd.read_csv('ANT1SWG_production_data_dec_4_22_3000_1.dat', delimiter='\t')
random_samples_4 = data4.sample(n=1318, random_state=42)  
random_samples_4.to_csv('random_samples_4.dat', sep='\t', index=False)
data4 = np.loadtxt('random_samples_4.dat')
print(data4.shape)

data5 = pd.read_csv('ANT1SWG_production_data_dec_4_22_3000_2.dat', delimiter='\t')
random_samples_5 = data5.sample(n=1318, random_state=42)  
random_samples_5.to_csv('random_samples_5.dat', sep='\t', index=False)
data5 = np.loadtxt('random_samples_5.dat')
print(data5.shape)

data6 = pd.read_csv('DatagenANT1SWG_pList_dec_9_22_longLvalues_2_data.dat', delimiter='\t')
random_samples_6 = data6.sample(n=1318, random_state=42)  
random_samples_6.to_csv('random_samples_6.dat', sep='\t', index=False)
data6 = np.loadtxt('random_samples_6.dat')
print(data6.shape)

data7 = pd.read_csv('DatagenANT1SWG_pList_dec_9_22_longLvalues_data.dat', delimiter='\t')
random_samples_7 = data7.sample(n=1318, random_state=42)  
random_samples_7.to_csv('random_samples_7.dat', sep='\t', index=False)
data7 = np.loadtxt('random_samples_7.dat')
print(data7.shape)

#Testing

data8 = pd.read_csv('Testing/ANT1SWG_production_data_dec_3_22_3000_1.dat', delimiter='\t')
random_samples_8 = data8.sample(n=1651, random_state=42)  
random_samples_8.to_csv('random_samples_8.dat', sep='\t', index=False)
data8 = np.loadtxt('random_samples_8.dat')
print(data8.shape)

data9 = pd.read_csv('Testing/ANT1SWG_production_data_nov_28_22_joshmachine.dat', delimiter='\t')
random_samples_9 = data9.sample(n=547, random_state=42)  
random_samples_9.to_csv('random_samples_9.dat', sep='\t', index=False)
data9 = np.loadtxt('random_samples_9.dat')
print(data9.shape)





# In[7]:


#For dataset 2


# In[8]:


from sklearn.model_selection import train_test_split

data = np.loadtxt('2D_3seg_jun_27_output.dat')
print(data.shape)

import seaborn as sns
import matplotlib.pyplot as plt

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_file_path = 'train_data.dat'
test_file_path = 'test_data.dat'

# Save training and testing data to .dat files
np.savetxt(train_file_path, train_data)
np.savetxt(test_file_path, test_data)

training_data_file_paths = [train_file_path]
testing_data_file_paths = [test_file_path]

print("Training data saved to:", train_file_path)
print("Testing data saved to:", test_file_path)
print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)





training_data_file_paths = [
    "random_samples_1.dat",
    "random_samples_2.dat",
    "random_samples_3.dat",
    "random_samples_4.dat",
    "random_samples_5.dat",
    "random_samples_6.dat",
    "random_samples_7.dat",
    "train_data.dat"]
testing_data_file_paths = [
    "random_samples_8.dat",
    "random_samples_9.dat",
    "test_data.dat"]







wavelengths = np.loadtxt("wavelengths.dat", delimiter=',')


# In[13]:


def seperate_out_by_wavelength(linked_data):
    for_output_features = []
    for_output_labels = []
    for data in linked_data:
        for tlabel, tuplabel, tdownlabel, rlabel, wavelength in zip(data[0], data[1], data[2], data[3], wavelengths):
            feature = [*data[4], wavelength]
            for_output_features.append(feature)
            for_output_labels.append([tlabel, tuplabel, tdownlabel, rlabel])
    return (np.array(for_output_features), np.array(for_output_labels))


# In[14]:


def load_data_into_arrays(file_paths, shuffle=False):
    #Link labels and features to shuffle, avoiding having same geometric parameters in training and testing data
    temp_combined = []

    for file_path in file_paths:
        file = np.loadtxt(file_path)
        T_matrix = file[:,0:100]
        Tup_matrix = file[:,100:200]
        Tdown_matrix = np.abs(file[:,200:300])
        R_matrix = np.abs(file[:,300:400])

        raw_features = np.array(file[:,400:410]) #missing wavelength still
        # [L1 L2 L3 L4 Lam Nswg _ SiLength Pmode]
        raw_features = np.delete(raw_features, 7, 1) #removing Si_Length
        raw_features = np.delete(raw_features, 6, 1) #removing Center Height
        raw_features = np.delete(raw_features, 4, 1) #removing Lam

        for t_row, tup_row, tdown_row, r_row, temp_feature in zip(T_matrix, Tup_matrix, Tdown_matrix, R_matrix, raw_features):
            if temp_feature[-1] == TM_MODE:
                temp_combined.append([t_row, tup_row, tdown_row, r_row, temp_feature[0:-1]]) #combine removing pmode as a feature
    temp_combined = np.array(temp_combined, dtype=object)
    
    if shuffle:
        np.random.shuffle(temp_combined) #need to seed for perfect reproducibility

    return seperate_out_by_wavelength(temp_combined)

train_features, train_labels = load_data_into_arrays(training_data_file_paths, shuffle=True)
test_features, test_labels = load_data_into_arrays(testing_data_file_paths)


print(train_labels.shape[0])

# for training
labl = [] 
ft = []
for kk in range(int(train_labels.shape[0]/100)):
    labl.append(train_labels[kk*100+70:kk*100+100])
    ft.append(train_features[kk*100+70:kk*100+100])
    
    #print(test_labels[kk*100+70:kk*100+100])
#test_features = test_features[70:99]

labl = np.array(labl[::2])
labl = labl.reshape(-1, labl.shape[-1])

ft = np.array(ft[::2])
ft = ft.reshape(-1, ft.shape[-1])
train_labels = labl
train_features = ft



print(test_labels.shape[0])

# for testing
labl1 = [] 
ft1 = []
for kk1 in range(int(test_labels.shape[0]/100)):
    labl1.append(test_labels[kk1*100+70:kk1*100+100])
    ft1.append(test_features[kk1*100+70:kk1*100+100])
    
    #print(test_labels[kk*100+70:kk*100+100])
#test_features = test_features[70:99]

labl1 = np.array(labl1[::2])
labl1 = labl1.reshape(-1, labl1.shape[-1])

ft1 = np.array(ft1[::2])
ft1 = ft1.reshape(-1, ft1.shape[-1])
test_labels = labl1
test_features = ft1


#print(test_features.shape)
#test_labels
#print(wavelengths[70])
#print(test_labels)

wavelengths = wavelengths[70:100]
wavelengths = wavelengths[::2]


print(f"Training features: {train_features.shape}, labels {train_labels.shape} \nTesting Features {test_features.shape}, labels = {test_labels.shape}")

#turn into tensorflow datasets for batching etc.
training_data = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
test_data = tf.data.Dataset.from_tensor_slices((test_features, test_labels))

#configure batches
training_data = training_data.shuffle(buffer_size=100000, reshuffle_each_iteration=True).batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
test_data = test_data.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)

training_data = training_data.prefetch(batch_size)
test_data = test_data.prefetch(batch_size)



# In[15]:


initializer_seed = 5678
def model_creator():
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        layer_node_count = [7, 100, 200, 400, 400, 800, 800, 800, 400, 400, 200, 200, 100, 100, 50, 20]

        modelv1 = keras.Sequential()
        modelv1.add(norm_layer)
        for node_count in layer_node_count:
            modelv1.add(layers.Dense(node_count, kernel_initializer=keras.initializers.random_normal(seed=initializer_seed)))
            modelv1.add(layers.LeakyReLU())

        modelv1.add(layers.Dense(4, activation=keras.activations.linear, dtype='float32'))

        modelv1.compile(loss=keras.losses.MeanSquaredError(),
                        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                        metrics=[keras.metrics.MeanAbsolutePercentageError(), keras.metrics.MeanAbsoluteError(), keras.metrics.MeanSquaredError()])
    return modelv1

modelv1 = model_creator()


# In[16]:


model_coded_name = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_doubledkite_manyextralayers_leaky_adam_learningdecay_batch{batch_size}_seed{initializer_seed}"

file_path_for_save = 'C:/Users/smitp/OneDrive/Desktop/Dataset/Saving_Cloud'+model_coded_name+'.h5py'

file_path_for_checkpoints = 'C:/Users/smitp/OneDrive/Desktop/Dataset/Saving_Cloud'+model_coded_name

log_dir = "C:/Users/smitp/OneDrive/Desktop/Dataset/Saving_Cloud" + model_coded_name

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path_for_checkpoints+".{epoch:02d}.h5py", save_freq=3907*20) #save every 20 epochs
learning_rate_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-5, min_delta=1e-6)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True)



# In[17]:


history = modelv1.fit(training_data, epochs=1000, callbacks=[tensorboard_callback, learning_rate_callback, model_checkpoint], validation_data=test_data, validation_freq=1, verbose=2)


# In[18]:


modelv1.save(file_path_for_save)




