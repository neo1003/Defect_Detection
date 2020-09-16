import numpy as np
import pandas as pd
import os
import random

from keras_preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import tensorflow.keras.preprocessing.image as image
from tensorflow.keras import layers
from tensorflow.keras import models
import  tensorflow.keras.optimizers as optimizers
model = models.Sequential()

if __name__ == '__main__':
	#Definde folder path
	folder_good = 'train/good/'
	folder_bad = 'train/bad'

	#This folder contains all images
    folder_main ='train_final'

	#DataAugmentation
	datagen = ImageDataGenerator(
    	rotation_range=30,
    	horizontal_flip=True,
    	width_shift_range=0.1,
    	height_shift_range=0.1,
		fill_mode='nearest'
	)

	filenames_good = os.listdir(folder_good)
	for filename in filenames_good:
		img= load_img(folder_good + filename)
		img_array = img_to_array(img)
		img_array = img_array.reshape((1,)+img_array.shape)
		i=0

		for batch in datagen.flow(img_array,batch_size=1,save_to_dir=folder_main,save_prefix='good.',save_format='tif'):
			i=i+1
			if i < 10:
				break

	filenames_bad = os.listdir(folder_bad)
	for filename in filenames_good:
		img= load_img(folder_bad + filename)
		img_array = img_to_array(img)
		img_array = img_array.reshape((1,)+img_array.shape)
		i=0

		for batch in datagen.flow(img_array,batch_size=1,save_to_dir=folder_main,save_prefix='bad.',save_format='tif'):
			i=i+1
			if i < 30:
				break

	# gather images into a dataframe
	filenames_main = os.listdir(folder_main)
	categories = []
	for filename in filenames_main:
		category = filename.split('.')[0]
		if category == 'good':
			categories.append('good')
		else:
			categories.append('bad')


	all_df = pd.DataFrame({
		'filename': filenames_main,
		'category': categories
	})


	# split into train/validate and test sets
	train_validate, test_df = train_test_split(all_df, test_size=0.20, random_state=0)
	train_validate = train_validate.reset_index(drop=True)
	test_df = test_df.reset_index(drop=True)

	# split train/validate into train and validation sets
	train_df, validate_df = train_test_split(train_validate, test_size=0.20, random_state=0)
	train_df = train_df.reset_index(drop=True)
	validate_df = validate_df.reset_index(drop=True)


	train_datagen = image.ImageDataGenerator(rescale=1./255)
	validate_datagen = image.ImageDataGenerator(rescale=1./255)
	test_datagen = image.ImageDataGenerator(rescale=1./255)


	## using ImageDataGenerator to read train_df images from directories
	train_generator = train_datagen.flow_from_dataframe(
    	train_df,
    	folder_main,
    	x_col='filename',
    	y_col='category',
    	target_size=(128, 128),
    	class_mode='binary',
    	batch_size=5
	)


	## using ImageDataGenerator to read validate_df images from directories
	validation_generator = validate_datagen.flow_from_dataframe(
    	validate_df,
		folder_main,
   		x_col='filename',
    	y_col='category',
    	target_size=(128, 128),
    	class_mode='binary',
    	batch_size=2
	)

	## using ImageDataGenerator to read test_df images from directories
	test_generator = test_datagen.flow_from_dataframe(
    	test_df,
    	folder_main,
    	x_col='filename',
    	y_col='category',
    	target_size=(128,128),
    	class_mode='binary',
    	batch_size=50
	)



	# convolutional-base
	model.add(layers.Conv2D(32, (3, 3), activation='relu',
	input_shape=(128, 128, 3)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(128, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(256, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Flatten())

	# densely connected classifier
	model.add(layers.Dense(256, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	model.summary()


	# optimizing model performance
	from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
	callbacks = [
    	EarlyStopping(patience=10, verbose=1),
    	ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
	]


	# configuring the model for training

	model.compile(loss='binary_crossentropy',
	optimizer=optimizers.RMSprop(lr=1e-4),
	metrics=['acc'])


	# fitting the model using a batch generator
	history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=100,
        callbacks=callbacks,)


	#Calculating Accuracy
	test_loss, test_acc = model.evaluate_generator(test_generator, steps=100)
	print('test acc:', test_acc)




