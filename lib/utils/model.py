"""
Author: Chengming He
Models for labeling cleaning and image classification (Model2)
"""


import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.efficientnet import EfficientNetB5,EfficientNetB0
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten,Dense,Concatenate,add,CategoryEncoding,BatchNormalization,Softmax
import tensorflow as tf
from tensorflow.keras import regularizers

class LabelCleaning(Model):
	def __init__(self):
		super(LabelCleaning,self).__init__()
		self.dense1 = Dense(502,activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    		bias_regularizer=regularizers.L2(1e-4),activity_regularizer=regularizers.L2(1e-5))
		self.bn_1 = BatchNormalization()
		self.dense2 = Dense(10,activation='relu')
		self.bn_2 = BatchNormalization()
		self.dense3 = Dense(512,activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    		bias_regularizer=regularizers.L2(1e-4),activity_regularizer=regularizers.L2(1e-5))
		self.bn_3 = BatchNormalization()
		self.dense4 = Dense(10)
		self.bn_4 = BatchNormalization()
		self.sm = Softmax()
	def call(self,x,y,training=False):
		"""
		x: input feature vectors of shape (none,2048)
		y: noisy labels of shape (none,10)
		"""
		x1 = self.dense1(x)
		x1 = self.bn_1(x1,training=training)
		y = CategoryEncoding(num_tokens=10, output_mode="one_hot")(y)
		y = tf.cast(y, tf.float32)
		x2 = self.dense2(y)
		x2 = self.bn_2(x2,training=training)
		x = Concatenate()([x1, x2])
		x = self.dense3(x)
		x = self.bn_3(x,training=training)
		x = self.dense4(x)
		x = self.bn_4(x,training=training)
		x = add([x,y])

		return self.sm(x)

class ImageClassifier(Model):
	def __init__(self):
		super(ImageClassifier,self).__init__()
		self.fc1 = Dense(512,activation='elu')
		self.dense_classify = Dense(10,activation='softmax')
	def call(self,x):
		return self.dense_classify(x)


class NoisyNet(Model):
	def __init__(self, input_shape=(32,32,3),base_model='resnet50'):
		super(NoisyNet, self).__init__()
		if base_model == 'resnet50':
			model = ResNet50(input_shape=input_shape,include_top=False)
			self.base = Model(model.input, model.layers[-2].output)
		if base_model == 'EfficientNetB5':
			model = EfficientNetB5(input_shape=input_shape,include_top=False)
			self.base = Model(model.input, model.layers[-2].output)
		if base_model == 'densenet':
			model = DenseNet121(input_shape=input_shape,include_top=False)
			self.base = Model(model.input, model.layers[-2].output)
		if base_model == 'EfficientNetB0':
			model = EfficientNetB0(input_shape=input_shape,include_top=False)
			self.base = Model(model.input, model.layers[-2].output)

		self.flatten = Flatten()
		self.ImageClassifier = ImageClassifier()
		self.LabelCleaning = LabelCleaning()

	def call(self,inputs,training=False):
		"""inputs: list"""
		for layer in self.base.layers: layer.trainable =False
		# for layer in self.base.layers[-2:]: layer.trainable =True
		(x,y) = (inputs[0],inputs[1]) if len(inputs)==2 else  (inputs[0],None)
		x = self.base(x)
		x = self.flatten(x)
		if y is None: 
			for layer in self.LabelCleaning.layers: layer.trainable =False
			for layer in self.ImageClassifier.layers: layer.trainable =True
			return self.ImageClassifier(x)
		else:
			for layer in self.ImageClassifier.layers: layer.trainable =False
			for layer in self.LabelCleaning.layers: layer.trainable =True
			return self.LabelCleaning(x,y,training=training)

		





