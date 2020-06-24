import tensorflow_hub as hub

from tensorflow.keras import Model
from  tensorflow.keras.layers import Dense, Softmax, Flatten
from tensorflow.keras.applications import ResNet50


# Define our model with keras model subclassing
class ResNet_50(Model):
    def __init__(self, fine_tune=False):
        super(ResNet_50, self).__init__()
        self.backbone = hub.KerasLayer(
            'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4',
            trainable=fine_tune,
            #output_shape=[1280],
        )
        self.dense = Dense(units=2)
        self.softmax = Softmax()
    
    def call(self, x):
        h = self.backbone(x)
        h = self.dense(h)
        y = self.softmax(h)
        return y

class ResNet_perso_50(Model):
    def __init__(self, resnet_trainable_layers=0, **kwargs):
        super(ResNet_perso_50, self).__init__(**kwargs)

        self.resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
    
        for layer in self.resnet.layers:
            layer.trainable=False
        if resnet_trainable_layers!=0:
            for layer in self.resnet.layers[-resnet_trainable_layers:]:
                layer.trainable=True

        self.flatten = Flatten()
        self.dense = Dense(2, activation='softmax') 

    def call(self, x):
        h = self.resnet(x)
        h = self.flatten(h)
        y = self.dense(h)
        return y

