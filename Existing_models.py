# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Conv1D ,DepthwiseConv1D,MaxPooling1D,Dropout,Flatten,Dense,BatchNormalization, Embedding,Bidirectional, LSTM,Activation
from keras.models import Model,Sequential


########                                                                       ########
########                                          LSTM                         ########
########                                                                       ########


def LS_TM(input_shape,n_classes):
    LSTM_model = Sequential()
    LSTM_model.add(Conv1D(filters=16, kernel_size=2, activation='relu',input_shape=(input_shape)))
    LSTM_model.add(Dropout(0.3)) 
    LSTM_model.add(Conv1D(16, kernel_size=2, activation='relu'))
    LSTM_model.add(LSTM(20))
    LSTM_model.add(Dropout(0.2))
    LSTM_model.add(Dense(10, activation='relu'))
    LSTM_model.add(Dropout(0.3))
    
    if n_classes==2:
        LSTM_model.add(Dense(1, activation='sigmoid'))
        LSTM_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    else:
        LSTM_model.add(Dense(n_classes, activation='softmax'))
        LSTM_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return LSTM_model

########                                                                       ########
########                                          CNN                          ########
########                                                                       ########

def CNN(input_shape,n_classes):
    x_input = tf.keras.layers.Input(shape=input_shape)
    layer1 = Conv1D(16, kernel_size=(1), activation='relu',padding='same')(x_input)
    layer2 = Conv1D(8, kernel_size=(1), activation='relu',padding='same')(layer1)
    layer3 = MaxPooling1D(pool_size=(1))(layer2)
    layer4 = Dropout(0.5)(layer3)
    layer5 = Flatten()(layer4)
    layer6 = Dense(100, activation='relu')(layer5)
    layer7 = Dense(n_classes, activation='softmax')(layer6)
    model = Model([x_input], layer7)
    
    if n_classes==2:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    else:
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

        
        
        
    

########                                                                       ########
########                                        BILSTM                         ########
########                                                                       ########



def BI_LSTM(input_shape,n_classes):
    model = Sequential()
    model.add(Embedding(input_shape[0]+1,input_shape[1], input_length=input_shape[0]))
    model.add(Bidirectional(LSTM(8)))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    
    if n_classes==2:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    else:
         model.add(Dense(n_classes, activation='softmax'))
         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    

########                                                                       ########
########                                      Alex Net                         ########
########                                                                       ########



def Alex_Net(Input_shape,n_classes):
    model = Sequential()
    model.add(Conv1D(filters = 16,   kernel_size = (1),strides = (1),input_shape =Input_shape,  padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size = (1), strides = (1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Conv1D(filters = 8, kernel_size = (1), strides = (1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size = (1), strides = (1),  padding = 'same'))
    model.add(BatchNormalization())
    model.add(Conv1D(filters = 8, kernel_size = (1),  strides = (1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(50 ,activation="relu"))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(n_classes,activation="softmax"))
    
    if n_classes==2:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    else:
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    

########                                                                       ########
########                                      VGG Net                          ########
########                                                                       ########




def Vgg(Input_shape,n_classes):
    vggmodel = Sequential()
    vggmodel.add(Conv1D(filters=8,kernel_size=(3),padding="same", activation="relu",input_shape=(Input_shape)))
    vggmodel.add(Conv1D(filters=8,kernel_size=(3),padding="same", activation="relu"))
    vggmodel.add(MaxPooling1D(pool_size=(2),strides=(2)))
    vggmodel.add(Conv1D(filters=16, kernel_size=(3), padding="same", activation="relu"))
    vggmodel.add(Conv1D(filters=16, kernel_size=(3), padding="same", activation="relu"))
    vggmodel.add(MaxPooling1D(pool_size=(2),strides=(2)))
    vggmodel.add(Conv1D(filters=32, kernel_size=(3), padding="same", activation="relu"))
    vggmodel.add(Conv1D(filters=32, kernel_size=(3), padding="same", activation="relu"))
    vggmodel.add(Conv1D(filters=32, kernel_size=(3), padding="same", activation="relu"))
    vggmodel.add(MaxPooling1D(pool_size=(2),strides=(2)))
    vggmodel.add(Conv1D(filters=64, kernel_size=(3), padding="same", activation="relu"))
    vggmodel.add(Conv1D(filters=64, kernel_size=(3), padding="same", activation="relu"))
    vggmodel.add(Conv1D(filters=64, kernel_size=(3), padding="same", activation="relu"))
    vggmodel.add(MaxPooling1D(pool_size=(2),strides=(2)))
    vggmodel.add(Conv1D(filters=64, kernel_size=(3), padding="same", activation="relu"))
    vggmodel.add(Conv1D(filters=64, kernel_size=(3), padding="same", activation="relu"))
    vggmodel.add(Conv1D(filters=64, kernel_size=(3), padding="same", activation="relu"))
    vggmodel.add(MaxPooling1D(pool_size=(2),strides=(2)))
    vggmodel.add(Flatten())
    vggmodel.add(Dense(units=100,activation="relu"))
    vggmodel.add(Dense(units=50,activation="relu"))
    
    if n_classes==2:
        vggmodel.add(Dense(1, activation='sigmoid'))
        vggmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    else:
        vggmodel.add(Dense(n_classes, activation='softmax'))
        vggmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return vggmodel
    


    
########                                                                       ########
########                                      RESNet                         ########
########                                                                       ########




def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = Conv1D(filter, (3), padding = 'same', strides = (2))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = Conv1D(filter, (3), padding = 'same')(x)
    x = BatchNormalization()(x)
    # Processing Residue with conv(1,1)
    x_skip = Conv1D(filter, (1), strides = (2))(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x

def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = Conv1D(filter, (3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = Conv1D(filter, (3), padding = 'same')(x)
    x =BatchNormalization()(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x



def ResNet34(Input_shape,n_classes):
    # Step 1 (Setup Input Layer)
    x_input = tf.keras.layers.Input(Input_shape)
    x = tf.keras.layers.ZeroPadding1D((3, 3))(x_input)
    # Step 2 (Initial Conv layer along with maxPool)
    x = Conv1D(8, kernel_size=2, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 8
    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    # Step 4 End Dense Network
    x = tf.keras.layers.AveragePooling1D((2), padding = 'same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(500, activation = 'relu')(x)
    if n_classes==2:
        x = tf.keras.layers.Dense(n_classes, activation = 'sigmoid')(x)
        model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "ResNet34")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
       
    else:
        x = tf.keras.layers.Dense(n_classes, activation = 'softmax')(x)
        model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "ResNet34")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    











