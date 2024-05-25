import tensorflow as tf
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Add, ReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from models import train_multi_epoch, train_deepnn

NUM_FEATURES = 145

def residual_block(x, filters, kernel_size=3, stride=1):      
    shortcut = x
    
    if x.shape[-1] != filters or stride != 1:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(x)
        shortcut = BatchNormalization()(shortcut)
    
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Add()([x, shortcut])
    x = ReLU()(x)
    
    return x

def rezoNet_age_model(num_labels):          
    input_shape = (35, NUM_FEATURES, 1)
    inputs = Input(shape=input_shape)
    
    # Initial convolution and max pooling
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    
    x = residual_block(x, 128, stride=1)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    
    x = residual_block(x, 256, stride=1)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    
    x = residual_block(x, 512, stride=1)
    x = MaxPooling2D(pool_size=(1, 2))(x)
    x = Dropout(0.3)(x)
    
    # Fully connected layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    
    outputs = Dense(num_labels, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    # optimizer = Adam(0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    return model

def main_class_age_train():
    dataset = "C:/Users/admin/Documents/Voice_Based_Age_Gender_and_Emotion/New_Project/age_data_clean"  
    model = "C:/Users/admin/Documents/Voice_Based_Age_Gender_and_Emotion/New_Project/model/rezoNet_age_"
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    train_multi_epoch(dataset, model + str(NUM_FEATURES),
                      rezoNet_age_model, train_deepnn,
                      num_epoch_start=70,
                      num_features=NUM_FEATURES,
                      file_prefix="age",
                      callbacks=[early_stopping])


if __name__ == '__main__':
    main_class_age_train()