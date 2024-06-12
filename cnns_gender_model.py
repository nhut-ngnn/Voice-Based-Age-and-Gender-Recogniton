import tensorflow as tf

from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


from models import train_multi_epoch, train_deepnn

NUM_FEATURES = 54 


def cnns_gender_model(num_labels):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(35, NUM_FEATURES)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def main_class_gender_train():
    dataset = "C:/Users/admin/Documents/Voice_Based_Age_Gender_and_Emotion/New_Project/gender_data_clean"
    model = "C:/Users/admin/Documents/Voice_Based_Age_Gender_and_Emotion/New_Project/model/cnns_gender_"
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    train_multi_epoch(dataset, model + str(NUM_FEATURES),
                      cnns_gender_model, train_deepnn,
                      num_epoch_start=15,
                      num_features=NUM_FEATURES,
                      file_prefix="gender",
                      callbacks=[early_stopping])


if __name__ == '__main__':
    main_class_gender_train()