import tensorflow as tf

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping


from models import train_multi_epoch, train_deepnn

NUM_FEATURES = 145 


def lstm_gender_model(num_labels):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(35, NUM_FEATURES, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(num_labels, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam',
    #               metrics=['accuracy'])
    # return model
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def main_class_gender_train():
    dataset = "C:/Users/admin/Documents/Voice_Based_Age_Gender_and_Emotion/New_Project/gender_data_clean"
    model = "C:/Users/admin/Documents/Voice_Based_Age_Gender_and_Emotion/New_Project/model/lstm_gender_"
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    train_multi_epoch(dataset, model + str(NUM_FEATURES),
                      lstm_gender_model, train_deepnn,
                      num_epoch_start=30,
                      num_features=NUM_FEATURES,
                      file_prefix="gender",
                      callbacks=[early_stopping])


if __name__ == '__main__':
    main_class_gender_train()