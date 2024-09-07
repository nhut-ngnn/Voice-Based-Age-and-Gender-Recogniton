import tensorflow as tf

from keras import Sequential
from keras.layers import Bidirectional, LSTM, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


from models import train_multi_epoch, train_deepnn

NUM_FEATURES = 54 


def cnns_gender_model(num_labels):
    model = Sequential()
    model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3), input_shape=(35, NUM_FEATURES)))
    model.add(Bidirectional(LSTM(256, dropout=0.3, recurrent_dropout=0.3)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(0.0001))
    model.summary()
    return model


def main_class_gender_train():
    dataset = "C:/Users/admin/Documents/Voice_Based_Age_Gender_and_Emotion/New_Project/gender_data_clean"
    model = "C:/Users/admin/Documents/Voice_Based_Age_Gender_and_Emotion/New_Project/model/blstm_gender_"
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    train_multi_epoch(dataset, model + str(NUM_FEATURES),
                      cnns_gender_model, train_deepnn,
                      num_epoch_start=15,
                      num_features=NUM_FEATURES,
                      file_prefix="gender",
                      callbacks=[early_stopping])


if __name__ == '__main__':
    main_class_gender_train()
