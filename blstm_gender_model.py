import tensorflow as tf

from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from keras.callbacks import EarlyStopping


from models import train_multi_epoch, train_deepnn

NUM_FEATURES = 145 


def lstm_gender_model(num_labels):
    model = Sequential() 
    model.add(Bidirectional(LSTM(256, dropout=0.3, return_sequences=True), input_shape=(35, NUM_FEATURES)))
    model.add(Bidirectional(LSTM(256, dropout=0.3)))
    model.add(Dense(128 * 2, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def main_class_gender_train():
    dataset = "C:/Users/admin/Documents/Voice_Based_Age_Gender_and_Emotion/New_Project/gender_data_clean"
    model = "C:/Users/admin/Documents/Voice_Based_Age_Gender_and_Emotion/New_Project/model/blstm_gender_"
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    train_multi_epoch(dataset, model + str(NUM_FEATURES),
                      lstm_gender_model, train_deepnn,
                      num_epoch_start=30,
                      num_features=NUM_FEATURES,
                      file_prefix="gender",
                      callbacks=[early_stopping])


if __name__ == '__main__':
    main_class_gender_train()
