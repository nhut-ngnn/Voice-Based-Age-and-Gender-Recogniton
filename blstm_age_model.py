
import keras_metrics as km
from keras import metrics
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from models import train_multi_epoch, train_deepnn

NUM_FEATURES = 145  


def lstm_age_model(num_labels):
    model = Sequential() 
    model.add(Bidirectional(LSTM(256, dropout=0.3, return_sequences=True), input_shape=(35, NUM_FEATURES)))
    model.add(Bidirectional(LSTM(256, dropout=0.3)))
    model.add(Dense(128 * 2, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(num_labels, activation='softmax'))
    optimizer=Adam(0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def main_class_age_train():
    dataset = "C:/Users/admin/Documents/Voice_Based_Age_Gender_and_Emotion/New_Project/age_data_clean_small"  
    model = "C:/Users/admin/Documents/Voice_Based_Age_Gender_and_Emotion/New_Project/model/blstm_age_"
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    train_multi_epoch(dataset, model + str(NUM_FEATURES),
                      lstm_age_model, train_deepnn,
                      num_epoch_start=70,
                      num_features=NUM_FEATURES,
                      file_prefix="age",
                      callbacks=[early_stopping])


if __name__ == '__main__':
    main_class_age_train()
