
import keras_metrics as km
from keras import metrics
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from models import train_multi_epoch, train_deepnn

NUM_FEATURES = 145  


def lstm_age_model(num_labels):
    model = Sequential()
    model.add(LSTM(256, input_shape=(35, NUM_FEATURES), return_sequences=True, dropout=0.3))
    model.add(LSTM(256, dropout=0.3))
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
    optimizer=Adam(0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                   metrics=['accuracy'])#, metrics.F1Score(threshold=0.5)])
    return model


def main_class_age_train():
    dataset = "C:/Users/admin/Documents/AgeDetection/voice-bases-age-gender-classification/age_data_clean"  
    model = "C:/Users/admin/Documents/AgeDetection/voice-bases-age-gender-classification/model/lstm_age_"
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    train_multi_epoch(dataset, model + str(NUM_FEATURES),
                      lstm_age_model, train_deepnn,
                      num_epoch_start=50,
                      num_features=NUM_FEATURES,
                      file_prefix="age",
                      callbacks=[early_stopping])


if __name__ == '__main__':
    main_class_age_train()
