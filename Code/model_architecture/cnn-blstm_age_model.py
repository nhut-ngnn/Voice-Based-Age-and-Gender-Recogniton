import tensorflow as tf
from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, Bidirectional, LSTM, Dense, BatchNormalization, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


from models import train_multi_epoch, train_deepnn

NUM_FEATURES = 145  


def cnn_bilstm_age_model(num_labels):
    model = Sequential() 
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(35, NUM_FEATURES)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    
    # BiLSTM layers
    model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)))
    model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)))
    model.add(Bidirectional(LSTM(256, dropout=0.3, recurrent_dropout=0.3)))
    
    # Dense layers with Batch Normalization and Dropout
    model.add(Dense(256, activation='relu'))
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
    dataset = "C:/Users/admin/Documents/Voice_Based_Age_Gender_and_Emotion/New_Project1/age_data_clean"  
    model = "C:/Users/admin/Documents/Voice_Based_Age_Gender_and_Emotion/New_Project1/model/blstm_age_"
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    train_multi_epoch(dataset, model + str(NUM_FEATURES),
                      cnn_bilstm_age_model, train_deepnn,
                      num_epoch_start=60,
                      num_features=NUM_FEATURES,
                      file_prefix="age",
                      callbacks=[early_stopping])


if __name__ == '__main__':
    main_class_age_train()
