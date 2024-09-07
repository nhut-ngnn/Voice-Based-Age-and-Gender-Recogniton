import pandas as pd
import numpy as np

from file_io import write_to_file_labels, concat_files
from extract_features import get_features
from utils import get_count, min_label_count

folder_path = "C:/Users/admin/Documents/AgeDetection/voice-bases-age-gender-classification/DataSet/"
english_dataset_path = "ja/"
audio_path = "clips/"

train_type = {"gender", "age"}

file_list = [
    "dev.tsv",
    "invalidated.tsv",
    "other.tsv",
    "test.tsv",
    "train.tsv",
    "validated.tsv"]

out_accent_file = "new_train_accent.tsv"
out_age_file = "new_train_age.tsv"
out_gender_file = "new_train_gender.tsv"


def get_file_data(file_path, features):
    """
    load, clean and return the cleaned data file
    """
    data_frame = pd.read_csv(file_path, sep='\t', header=0)
    data_frame = data_frame.dropna(subset=features)
    if "gender" in features:
        data_frame = data_frame[data_frame["age"] != "teens"]
    if "accent" in features:
        data_frame = data_frame[(data_frame["age"] == "twenties")
                                (data_frame["age"] == "thirties")]
    return data_frame


def prepare_file(filepath, out_file, **kwargs):
    """
    create a new file containing the desired attribute
    """
    train_file_path = filepath + kwargs["filename"]
    feature = kwargs["feature"]
    if type(feature) is not list:
        feature = [feature]
    file_data = get_file_data(train_file_path, features=feature)
    out_path = filepath + out_file
    file_data.to_csv(out_path, sep='\t', index=False)


def get_labels(data_file: pd.DataFrame, label_column: str):
    """
    get the output labels from the dataset
    :return: labels
    """
    return data_file[label_column].values.tolist()


def get_audio(path_audio, data_file: pd.DataFrame):
    """
    get the audio files from the dataset
    :return: audio file paths
    """
    audio_files = path_audio + data_file["path"].str[:99]
    return audio_files.values.tolist()


def get_data(feature, dataset, in_file, out_file):
    """
    get dataset from file
    :return: features, labels
    """
    filepath = folder_path + dataset
    prepare_file(filepath, out_file,
                 filename=in_file,
                 feature=feature)

    dataframe = pd.read_csv(filepath + out_file, sep='\t', header=0)
    audio_data_path = filepath + audio_path
    if type(feature) is list:
        feature = feature[0]

    inputs = get_audio(audio_data_path, dataframe)
    outputs = get_labels(dataframe, feature)

    return inputs, outputs


def create_equal_dataset(input_data, output_data, min_num) -> (np.array, np.array):
    """
    creates an equally sampled dataset based on the given labels and their count
    :param input_data: features
    :param output_data: labels
    :param min_num: minimum number of samples to contain
    """
    filtered_inputs = []
    filtered_outputs = []

    labels_count = get_count(output_data)
    num_samples = dict()
    for key in labels_count.keys():
        num_samples[key] = 0
    _, label_count = min_label_count(labels_count)
    label_count = min(min_num, label_count)

    for i in range(len(output_data)):
        label = output_data[i]
        if num_samples[label] < label_count:
            filtered_inputs.append(input_data[i])
            filtered_outputs.append(label)
            num_samples[label] += 1

    return np.array(filtered_inputs), np.array(filtered_outputs)


def clean_gender_dataset(inputs, outputs) -> (np.array, np.array):
    cleaned_in = []
    cleaned_out = []
    for i in range(len(outputs)):
        if outputs[i] == "other":
            continue
        cleaned_in.append(inputs[i])
        cleaned_out.append(outputs[i])
    return np.array(cleaned_in), np.array(cleaned_out)


def create_gender_dataset(out_data_path, min_samples=5000):
    """
    create the files holding the data for the gender prediction model
    :param out_data_path: where to save the files
    :param min_samples: minimum number of samples
    """
    if out_data_path[-1] != '/':
        out_data_path = out_data_path + '/'
    if min_samples <= 0:
        min_samples = 2 ** 20

    en_input, en_output = get_data(["gender", "age"], english_dataset_path,
                                   file_list[3], out_gender_file)                #Change file_list num to choice dataset be extracted

    inputs, outputs = clean_gender_dataset(en_input, en_output)
    inputs, outputs = create_equal_dataset(inputs, outputs, min_samples)

    print(len(inputs))
    print(len(outputs))
    print(get_count(outputs))

    get_features(out_data_path + "gender_", inputs, ['delta', 'delta2', 'pitch',"constract"])
    write_to_file_labels(out_data_path + "gender_out", outputs)
    in_files = ["gender_input" + str(i + 1) for i in range(6)]
    concat_files(out_data_path, in_files, "gender_in")
    

def clean_age_dataset(inputs, outputs) -> (np.array, np.array):
    cleaned_in = []
    cleaned_out = []
    for i in range(len(outputs)):
        if outputs[i] == "fifties" \
                or outputs[i] == "sixties":
            outputs[i] = "fifties_sixties"
        elif outputs[i] == "eighties" \
                or outputs[i] == "nineties"\
                or outputs[i] == "seventies":
            continue   
        cleaned_in.append(inputs[i])
        cleaned_out.append(outputs[i])
    return np.array(cleaned_in), np.array(cleaned_out)


def create_age_dataset(out_data_path, min_samples=0):
    if out_data_path[-1] != '/':
        out_data_path = out_data_path + '/'
    if min_samples <= 0:
        min_samples = 2 ** 20

    en_input, en_output = get_data("age", english_dataset_path, file_list[3], out_age_file)     #Change file_list num to choice dataset be extracted

    inputs, outputs = clean_age_dataset(en_input, en_output)
    inputs, outputs = create_equal_dataset(inputs, outputs, min_samples)

    print(len(inputs))
    print(len(outputs))
    print(get_count(outputs))

    get_features(out_data_path + "age_", inputs, ['delta', 'delta2', 'pitch','constract'])
    write_to_file_labels(out_data_path + "age_out", outputs)
    in_files = ["age_input" + str(i + 1) for i in range(6)]
    concat_files(out_data_path, in_files, "age_in")



if __name__ == "__main__":
    # gender_data_folder = "C:/Users/admin/Documents/Voice_Based_Age_Gender_and_Emotion/New_Project/gender_data_clean"
    test_gender_folder = "C:/Users/admin/Documents/Voice_Based_Age_Gender_and_Emotion/New_Project/gender_data_clean_test"
    # gender_data_small_folder = "C:/Users/admin/Documents/Voice_Based_Age_Gender_and_Emotion/New_Project/gender_data_clean_small"

    # age_data_folder = "C:/Users/admin/Documents/Voice_Based_Age_Gender_and_Emotion/New_Project/age_data_clean"
    test_age_folder = "C:/Users/admin/Documents/Voice_Based_Age_Gender_and_Emotion/New_Project/age_data_clean_test"
    # age_data_small_folder = "C:/Users/admin/Documents/Voice_Based_Age_Gender_and_Emotion/New_Project/age_data_clean_small"
   
    create_gender_dataset(test_gender_folder)
    create_age_dataset(test_age_folder)