# Voice-Based Age and Gender Recognition: A Comparative Study of LSTM, RezoNet, and Hybrid CNNs-BiLSTM Architecture</h1>
<i>Official implementation for the paper: **Voice-Based Age and Gender Recognition: A Comparative Study of LSTM, RezoNet and Hybrid CNNs-BiLSTM Architecture.** The paper has been accepted to <a href="https://ictc.org/">The 15th International Conference on ICT Convergence (ICTC2024).</a></i>
> Please press ⭐ button and/or cite papers if you feel helpful.


<p align="center">
<img src="https://img.shields.io/github/stars/nhut-ngnn/Voice-Based-Age-and-Gender-Recogniton">
<img src="https://img.shields.io/github/forks/nhut-ngnn/Voice-Based-Age-and-Gender-Recogniton">
<img src="https://img.shields.io/github/watchers/nhut-ngnn/Voice-Based-Age-and-Gender-Recogniton">
</p>

<div align="center">

[![python](https://img.shields.io/badge/-Python_3.11.11-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![pytorch](https://img.shields.io/badge/Torch_2.0.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![cuda](https://img.shields.io/badge/-CUDA_11.8-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit-archive)
[![IEEE](https://img.shields.io/badge/-IEEE-blue?logo=ieee&logoColor=white)](https://doi.org/10.1109/ICTC62082.2024.10827387)


</div>

<p align="center">
<img src="https://img.shields.io/badge/Last%20updated%20on-26.06.2024-brightgreen?style=for-the-badge">
<img src="https://img.shields.io/badge/Written%20by-Nguyen%20Minh%20Nhut-pink?style=for-the-badge"> 
</p>

<div align="center">

[**Abstract**](#abstract) •
[**Usage**](#usage) •
[**References**](#references) •
[**Contact**](#contact)

</div>

## Abstract 
> In this study, we compared three architectures for the task of age and gender recognition from voice data: Long Short-Term Memory networks (LSTM), Hybrid of Convolutional Neural Networks Bidirectional Long Short-Term Memory (CNNs-BiLSTM), and the recently released RezoNet architecture. The dataset used in the study is sourced from Mozilla Common Voice in Japanese. Features such as pitch, magnitude, Mel-frequency cepstral coefficients (MFCCs), and filter-bank energies were extracted from the voice data for signal processing, and three architectures were evaluated. Our evaluation revealed that LSTM was slightly less accurate than RezoNet (83.1%), with hybrid CNNs-BiLSTM (93.1%) and LSTM achieving the best accuracy for gender recognition (93.5%). However, hybrid CNNs-BiLSTM architecture outperformed the other models in age recognition, with an accuracy of 69.75%, compared to 64.25% and 44.88% for LSTM and RezoNet, respectively. Using Japanese language data and the extracted characteristics, the hybrid CNNs-BiLSTM architecture model demonstrated the highest accuracy in both tests, highlighting its efficacy in voice-based age and gender detection. These results suggest promising avenues for future research and practical applications in this field.
>
> Index Terms: Voice-Based Age and Gender Recognition, RezoNet, Convolutional Neural Network, Long Short-Term Memory, Bidirectional Long-Term Memory, Deep Learning.

## Usage
### Dataset
In this study, we use voice dataset from Mozilla Comman Voice. 

<a href="https://commonvoice.mozilla.org/en/datasets">Download in here</a>
### Clone this repository
```python
git clone "https://github.com/nhut-ngnn/Voice-Based-Age-and-Gender-Recogniton.git"
```
### Create Conda Enviroment and Install Requirement
```python
conda create -n Voice-Based-Age-and-Gender-Recogniton python=3.10 -y
conda activate Voice-Based-Age-and-Gender-Recogniton
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
## References
```
@INPROCEEDINGS{nguyen2024age-gender,
  author={Nguyen, Nhut Minh and Nguyen, Thanh Trung and Nguyen, Hua Hiep and Tran, Phuong-Nam and Dang, Duc Ngoc Minh},
  booktitle={2024 15th International Conference on Information and Communication Technology Convergence (ICTC)}, 
  title={Voice-Based Age and Gender Recognition: A Comparative Study of LSTM, RezoNet and Hybrid CNNs-BiLSTM Architecture}, 
  year={2024},
  volume={},
  number={},
  pages={191-196},
  keywords={Long short term memory;Voice-Based Age and Gender Recognition;RezoNet;Convolutional Neural Network;Long Short-Term Memory;Bidirectional Long-Term Memory;Deep Learning},
  doi={10.1109/ICTC62082.2024.10827387}}
```
## Contact
For any information, please contact the main author:

Nhut Minh Nguyen at FPT University, Vietnam

Email: <link>minhnhut.ngnn@gmail.com </link>

GitHub: <link>https://github.com/nhut-ngnn</link>



