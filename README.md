# Voice-Based Age and Gender recognize: A Comparative Study of LSTM, RezoNet, and Hybrid CNNs-BiLSTM Architecture</h1>

> Please press ⭐ button and/or cite papers if you feel helpful.

<p align="center">
<img src="https://img.shields.io/badge/Last%20updated%20on-26.06.2024-brightgreen?style=for-the-badge">
<img src="https://img.shields.io/badge/Written%20by-Nguyen%20Minh%20Nhut-pink?style=for-the-badge"> 
</p>


<p align="center">
<img src="https://img.shields.io/badge/Long%20Short%20Term%20Memory-white"> 
<img src="https://img.shields.io/badge/Bidirectional%20Long%20Short%20Term%20Memory-white">   
<img src="https://img.shields.io/badge/RezoNet-white">     
<img src="https://img.shields.io/badge/Hybrid%20CNN_BiLSTM-white">
</p>

## Abstract 
> In this study, we compared three architectures for the task of age and gender recognition from voice data: Long Short-Term Memory networks (LSTM), Hybrid of Convolutional Neural Networks Bidirectional Long Short-Term Memory (CNNs-BiLSTM), and the recently released RezoNet architecture. The dataset used in the study is sourced from Mozilla Common Voice in Japanese. Features such as pitch, magnitude, Mel-frequency cepstral coefficients (MFCCs), and filter-bank energies were extracted from the voice data for signal processing, and three architectures were evaluated. Our evaluation revealed that LSTM was slightly less accurate than RezoNet (83.1%), with hybrid CNNs-BiLSTM (93.1%) and LSTM achieving the best accuracy for gender recognition (93.5%). However, hybrid CNNs-BiLSTM architecture outperformed the other models in age recognition, with an accuracy of 69.75%, compared to 64.25% and 44.88% for LSTM and RezoNet, respectively. Using Japanese language data and the extracted characteristics, the hybrid CNNs-BiLSTM architecture model demonstrated the highest accuracy in both tests, highlighting its efficacy in voice-based age and gender detection. These results suggest promising avenues for future research and practical applications in this field.
> Index Terms: Voice-Based Age and Gender Recognition, RezoNet, Convolutional Neural Network, Long Short-Term Memory, Bidirectional Long-Term Memory, Deep Learning.
## Table of Contents

- [Abstract](#Abstract)
- [Usage](#Usage)
  - [Dataset](#dataset)
  - [Clone this repository](#clone-this-repository)
  - [Create Conda Enviroment and Install Requirement](#create-conda-enviroment-and-install-requirement)
  - [Extract feature](#extract-feature)
  - [Training process](#training-process)
  - [Testing process](#testing-process)
- [Contact](#Contact)
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
### Extract feature 
### Training process
### Testing process
## Contact
For any information, please contact the main author:

Nhut Minh Nguyen at FPT University, Vietnam

Email: <link>minhnhut.ngnn@gmail.com </link>

GitHub: <link>https://github.com/nhut-ngnn</link>



