
<body>
<h1>Voice-Based Age, Gender, and Emotion Recognition: A Comparative Study of CNNs, LSTM, BiLSTM, and RezoNet Architecture</h1>

<h2>I. PROPOSED SOLUTION</h2>
<p>The proposed model for voice-based age, gender, and emotion recognition involves a comprehensive feature extraction process. Key features include Mel-Frequency Cepstral Coefficients (MFCC), Delta Mel-Frequency Cepstral Coefficients (delta-MFCC), Delta delta Mel-Frequency Cepstral Coefficients (delta delta-MFCC), Pitch, Filter-Bank Energies, Zero-Crossing Rate (ZCR), and ZCR Density. After that, we are using Principal component analysis (PCA) to reduce feature. In this research, we develop three separate classification models: one for age recognition, one for gender recognition, and one for emotion recognition. We utilize various architectures for classification and recognition, including Support Vector Machine (SVM), Long - Short Term Memory (LSTM), and Convolutional Neural Network (CNN)-based architectures such as DummyNet 1D, RezoNet, and ExpoNet.</p>

<h2>II. PERFORMANCE METRICS</h2>
<p>In this research, we utilize precision, recall, and F1-score as metrics for evaluating the performance of our classification models.</p>
<ul>
    <li><strong>Precision:</strong> Precision evaluates the percentage of accurately predicted positive instances among all instances identified as positive by the model. This metric is determined by dividing the number of true positives by the total of true positives and false positives. It is followed by Eq. 1.</li>
</ul>
<p>Precision =<br> True Positives / (True Positives + False Positives)</p>

<ul>
    <li><strong>Recall:</strong> Recall, known as sensitivity, evaluates the proportion of correctly predicted positive instances out of all true positive instances within the dataset. In Eq. 2, this metric is calculated by dividing the number of true positives by the sum of true positives and false negatives.</li>
</ul>
<p>Recall =<br> True Positives / (True Positives + False Negatives)</p>

<ul>
    <li><strong>F1-score:</strong> The F1-score, depicted in Eq. 3, represents the harmonic mean of precision and recall. This metric offers a balanced assessment by considering both precision and recall, rendering it especially valuable in scenarios where the dataset exhibits imbalance.</li>
</ul>
<p>F1-score = 2 × (Precision × Recall) / (Precision + Recall)</p>

<ul>
    <li><strong>Macro average:</strong> Macro average is used to describe a specific type of averaging technique utilized in multi-class classification tasks. It is commonly applied to compute performance metrics like precision in Eq. 4, recall in Eq. 6, or F1 score in Eq. 5, particularly in situations involving imbalanced datasets. In the case of the Macro average, the metric is computed individually for each class, and subsequently, the average is calculated across all classes.</li>
</ul>
<p>Macro Precision = (Precision of Class 1 + Precision of Class 2 + ... + Precision of Class N) / N</p>
<p>Macro F1-score = 2 × (Macro Precision × Macro Recall) / (Macro Precision + Macro Recall)</p>
<p>Macro Recall = (Recall of Class 1 + Recall of Class 2 + ... + Recall of Class N) / N</p>

<ul>
    <li><strong>Weighted average:</strong> The weighted precision, recall, and F1-score functions are essential metrics in evaluating the performance of multi-class classification models. They incorporate weights assigned to each class to calculate a weighted average, reflecting the significance or contribution of individual classes to overall performance. Weighted precision assesses the accuracy of positive predictions, weighted recall measures the completeness of positive predictions, precision in Eq. 7 and balances recall in Eq. 8 and weighted F1-score is presented in Eq. 9, providing a comprehensive evaluation of model effectiveness while considering class-specific importance.</li>
</ul>
<p>Weighted Precision = (p1 × w1 + p2 × w2 + ... + pN × wN) / Total Weight<br> where pi is the precision of class i and wN is the weight of class i.</p>
<p>Weighted Recall = (r1 × w1 + r2 × w2 + ... + rN × wN) / Total Weight<br> where rN is the recall of class i and wN is the weight of class i.</p>
<p>Weighted F1-Score = (f1 × w1 + f2 × w2 + ... + fN × wN) / Total Weight<br> where fN is the F1-Score of class i and wN is the weight of class i.</p>

<h2>III. DATASET</h2>
<p>For Age and Gender model, we use dataset from Common Voice Mozilla [1] for Age and Gender Classification Model, and RAVDESS dataset [2], CREMA-D dataset [3], TESS dataset [4], SAVEE dataset [5], distribute in Table V, for Emotion Classification Model. The age, gender, and sentiment distributions in the dataset are shown in the tables II, III, IV.</p>

<table>
    <tr>
        <th>Class</th>
        <th>Percent</th>
    </tr>
    <tr>
        <td>0 - 19</td>
        <td>6%</td>
    </tr>
    <tr>
        <td>20 - 29</td>
        <td>25%</td>
    </tr>
    <tr>
        <td>30 - 39</td>
        <td>14%</td>
    </tr>
    <tr>
        <td>40 - 49</td>
        <td>9%</td>
    </tr>
    <tr>
        <td>50 - 59</td>
        <td>5%</td>
    </tr>
    <tr>
        <td>60 - 69</td>
        <td>4%</td>
    </tr>
    <tr>
        <td>70 - 79</td>
        <td>1%</td>
    </tr>
    <tr>
        <td>No information</td>
        <td>36%</td>
    </tr>
</table>
<p>COMMON VOICE CORPUS 17.0, AGE ANALYSIS</p>

<table>
    <tr>
        <th>Class</th>
        <th>Percent</th>
    </tr>
    <tr>
        <td>Female</td>
        <td>19%</td>
    </tr>
    <tr>
        <td>Male</td>
        <td>53%</td>
    </tr>
    <tr>
        <td>No information</td>
        <td>27%</td>
    </tr>
</table>
<p>COMMON VOICE CORPUS 17.0, GENDER ANALYSIS</p>

<table>
    <tr>
        <th>Class</th>
        <th>Percent</th>
    </tr>
    <tr>
        <td>Angry</td>
        <td>16.7%</td>
    </tr>
    <tr>
        <td>Happy</td>
        <td>16.46%</td>
    </tr>
    <tr>
        <td>Sad</td>
        <td>16.35%</td>
    </tr>
    <tr>
        <td>Neutral</td>
        <td>14.26%</td>
    </tr>
    <tr>
        <td>Fearful</td>
        <td>16.46%</td>
    </tr>
    <tr>
        <td>Disgusted</td>
        <td>15.03%</td>
    </tr>
    <tr>
        <td>Surprised</td>
        <td>4.74%</td>
    </tr>
</table>
<p>EMOTION ANALYSIS</p>

<table>
    <tr>
        <th>Dataset Name</th>
        <th>Percent</th>
    </tr>
    <tr>
        <td>CREMA-D</td>
        <td>58.15%</td>
    </tr>
    <tr>
        <td>TESS</td>
        <td>21.88%</td>
    </tr>
    <tr>
        <td>RAVDESS</td>
        <td>16.22%</td>
    </tr>
    <tr>
        <td>SAVEE</td>
        <td>3.75%</td>
    </tr>
</table>
<p>EMOTION DATASET ANALYSIS</p>

<h2>IV. RELATED WORK</h2>

<h3>A. Paper</h3>
<p>Noushin Hajarolasvadi et al. [6] propose a new system to recognize emotions from speech. It works by first breaking down speech recordings into short segments and extracting features like pitch and intensity. Then, it uses a clustering technique to identify the most important segments and creates a 3D representation based on those segments. Finally, a special kind of neural network analyzes these 3D representations to identify emotions. Experiments show this system performs better than previous methods.</p>

<p>Dr. Madhu M. Nashipudimath et al. [7] tackle recognizing both feelings and a person’s sex just by listening to their voice. Men and women naturally have different vocal qualities, and these along with various emotions can be identified through computer analysis. To achieve this, the system first cleanses the speech recording and extracts key characteristics like MFCCs. Then it utilizes a special kind of classifier to categorize the emotions and gender. This approach is reported to be more effective than past methods, making it potentially useful in healthcare, virtual assistants, security systems and other areas where machines interact with people.</p>

<p>Nammous, Mohammad K. and et al. [8] explore using LSTMs (a powerful type of neural network) for speaker recognition, specifically focusing on situations with limited training data. Traditionally, a lot of training data is needed for good results in speaker recognition. This paper shows that LSTMs can achieve high accuracy even with a smaller training set. The paper compares LSTMs to a different kind of neural network (feed-forward) and finds LSTMs perform better for speaker recognition, gender identification, and language identification – even when training data is limited.</p>

<h3>B. Code</h3>
<p>Yousef Kotp et al.’s notebook [9] is all about experimenting different set of features for audio representation and different CNN-based architectures for building speech emotion recognition model. The notebook contains implementation for three new CNN-based architectures for speech emotion recognition. which are: DummyNet 1D, RezoNet, ExpoNet.</p>

<p>Voice gender recognition from SuperKogito [10] can be achieved by analyzing speech patterns. Here, researchers use a dataset of speakers with known genders. They then extract Mel-frequency cepstral coefficients (MFCCs) which convert voice samples into a format that highlights speaker characteristics. Next, they train Gaussian mixture models (GMMs) to represent the male and female speaker distributions based on these MFCCs. Finally, when analyzing an unknown voice sample, they can compare its MFCCs to the trained GMMs and predict the speaker’s gender based on which model shows a better fit. The system results in a 95% accuracy of gender detection.</p>

<h3>REFERENCES</h3>
<ol>
    <li><a href="https://commonvoice.mozilla.org">Mozilla voice dataset</a>.</li>
    <li>S. R. Livingstone and F. A. Russo, “The ryerson audio-visual database of emotional speech and song (ravdess): A dynamic, multimodal set of facial and vocal expressions in north american english,” PloS one, vol. 13, no. 5, p. e0196391, 2018.</li>
    <li>H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova, and R. Verma, “Crema-d: Crowd-sourced emotional multimodal actors dataset,” IEEE transactions on affective computing, vol. 5, no. 4, pp. 377–390, 2014.</li>
    <li>P. Jackson and S. Haq, “Surrey audio-visual expressed emotion (savee) database,” University of Surrey: Guildford, UK, 2014.</li>
    <li>K. Dupuis and M. K. Pichora-Fuller, Toronto emotional speech set (TESS). University of Toronto, Psychology Department, 2010.</li>
    <li>N. Hajarolasvadi and H. Demirel, “3d cnn-based speech emotion recognition using k-means clustering and spectrograms,” Entropy, vol. 21, no. 5, p. 479, 2019.</li>
    <li>M. M. Nashipudimath, P. Pillai, A. Subramanian, V. Nair, and S. Khalife, “Voice feature extraction for gender and emotion recognition,” in ITM Web of Conferences, vol. 40. EDP Sciences, 2021, p. 03008.</li>
    <li>M. K. Nammous and K. Saeed, “Natural language processing: Speaker, language, and gender identification with lstm,” Advanced Computing and Systems for Security: Volume Eight, pp. 143–156, 2019.</li>
    <li><a href="https://github.com/yousefkotp/Speech-Emotion-Recognition">Speech emotion recognition</a>.</li>
    <li><a href="https://github.com/SuperKogito/Voice-based-gender-recognition">Voice based gender recognition</a>.</li>
</ol>

</body>
