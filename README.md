# Speech Emotion Recognition
This project explores the use of deep learning in speech emotion recognition.
Speech emotion recognition serves many valuable purposes, such as ensuring the safety of individuals by checking their sobriety and capability to drive, as well as assessing mental health to determine a person's readiness for specific tasks. It can be useful when evaluating the performance of client interactions to improve client support calls and save clients from lengthy questionairies. 

This could lead to a more dynamic feedback to learn some best practices, identify the level of satisfaction and make clients free of lengthy questionairies.

With this purpose, the dataset used in this project was the RAVDESS dataset that can be found <[here](https://zenodo.org/record/1188976)>.
The RAVDESS is one of the most used and most complete databases for emotion recognition. There are 60 audios of 24 different actors (male and female) expressing 8 different emotions: calm, happy, angry, disgust, neutral, sad, fearful and surprised.

Some characteristics of the data set, as the length and the amplitude (loudness) of the audio were shown and the need to manipulate this when modelling were discussed and later adressed. The dataset extracted was unbalanced, neutral represented 6.6% and other emotions represented 13.3% each. This can be overcome later with data augmentation, such as adding noise, shitfing and stretching the audio in time. This was not explored in the project, and the baseline was considered to be 13.3%.

## Data Dictionary
||Type|Dataset|Description|
|---|---|---|---|
|file|object|files_details|file path| 
|length|int|files_details|Length of the audio| 
|seconds|float|files_details|Seconds of the audio| 
|sample_rate|int|files_details|Sample rate of audio| 
|emotions|object|files_details|Emotion of the audio| 
|label|int|files_details|Label of the audio| 
|len_trim|int|files_details|Length of the audio - trimmed| 
|max_amp|float|files_details|Maximum Amplitude of file| 

## Preprocess - Librosa Library
Librosa was used to load the files and preprocess them to create the input of the deep learning models: the melspectrograms. This can be considered as the image of the file and it can serve as an input to convolutional neural networks. Librosa first loads the audio as numpy array with its sample rate. Sample rate is the number of measurements per second the audio has (every file in this dataset had 22050).
- 1: Setting a threshold of 30db below reference to be considered as silence. This was done because not all audio is relevant content to be considered, this can lead to smaller inputs and faster models without losing information.
- 2: Setting the numpy array to a size of 88200 values, which corresponds to 4s of audio. This is more than enough to extract the information for speech recognition purposes in this project. Longer audios were trimmed and shorter ones were padded with zeros (silence).
- 3: The signal was transformed to a melspectrogram. Sounds are composed by different frequencies with different amplitudes along time. Spectrograms show the strength of the frequencis along time. Melspectrograms take into account how humans perceive frequencies.
- 4: This strength of frequency is represented by the amplitude (loudness) and it was converted to dB (decibel). The decibel scale take into account how humans perceive loudness.

The melspectrograms were the input of a convolutional neural network. The final model had 2 convolutional 2D layers and some layers to avoid overfitting.
It reached an accuracy of 60% on validation data, a great improvement from the baseline 13.3%.
Further, this can be improved with the incoporation of more datasets, with different actors, statements and emotions and data augmentation to make the model more robust to changes of the sound and of the spectrograms. It is worth noting that speech emotion recognition is very subjective, people express and sense emotions differently and there are many more than 8. However, this model can be used to assess how positive the client call experience was and improve their interactions.
