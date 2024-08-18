# Speech Emotion Recognition
## Project Overview
Audio classification, particularly in the context of speech emotion recognition, plays a critical role in various real-world applications. From enhancing client interactions in call centers to assessing an individual's capability to perform crucial tasks such as driving, the ability to accurately detect and interpret emotions or mental state from speech can have significant implications. For instance, detecting stress or fatigue in a driver’s voice could trigger alerts to prevent accidents, while understanding customer emotions can help tailor responses in service industries, leading to improved satisfaction.

## Dataset
This project explores the application of deep learning techniques to speech emotion recognition, aiming to develop a model capable of identifying emotions based on a person's voice. The model is trained using the [RAVDESS dataset](https://zenodo.org/record/1188976), which includes 1,440 audio recordings from 24 actors, each contributing 60 trials across eight distinct emotions: neutral, calm, happy, sad, angry, fearful, disgust, and surprised. The emotions have two intensities, with the exception of 'neutral,' which is presented with a single intensity.

## Audio Analysis with Librosa
Librosa, a popular Python library for audio analysis, is utilized in this project to load and process the audio files. Key audio characteristics such as length and loudness are extracted to be further selected to analyze manipulations in the audios. Librosa processes audio files as numpy time series arrays, with absolute values representing the loudness of the audio, and their respective sample rates. Sample rate is the frequency data is captured, the audios of the dataset presented values per second. Audios had durations ranging from 2.9 to 5.3 seconds.

- Trimming Silence: Librosa's trim function is used to remove silence from the recordings, applying different thresholds to determine the optimal balance between data reduction and information retention.

- Downsampling: The sample rate is reduced from 48,000 Hz to 22,500 Hz to decrease data size without compromising audio quality significantly.

- Cutting and Padding: To standardize the dataset for model training, the audio files are adjusted to a uniform length by truncating longer files and padding shorter ones with zeros.

- Mel Spectrogram Conversion: The preprocessed audio time series are converted into mel spectrograms, which visually represent frequency patterns over time. These 2D representations serve as input for the neural network model.

# Model Development
A Convolutional Neural Network (CNN) is chosen for this task due to its efficacy in processing image-like data, such as mel spectrograms. Two versions of the model are created: one with bias initialization and one without.

## Training and Validation
Initial Results: Both models showed similar performance in the first epoch, achieving over 90% accuracy on the training and validation sets. The model without bias initialization slightly outperformed the other in terms of validation loss.

## Final Model Selection
Based on the validation performance, the model without bias initialization was selected for further training on the entire dataset.

## Model Performance
The model performed well on the training and validation sets, correctly predicting emotions with over 90% accuracy. However, it struggled with unseen data, particularly misclassifying emotions such as anger and sadness. This indicates that while the model effectively learns from the provided dataset, it has difficulty generalizing to new, diverse inputs.

## Challenges and Observations
- Ambiguity in Emotion Recognition: Some audio samples presented challenges, as emotions like "sad" and "calm" were difficult to distinguish. This reflects the complexity of real-world emotion recognition, where expressions are often subtle and context-dependent.

- Dataset Limitations: The dataset consists of scripted phrases, which may not capture the full range of emotional expression found in natural speech. This limitation could affect the model's performance in real-world applications.

## Future Directions
To improve the model's robustness and generalization:

- Expand the Dataset: Training on a larger and more diverse dataset, including various voices and unscripted statements, could enhance the model's ability to handle real-world scenarios.

- Data Augmentation: Implementing data augmentation techniques, such as pitch shifting or time-stretching, could help the model better generalize to different speech variations.

## Conclusion
This project demonstrates the potential of deep learning for speech emotion recognition, achieving high accuracy on the training and validation sets. However, the challenges with unseen data highlight the need for further research and development to create a model that can reliably perform in diverse, real-world applications.
