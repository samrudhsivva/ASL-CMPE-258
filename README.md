# ASL-CMPE-258

# Proposal: Deep Learning-Powered ASL Recognition with Multilingual and Adaptive Feedback Features

## Team Members:

- **Name:** Samrudh Sivva\
  **Email:** [samrudh.sivva@sjsu.edu](mailto:samrudh.sivva@sjsu.edu)\
  **SJSU ID:** 017520659
- **Name:** Sai Prasad Shivaratri\
  **Email:** [saiprasad.shivanatri@sjsu.edu](mailto:saiprasad.shivanatri@sjsu.edu)\
  **SJSU ID:** 017507191
- **Name:** Nithin Aleti\
  **Email:** [nithin.aleti@sjsu.edu](mailto:nithin.aleti@sjsu.edu)\
  **SJSU ID:** 017401930

## Proposed Application Area:

This proposal focuses on developing a real-time American Sign Language (ASL) gesture recognition system utilizing advanced deep learning techniques. By converting ASL gestures into textual language, this project aims to serve as a communication aid for individuals with hearing and speech impairments. The updated application allows users to record a sequence of gestures corresponding to characters, store the input, recognize the full word, and provide translations into more than five languages.

### Goals:

1. **Enhance Accuracy:** Boost recognition accuracy of ASL gestures using the **EfficientNet-B0** deep learning model.
2. **User-Friendly Interface:** Provide a React-based UI for seamless interaction.
3. **Inclusivity:** Ensure the solution is accessible for diverse user groups by offering accurate real-time feedback and translations in over five languages (e.g., English, Spanish, Hindi, French, and German).
4. **Cost Efficiency:** Leverage efficient architecture and open-source frameworks to reduce computational costs.
5. **Feedback Loop:** Accept user feedback on unrecognized gestures to iteratively improve the model.
6. **Word Recognition and Translation:** Recognize complete words from recorded gesture sequences and translate them into multiple languages.

## Identified Dataset and Features:

### Dataset Source:

- **ASL Alphabet Dataset** from Kaggle: Over 87,000 static images representing 26 ASL gestures.

### Features:

- **Classes:** 26 static gestures (A-Z).
- **Size:** Over 87,000 high-resolution images.
- **Distribution:** Balanced dataset for robust training.
- **Format:** Images organized in nested class folders for straightforward preprocessing.

## Proposed Architecture:

The system integrates the following key components:

1. **Dataset Preparation:** Data augmentation and preprocessing for enhanced generalization.
2. **Model Training:** Fine-tuned **EfficientNet-B0** model for high performance.
3. **Backend Deployment:** REST API for model inference and translation.
4. **Frontend Interface:** React-based UI for gesture display, word recognition, multilingual translation, and user feedback.

### Workflow:

- **Dataset Preparation:**

  - Split dataset into 80% training and 20% validation sets.
  - Apply preprocessing techniques like resizing to 224x224, normalization, and augmentation (e.g., random flips, color jitter).

- **Model Training:**

  - Replace the classifier layer of **EfficientNet-B0** with a custom head for 26 classes.
  - Utilize pre-trained ImageNet weights for transfer learning.
  - Optimize training with **AdamW optimizer** and **CrossEntropyLoss**.

- **Word Recognition and Translation:**

  - Record ASL gestures sequentially to construct words.
  - Use a backend service to recognize the word from the sequence.
  - Translate the recognized word into over five languages using a multilingual translation API (e.g., Google Translate API).

- **Inference and UI Integration:**

  - Real-time predictions served via a REST API.
  - React-based UI displays recognized gestures, constructs words, and provides multilingual text-to-speech feedback.
  - Feedback loop to allow users to submit corrections for unrecognized gestures or words.

## Model Selection:

**EfficientNet-B0** is proposed for its optimal balance of accuracy and computational efficiency:

- **Baseline CNN:** Achieved lower accuracy (~91%).
- **ResNet-18:** Provided better accuracy (~96%) but required more computational resources.
- **EfficientNet-B0:** Achieved ~99.6% accuracy with reduced resource consumption, making it ideal for real-time applications.

### Code for Model Selection:

```python
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

# Load pre-trained EfficientNetB0 model
base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
base_model.trainable = False

# Build the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(26, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Summary of the model
model.summary()
```

## Training Strategy:

- **Loss Function:** CrossEntropyLoss for multi-class classification.
- **Batch Size:** 16 (optimized for memory constraints).
- **Epochs:** Proposed training for 5 epochs.
- **Augmentation Techniques:** Resizing, normalization, random flips, and color jitter to ensure robust generalization.

## Evaluation Plan:

- **Evaluation Metrics:**
  - Accuracy: Targeting ~99.6% on validation data.
  - Precision, Recall, and F1-Score: To assess performance across all 26 gesture classes.
- **Confusion Matrix:** Identify potential misclassifications and iteratively improve model performance.

## Implementation Plan:

1. **Dataset Preparation:**

   - Preprocess the ASL Alphabet Dataset and split into training and validation sets.
   - Apply augmentations to improve model robustness.

2. **Model Training:**

   - Train **EfficientNet-B0** on preprocessed data.
   - Fine-tune hyperparameters using cross-validation techniques.

3. **Word Recognition Module:**

   - Develop a gesture sequence recognition algorithm to identify complete words.
   - Integrate with multilingual translation APIs for word translation into over five languages.

4. **Backend Development:**

   - Deploy the trained model via a REST API for both single gestures and word recognition.

5. **Frontend Integration:**

   - Develop a React-based interface to display predictions and translations in real-time.
   - Enable text-to-speech feedback in multiple languages.
   - Implement a feedback loop for user corrections.

### React Code for Frontend Integration:

```jsx
import React, { useState } from 'react';
import axios from 'axios';

const ASLRecognition = () => {
  const [gesture, setGesture] = useState('');
  const [translation, setTranslation] = useState('');

  const handleRecognition = async () => {
    try {
      const response = await axios.post('/api/recognize', { image: "sampleImage" });
      setGesture(response.data.gesture);
      const translationResponse = await axios.post('/api/translate', { word: response.data.gesture });
      setTranslation(translationResponse.data.translation);
    } catch (error) {
      console.error('Error recognizing or translating gesture:', error);
    }
  };

  return (
    <div>
      <h1>ASL Recognition System</h1>
      <button onClick={handleRecognition}>Recognize Gesture</button>
      <p>Recognized Gesture: {gesture}</p>
      <p>Translation: {translation}</p>
    </div>
  );
};

export default ASLRecognition;
```

6. **Testing:**

   - Evaluate the system under varying lighting and background conditions to ensure reliability.

## Outcomes:

1. Achieve high accuracy (~99.6%) in recognizing ASL gestures.
2. Deliver a seamless real-time experience through the React-based UI.
3. Provide multilingual feedback for complete words to enhance inclusivity.
4. Establish a feedback mechanism for continuous model improvement.
5. Ensure cost-efficient operation leveraging EfficientNet-B0 and open-source tools.

## Task Distribution:

| Task                                         | Description                                                                                         | Team Member           |
|----------------------------------------------|-----------------------------------------------------------------------------------------------------|-----------------------|
| Dataset Preparation                          | Preprocessing and augmentation of the ASL Alphabet dataset for model training.                     | Samrudh Sivva         |
| Model Training and Optimization              | Training the EfficientNet-B0 model, tuning hyperparameters, and evaluating performance.             | Sai Prasad Shivaratri |
| Backend Development                          | Developing REST APIs for gesture recognition and translation services.                              | Nithin Aleti          |
| Word Recognition and Translation Integration | Implementing the word recognition algorithm and integrating with multilingual translation APIs.      | Sai Prasad Shivaratri |
| Frontend Interface Development               | Creating a React-based interface to display gestures, recognized words, and translations in real-time.| Samrudh Sivva       |
| Testing and Evaluation                       | Conducting system reliability tests under different conditions and iteratively improving performance. | Nithin Aleti          |
| GitHub Repository Setup and Management       | Setting up, maintaining, and documenting the project repository for seamless team collaboration.     | All Members           |
| Team Collaboration and Progress Tracking     | Organizing team meetings, tracking milestones, and ensuring timely completion of tasks.             | All Members           |

## UI Output:

## Comparison:

## Future Enhancements:

- Expand to include dynamic gesture recognition using video sequences.
- Optimize for cloud-based scalability and mobile deployment.
- Introduce additional features like translation support for more than 10 languages.

## References:

- Kaggle ASL Alphabet Dataset: [https://www.kaggle.com/grassknoted/asl-alphabet](https://www.kaggle.com/grassknoted/asl-alphabet)
- EfficientNet: [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)
- React Documentation: [https://reactjs.org/](https://reactjs.org/)
- TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)

