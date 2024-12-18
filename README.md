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
3. **Backend Deployment:**  REST API for model inference and translation using Python Flask.
4. **Frontend Interface:** React-based UI for gesture display, word recognition, multilingual translation, and user feedback.


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

   - Deploy the trained model via a REST API for both single gestures and word recognition using the provided FastAPI code.

5. **Frontend Integration:**

   - Develop a React-based interface to display predictions and translations in real-time.
   - Enable text-to-speech feedback in multiple languages.
   - Implement a feedback loop for user corrections.

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
| Dataset Preparation                          | Preprocessing and augmentation of the ASL Alphabet dataset for model training.                     | Sai Prasad         |
| Model Training and Optimization              | Training the EfficientNet-B0 model, tuning hyperparameters, and evaluating performance.             |  Samrudh Sivva |
| Backend Development                          | Developing REST APIs for gesture recognition and translation services using FastAPI.                | Nithin Aleti          |
| Word Recognition and Translation Integration | Implementing the word recognition algorithm and integrating with multilingual translation APIs.      | Sai Prasad Shivaratri |
| Frontend Interface Development               | Creating a React-based interface to display gestures, recognized words, and translations in real-time.| Samrudh Sivva       |
| Testing and Evaluation                       | Conducting system reliability tests under different conditions and iteratively improving performance. | Nithin Aleti          |
| GitHub Repository Setup and Management       | Setting up, maintaining, and documenting the project repository for seamless team collaboration.     | Samrudh Sivva           |
| Team Collaboration and Progress Tracking     | Organizing team meetings, tracking milestones, and ensuring timely completion of tasks.             | All Members           |

## UI Output:
<img width="1470" alt="Screenshot 2024-12-18 at 12 32 34 AM" src="https://github.com/user-attachments/assets/59757201-d6b4-4837-83da-f5b3103f644a" />


<img width="1470" alt="Screenshot 2024-12-18 at 12 29 55 AM" src="https://github.com/user-attachments/assets/348dd027-c65e-4c2d-87e7-7c52dd5bfdf6" />


## Comparison:

![WhatsApp Image 2024-12-18 at 00 24 57](https://github.com/user-attachments/assets/a0d33658-764d-4a41-abbc-bc9562fa1583)


<img width="774" alt="Screenshot 2024-12-18 at 12 21 55 AM" src="https://github.com/user-attachments/assets/96405f2f-22eb-4d77-94de-a4b2eb92fbc4" />



## Future Enhancements:

- Expand to include dynamic gesture recognition using video sequences.
- Improve the Feedback Component
- Optimize for cloud-based scalability and mobile deployment.
- Introduce additional features like translation support for more than 10 languages.

## References:

References:

Kaggle ASL Alphabet Dataset: https://www.kaggle.com/grassknoted/asl-alphabet

EfficientNet: https://arxiv.org/abs/1905.11946

React Documentation: https://reactjs.org/

TensorFlow Documentation: https://www.tensorflow.org/

Flask Documentation: https://flask.palletsprojects.com/

PyTorch Documentation: https://pytorch.org/docs/stable/index.html

Google Translate API: https://cloud.google.com/translate/docs

pyttsx3 Documentation: https://pyttsx3.readthedocs.io/en/latest/

FastAPI Documentation: https://fastapi.tiangolo.com/

