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

This proposal focuses on developing a real-time American Sign Language (ASL) gesture recognition system utilizing advanced deep learning techniques. By converting ASL gestures into textual language, this project aims to serve as a communication aid for individuals with hearing and speech impairments. The system will process static images of hand gestures to identify ASL signs accurately and efficiently.

### Goals:

1. **Enhance Accuracy:** Boost recognition accuracy of ASL gestures using the **EfficientNet-B0** deep learning model.
2. **User-Friendly Interface:** Provide a React-based UI for seamless interaction.
3. **Inclusivity:** Ensure the solution is accessible for diverse user groups by offering accurate real-time feedback in multiple languages (English, Spanish, Hindi).
4. **Cost Efficiency:** Leverage efficient architecture and open-source frameworks to reduce computational costs.
5. **Feedback Loop:** Accept user feedback on unrecognized gestures to iteratively improve the model.

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
3. **Backend Deployment:** REST API for model inference.
4. **Frontend Interface:** React-based UI for gesture display, multilingual support, and user feedback.

### Workflow:

- **Dataset Preparation:**
  - Split dataset into 80% training and 20% validation sets.
  - Apply preprocessing techniques like resizing to 224x224, normalization, and augmentation (e.g., random flips, color jitter).

- **Model Training:**
  - Replace the classifier layer of **EfficientNet-B0** with a custom head for 26 classes.
  - Utilize pre-trained ImageNet weights for transfer learning.
  - Optimize training with **AdamW optimizer** and **CrossEntropyLoss**.

- **Inference and UI Integration:**
  - Real-time predictions served via a REST API.
  - React-based UI displays recognized gestures in real-time with multilingual text-to-speech feedback (English, Spanish, Hindi).
  - Feedback loop to allow users to submit corrections for unrecognized gestures.

## Model Selection:

**EfficientNet-B0** is proposed for its optimal balance of accuracy and computational efficiency:
- **Baseline CNN:** Achieved lower accuracy (~91%).
- **ResNet-18:** Provided better accuracy (~96%) but required more computational resources.
- **EfficientNet-B0:** Achieved ~99.6% accuracy with reduced resource consumption, making it ideal for real-time applications.

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

3. **Backend Development:**
   - Deploy the trained model via a REST API.

4. **Frontend Integration:**
   - Develop a React-based interface to display predictions in real-time.
   - Integrate multilingual text-to-speech feedback (English, Spanish, Hindi).
   - Implement a feedback loop for user corrections.

5. **Testing:**
   - Evaluate the system under varying lighting and background conditions to ensure reliability.

## Outcomes:

1. Achieve high accuracy (~99.6%) in recognizing ASL gestures.
2. Deliver a seamless real-time experience through the React-based UI.
3. Provide multilingual feedback to enhance inclusivity.
4. Establish a feedback mechanism for continuous model improvement.
5. Ensure cost-efficient operation leveraging EfficientNet-B0 and open-source tools.

<img width="544" alt="Screenshot 2024-12-15 at 8 53 11 PM" src="https://github.com/user-attachments/assets/326e8459-7710-4ac3-aca0-02445dc938e9" />

## Future Enhancements:

- Expand to include dynamic gesture recognition using video sequences.
- Optimize for cloud-based scalability and mobile deployment.
- Introduce additional features like multi-language support for more languages.

## References:

- **ASL Alphabet Dataset:** Kaggle
- **EfficientNet-B0 Architecture:** TensorFlow and PyTorch GitHub repositories
- **React.js:** Used for building the frontend interface.






