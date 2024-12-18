import React, { useState } from "react";
import { useSpeechSynthesis } from "react-speech-kit";
import "./App.css";

const App = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [predictedAlphabet, setPredictedAlphabet] = useState("");
  const [language, setLanguage] = useState("en"); // Default: English
  const [recordedWord, setRecordedWord] = useState(""); // Word being formed
  const [translatedWord, setTranslatedWord] = useState(""); // Translated word
  const { speak } = useSpeechSynthesis();

  // Handles image upload and triggers prediction by sending the image to the backend
const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      handlePredict(file);
    }
  };

  // Sends the uploaded image to the backend and retrieves the predicted alphabet
const handlePredict = async (image) => {
    try {
      const formData = new FormData();
      formData.append("image", image);

      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Prediction failed");
      }

      const data = await response.json();
      setPredictedAlphabet(data.predictedAlphabet);
      callOutAlphabet(data.predictedAlphabet);
    } catch (error) {
      console.error("Error predicting alphabet:", error);
    }
  };

  // Uses the speech synthesis API to vocalize the predicted alphabet in the selected language
const callOutAlphabet = (text) => {
    const voices = window.speechSynthesis.getVoices();
    const languageMap = {
      en: "en-US",
      es: "es-ES",
      te: "te-IN", // Telugu (requires specific system setup for support)
      hi: "hi-IN", // Hindi
    };

    const voice = voices.find((v) => v.lang === languageMap[language]);
    speak({
      text,
      voice,
      lang: languageMap[language],
    });
  };

  const handleLanguageChange = (event) => {
    setLanguage(event.target.value);
  };

  // Appends the predicted alphabet to the recorded word
const handleRecordIn = () => {
    setRecordedWord((prevWord) => prevWord + predictedAlphabet);
  };

  // Translates the recorded word into the selected language and vocalizes the translation
const handleRecordOut = async () => {
    if (recordedWord) {
      const translated = await translateWord(recordedWord, language);
      setTranslatedWord(translated);
      alert(`Recorded Word: ${recordedWord}\nTranslated Word: ${translated}`);
      callOutAlphabet(translated);
    }
  };

  // Resets the recorded and translated words
const handleReset = () => {
    setRecordedWord(""); // Clear the recorded word
    setTranslatedWord(""); // Clear the translated word
  };

  // Sends the recorded word to the backend for translation into the selected language
const translateWord = async (word, targetLang) => {
    try {
      const response = await fetch("http://127.0.0.1:5000/translate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: word,
          target: targetLang,
        }),
      });

      if (!response.ok) {
        throw new Error("Translation failed");
      }

      const data = await response.json();
      return data.translatedText;
    } catch (error) {
      console.error("Error translating word:", error);
      return word; // Fallback to original word if translation fails
    }
  };

  const targetLangMap = {
    en: "English",
    es: "Spanish",
    hi: "Hindi",
    te: "Telugu",
    ja: "Japanese",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    fr: "French",
    de: "German",
  };

  return (
    <div className="app-container">
      <h1 className="title">Alphabet Predictor</h1>
      <div className="team-details">
        <p>CMPE 258 - Team 16</p>
        <p>Samrudh Sivva</p>
        <p>Sai Prasad Shivanatri</p>
        <p>Nithin Aleti</p>
      </div>
      <input
        type="file"
        accept="image/*"
        onChange={handleImageUpload}
        className="upload-button"
      />
      {selectedImage && (
        <img
          src={URL.createObjectURL(selectedImage)}
          alt="Uploaded"
          className="uploaded-image"
        />
      )}
      <div className="button-container">
        <button
          onClick={handleRecordIn}
          className="record-in-button"
          disabled={!predictedAlphabet}
        >
          Record In
        </button>
        <button onClick={handleRecordOut} className="record-out-button">
          Record Out
        </button>
        <button onClick={handleReset} className="reset-button">
          Reset
        </button>
        <button
          onClick={() => alert("Feedback button pressed!")}
          className="feedback-button"
        >
          Feedback
        </button>
      </div>
      <h2 className="predicted-alphabet">
        Predicted Alphabet: {predictedAlphabet}
      </h2>
      <h2 className="recorded-word">Recorded Word: {recordedWord}</h2>
      <h2 className="translated-word">Translated Word: {translatedWord}</h2>
      <label className="language-selector">
        Language:
        <select value={language} onChange={handleLanguageChange}>
          <option value="en">English</option>
          <option value="es">Spanish</option>
          <option value="hi">Hindi</option>
          <option value="te">Telugu</option>
          <option value="ja">Japanese</option>
          <option value="zh-cn">Chinese (Simplified)</option>
          <option value="zh-tw">Chinese (Traditional)</option>
          <option value="fr">French</option>
          <option value="de">German</option>
        </select>
      </label>
    </div>
  );
};

export default App;
