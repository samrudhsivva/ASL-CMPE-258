import React, { useState } from "react";
import axios from "axios";
import { useSpeechSynthesis } from "react-speech-kit";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState("");
  const { speak } = useSpeechSynthesis();

  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!image) {
      alert("Please upload an image first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await axios.post("http://127.0.0.1:8000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(response.data.predicted_label);
    } catch (error) {
      console.error("Error:", error);
      alert("Error predicting the image. Please try again.");
    }
  };

  const handleSpeak = () => {
    if (result) {
      speak({ text: `The predicted sign is ${result}` });
    }
  };

  return (
    <div className="app-container">
      <h1 className="title">ASL Recognition</h1>
      
      <form onSubmit={handleSubmit} className="upload-form">
        <input type="file" accept="image/*" onChange={handleImageChange} className="file-input" />
        <button type="submit" className="predict-button">Predict</button>
      </form>

      {result && (
        <div className="result-container">
          <h2 className="result">Predicted Sign: {result}</h2>
          <button onClick={handleSpeak} className="speak-button">Speak</button>
        </div>
      )}
    </div>
  );
}

export default App;