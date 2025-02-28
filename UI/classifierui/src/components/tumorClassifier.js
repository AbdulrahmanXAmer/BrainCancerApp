import React, { useState, useCallback, useRef } from "react";
import axios from "axios";
import "bulma/css/bulma.min.css";
import { FaUpload } from "react-icons/fa";
import { FaSpinner } from "react-icons/fa";
import { FaTimes } from "react-icons/fa";

const CancerClassifier = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);

  const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB

  const handleFileChange = useCallback((event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file (JPG, PNG, etc.)");
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      setError("Image must be smaller than 5MB");
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setPrediction("");
    setConfidence(null);
    setError("");
  }, []);

  const handleDrop = useCallback((event) => {
    event.preventDefault();
    event.stopPropagation();
    const file = event.dataTransfer.files[0];
    if (file) handleFileChange({ target: { files: [file] } });
  }, [handleFileChange]);

  const handleDragOver = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please select an image first");
      return;
    }

    setLoading(true);
    setError("");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 30000,
      });

      const { prediction, confidence } = response.data;
      setPrediction(prediction);
      setConfidence(confidence);
    } catch (err) {
      setError(
        err.response?.data?.message ||
        "Failed to process image. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setPrediction("");
    setConfidence(null);
    setError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="container" style={styles.container}>
      <div style={styles.card}>
        <h1 style={styles.title}>Cancer Classifier</h1>
        <p style={styles.subtitle}>AI-Powered Image Diagnosis</p>

        <div
          style={{
            ...styles.uploadArea,
            ...(loading ? styles.uploadAreaDisabled : {}),
          }}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => !loading && fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            style={styles.hiddenInput}
            disabled={loading}
          />
          
          {previewUrl ? (
            <div style={styles.previewContainer}>
              <img src={previewUrl} alt="Preview" style={styles.previewImage} />
              <button
                style={styles.removeButton}
                onClick={(e) => {
                  e.stopPropagation();
                  resetForm();
                }}
                disabled={loading}
              >
                <FaTimes />
              </button>
            </div>
          ) : (
            <div style={styles.uploadPrompt}>
              <FaUpload style={styles.uploadIcon} />
              <p>Drop an image here or click to browse</p>
              <p style={styles.uploadHint}>Supports JPG, PNG (max 5MB)</p>
            </div>
          )}
        </div>

        <button
          style={{
            ...styles.uploadButton,
            ...(loading || !selectedFile ? styles.buttonDisabled : {}),
          }}
          onClick={handleUpload}
          disabled={loading || !selectedFile}
        >
          {loading ? (
            <>
              <FaSpinner style={styles.spinner} />
              Analyzing...
            </>
          ) : (
            "Classify Image"
          )}
        </button>

        {error && (
          <div style={styles.errorBox}>
            <p>{error}</p>
          </div>
        )}

        {(loading || prediction) && (
          <div
            style={{
              ...styles.resultBox,
              backgroundColor: loading
                ? "#dfe6e9" // Neutral gray for loading
                : prediction === "Tumor"
                ? "#ff6b6b"
                : "#4ecdc4",
            }}
          >
            {loading ? (
              <div style={styles.loadingContainer}>
                <FaSpinner style={styles.spinner} />
                <p style={styles.loadingText}>Predicting...</p>
              </div>
            ) : (
              <>
                <p style={styles.resultText}>
                  Diagnosis:{" "}
                  <span style={styles.resultHighlight}>{prediction}</span>
                </p>
                {confidence && (
                  <p style={styles.confidenceText}>
                    Confidence: {(confidence * 100).toFixed(1)}%
                  </p>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const styles = {
  container: {
    minHeight: "100vh",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
    padding: "20px",
  },
  card: {
    background: "#ffffff",
    borderRadius: "16px",
    padding: "40px",
    width: "100%",
    maxWidth: "480px",
    boxShadow: "0 10px 30px rgba(0, 0, 0, 0.1)",
    fontFamily: "'Inter', sans-serif",
  },
  title: {
    fontSize: "2.5rem",
    fontWeight: 700,
    color: "#2d3436",
    marginBottom: "8px",
    letterSpacing: "-0.5px",
  },
  subtitle: {
    fontSize: "1.1rem",
    color: "#636e72",
    marginBottom: "30px",
    fontWeight: 400,
  },
  uploadArea: {
    border: "2px dashed #dfe6e9",
    borderRadius: "12px",
    padding: "20px",
    textAlign: "center",
    cursor: "pointer",
    transition: "all 0.3s ease",
    backgroundColor: "#f9fbfc",
    "&:hover": {
      borderColor: "#74b9ff",
      backgroundColor: "#f1f9ff",
    },
  },
  uploadAreaDisabled: {
    opacity: 0.6,
    cursor: "not-allowed",
  },
  hiddenInput: {
    display: "none",
  },
  uploadPrompt: {
    color: "#636e72",
    fontSize: "1rem",
  },
  uploadIcon: {
    fontSize: "24px",
    color: "#74b9ff",
    marginBottom: "10px",
  },
  uploadHint: {
    fontSize: "0.85rem",
    color: "#b2bec3",
    marginTop: "5px",
  },
  previewContainer: {
    position: "relative",
    marginBottom: "10px",
  },
  previewImage: {
    maxWidth: "100%",
    maxHeight: "200px",
    borderRadius: "8px",
    objectFit: "cover",
  },
  removeButton: {
    position: "absolute",
    top: "10px",
    right: "10px",
    background: "rgba(0, 0, 0, 0.7)",
    border: "none",
    borderRadius: "50%",
    width: "24px",
    height: "24px",
    color: "#fff",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    transition: "all 0.2s",
    "&:hover": { background: "rgba(0, 0, 0, 0.9)" },
  },
  uploadButton: {
    background: "linear-gradient(45deg, #0984e3, #74b9ff)",
    color: "#fff",
    border: "none",
    borderRadius: "8px",
    padding: "12px 24px",
    fontSize: "1rem",
    fontWeight: 600,
    width: "100%",
    marginTop: "20px",
    cursor: "pointer",
    transition: "all 0.3s ease",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    "&:hover": {
      background: "linear-gradient(45deg, #74b9ff, #0984e3)",
    },
  },
  buttonDisabled: {
    background: "#b2bec3",
    cursor: "not-allowed",
    "&:hover": { background: "#b2bec3" },
  },
  spinner: {
    marginRight: "8px",
    animation: "spin 1s linear infinite",
  },
  errorBox: {
    background: "#ffebeb",
    color: "#d63031",
    padding: "12px",
    borderRadius: "8px",
    marginTop: "20px",
    fontSize: "0.9rem",
  },
  resultBox: {
    padding: "15px",
    borderRadius: "8px",
    marginTop: "20px",
    color: "#fff",
    animation: "fadeIn 0.5s ease",
  },
  resultText: {
    fontSize: "1.1rem",
    fontWeight: 600,
    marginBottom: "5px",
  },
  resultHighlight: {
    fontWeight: 700,
    textTransform: "uppercase",
  },
  confidenceText: {
    fontSize: "0.95rem",
    opacity: 0.9,
  },
  loadingContainer: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "column",
    color: "#2d3436", // Dark text for visibility on light background
  },
  loadingText: {
    fontSize: "1.1rem",
    fontWeight: 600,
    marginTop: "10px",
  },
};

// Add this to your CSS file or a <style> tag
const globalStyles = `
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }
`;

export default CancerClassifier;