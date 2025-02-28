import React from "react";
import { FaDownload } from "react-icons/fa";
import "bulma/css/bulma.min.css";

const Images = () => {
  const handleDownload = (imageName) => {
    const link = document.createElement("a");
    link.href = `/${imageName}`;
    link.download = imageName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <section className="section" style={styles.section}>
      <div className="container">
        <h2 className="title is-2 has-text-centered" style={styles.title}>
          Test the Model
        </h2>
        <p className="subtitle is-5 has-text-centered" style={styles.subtitle}>
          Download these images for testing
        </p>

        <div className="columns is-multiline is-centered" style={styles.imageGrid}>
          {/* Healthy Brain Image */}
          <div className="column is-narrow" style={styles.imageWrapper}>
            <figure className="image">
              <img
                src="/healthy.jpg"
                alt="Healthy Brain"
                style={styles.image}
              />
            </figure>
            <p className="has-text-weight-bold has-text-centered" style={styles.imageLabel}>
              Healthy Brain
            </p>
            <button
              className="button is-info is-fullwidth"
              style={styles.downloadButton}
              onClick={() => handleDownload("healthy.jpg")}
            >
              <span className="icon">
                <FaDownload />
              </span>
              <span>Download</span>
            </button>
          </div>

          {/* Tumor Brain Image */}
          <div className="column is-narrow" style={styles.imageWrapper}>
            <figure className="image">
              <img
                src="/tumor.jpg"
                alt="Brain with Tumor"
                style={styles.image}
              />
            </figure>
            <p className="has-text-weight-bold has-text-centered" style={styles.imageLabel}>
              Brain with Tumor
            </p>
            <button
              className="button is-info is-fullwidth"
              style={styles.downloadButton}
              onClick={() => handleDownload("tumor.jpg")}
            >
              <span className="icon">
                <FaDownload />
              </span>
              <span>Download</span>
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

const styles = {
  section: {
    background: "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
    padding: "3rem 1.5rem",
  },
  title: {
    fontSize: "2.5rem",
    fontWeight: 700,
    color: "#2d3436",
    marginBottom: "0.5rem",
    fontFamily: "'Inter', sans-serif",
  },
  subtitle: {
    fontSize: "1.1rem",
    color: "#636e72",
    marginBottom: "2rem",
    fontFamily: "'Inter', sans-serif",
  },
  imageGrid: {
    gap: "2rem",
  },
  imageWrapper: {
    background: "#ffffff",
    borderRadius: "12px",
    padding: "1.5rem",
    boxShadow: "0 10px 30px rgba(0, 0, 0, 0.1)",
    transition: "transform 0.3s ease",
    maxWidth: "300px",
    "&:hover": {
      transform: "translateY(-5px)",
    },
  },
  image: {
    width: "100%",
    height: "200px",
    borderRadius: "8px",
    objectFit: "cover",
    transition: "transform 0.2s ease",
    "&:hover": {
      transform: "scale(1.02)",
    },
  },
  imageLabel: {
    marginTop: "1rem",
    fontSize: "1rem",
    color: "#2d3436",
    fontFamily: "'Inter', sans-serif",
  },
  downloadButton: {
    marginTop: "1rem",
    background: "linear-gradient(45deg, #0984e3, #74b9ff)",
    border: "none",
    color: "#fff",
    fontWeight: 600,
    transition: "all 0.3s ease",
    "&:hover": {
      background: "linear-gradient(45deg, #74b9ff, #0984e3)",
      color: "#fff",
    },
  },
};

export default Images;