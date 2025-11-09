# 🎧 Music BPM Predictor

An interactive **Streamlit web application** that predicts the **Beats Per Minute (BPM)** of a music track based on its audio features using a trained Machine Learning model.

---

## 🚀 Overview

This project uses a pre-trained regression model (`best_model.pkl`) to predict the tempo (BPM) of a song from its input features.  
It includes multiple functionalities for user interaction and analysis:

- 🎛️ **Single Prediction:** Enter audio feature values manually to get a BPM prediction.  
- 📂 **Batch Prediction:** Upload a CSV file containing multiple records to predict BPMs in bulk.  
---

## 🧠 Tech Stack  

| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **Backend / ML** | Python, Scikit-learn |
| **Data Handling** | Pandas, NumPy |
| **Model Serialization** | Pickle |
| **Visualization** | Matplotlib / Seaborn |

---

## 🗂️ Project Structure
├── beats_app.py # Main Streamlit application
|-- untitled22.py # THE MAIN CODE FOR THE MODEL 
├── best_model.pkl # Pre-trained ML model
├── requirements.txt # Python dependencies
├── README.md # Project documentation

## Create a virtual environment

python -m venv venv
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows

## INSTALL DEPENDENCIES
pip install -r requirements.txt

## RUN APP LOCALLY 
streamlit run beats_app.py

## 🌐 Deployment
1) Deploy on Streamlit Cloud

2) Push your project to a GitHub repository.

3) Go to Streamlit Cloud.

4) Click “New app” → Connect your GitHub repo.

5) Select the branch and file (e.g. app.py).

6) Add requirements.txt in the repo root.

7) Click Deploy 🚀.

## REQUIREMENTS 
streamlit
pandas
numpy
scikit-learn
matplotlib
pickle-mixin
gdown

✅ gdown is included since the dataset is downloaded from Google Drive during runtime.

📊 Model Information

- Model file: best_model.pkl

- Regression (trained using scikit-learn)

- Dataset: Downloaded automatically from Google Drive via gdown

- Output: Predicted Beats Per Minute (BPM)

🧑‍💻 Author

Jenish Kharva
📧 22aikha028@ldce.ac.in



