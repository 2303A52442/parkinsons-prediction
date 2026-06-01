# Parkinson's Disease Prediction — Portfolio Project

Brief production-ready demo that predicts Parkinson's Disease from biomedical voice features using a Logistic Regression model.

## What this repo contains
- `sml_project.ipynb` — original analysis and preprocessing notes.
- `train_model.py` — trains and saves a scikit-learn pipeline to `models/model.pkl`.
- `models/model.pkl` — trained Logistic Regression pipeline (if present).
- `streamlit_app.py` — production-ready Streamlit UI for predictions, visualizations, history, and PDF export.
- `app.py` + `templates/index.html` — simple Flask API/HTML demo (alternative UI).
- `dataset/Parkinsson disease.csv` — UCI Parkinson's dataset (source file).

## Features
- 22 biomedical voice features as model inputs
- Logistic Regression model
- Streamlit UI with grouped inputs, PCA, confusion matrix, top-feature importance
- Prediction card with status, confidence, and recommendation
- Prediction history with timestamps and PDF export

## Quick local run (Windows)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py   # creates models/model.pkl if missing
streamlit run streamlit_app.py
```

## Notes
- Run the app with `streamlit run` to enable full session and download features.
- This project is for educational and research purposes only and is not a medical diagnostic tool.

## Next steps
- Polish UI visuals (icons, typography)
- Add SHAP explanations (optional)
- Add CI/CD or Dockerfile for reproducible deployment
# Parkinson's disease prediction — small web app

Files added:

- `train_model.py`: trains a MinMaxScaler + LogisticRegression pipeline and saves `models/model.pkl`.
- `app.py`: Flask app that loads the saved model and serves a basic prediction UI and JSON API.
- `requirements.txt`: Python dependencies.

Quick start:

1. Create a virtualenv and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. Train model:

```bash
python train_model.py
```

3. Run the app:

```bash
python app.py
```

Open http://localhost:5000 and use the form or POST JSON to `/predict`.

Streamlit UI
------------

There is also a Streamlit app `streamlit_app.py` that provides a clean UI with a "Load Sample Patient" button and shows prediction + confidence.

Run it locally:

```bash
streamlit run streamlit_app.py
```

Hosting on GitHub / Streamlit Community
--------------------------------------

- Create a new GitHub repository and push this project (follow `git init`, `git add .`, `git commit`, `git branch -M main`, `git remote add origin <url>`, `git push -u origin main`).
- On Streamlit Community Cloud (https://share.streamlit.io) you can deploy by connecting your GitHub repo and specifying `streamlit_app.py` as the app entrypoint. Make sure `requirements.txt` is present.

If you want, I can create a GitHub Actions workflow to run tests or automatically train and save the model on push — tell me if you'd like that.
# Parkinsons prediction web app

Quick steps:

- Install dependencies:

```bash
python -m pip install -r requirements.txt
```

- Train and save model:

```bash
python train_model.py
```

- Run the web app:

```bash
python app.py
```

Then open http://localhost:5000 in your browser and use the textarea to paste comma-separated feature values (same order as in the dataset, excluding `name` and `status`).
