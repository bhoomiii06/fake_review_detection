# AI Trust Scoring System for Fake Review Detection
<img width="2880" height="1708" alt="index" src="https://github.com/user-attachments/assets/3beb7f01-7e19-4d84-a414-f83d8b19ba34" />

<img width="2880" height="1722" alt="index with review" src="https://github.com/user-attachments/assets/9e84cdc2-b5d5-4f98-8316-6ab751213b10" />

<img width="2880" height="1717" alt="admin" src="https://github.com/user-attachments/assets/9cdfe27e-698c-4677-99da-4cd6935b4250" />


## Project Overview

The AI Trust Scoring System is a full-stack machine learning application designed to identify and flag sophisticated fake reviews (opinion spam) on e-commerce and local business platforms. 

Traditional fake review detectors rely solely on Natural Language Processing (NLP). However, modern deceptive reviews are often written by paid human crowdsourcers or Large Language Models (LLMs), making their text indistinguishable from genuine reviews. To solve this, this project implements a **Hybrid Dual-Engine Architecture** that analyzes both textual linguistics and user behavioral metadata to generate a comprehensive Trust Score.

## Methodology & Architecture

This system evaluates reviews through two distinct layers of analysis:

### 1. The Textual Intelligence Engine (NLP)
This engine analyzes the actual content of the review to detect deceptive linguistic patterns.
* **Data Preprocessing:** Utilizes `nltk` for text normalization, stop-word removal, and lemmatization.
* **Feature Extraction:** Converts cleaned text into numerical vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
* **Classification Model:** Employs a **Random Forest Classifier** trained on a dataset of over 350,000 reviews. To combat class imbalance and ensure accurate identification of genuine reviews, the model is optimized using Cost-Sensitive Learning (`class_weight='balanced'`).

### 2. The Behavioral Analytics Engine (Heuristics)
This deterministic engine evaluates the user's metadata and posting history. Even if the NLP engine passes the text as human-written, the behavioral engine applies mathematical penalties for bot-like activity:
* **Review Velocity Penalty:** Flags users who post an impossibly high volume of reviews within a 24-hour window.
* **Rating Extremity Penalty:** Calculates the standard deviation of a user's lifetime ratings. Users with zero variance (e.g., exclusively posting 5-star reviews to boost a product) are mathematically flagged.

The final output is a **0-100% Trust Score** derived from the base machine learning probability minus the behavioral penalties.

## Key Features

* **Real-Time Analysis Dashboard:** A front-end interface where users can input review text and a User ID to receive an instant Trust Score classification.
* **Persistent Logging Integration:** Powered by **SQLite**, every analyzed review is permanently stored in a local database to ensure auditable system actions.
* **Admin Security Portal:** A dedicated web route (`/admin`) that queries the database to display a live, historical log of all processed reviews and triggered behavioral warnings.

## Tech Stack

* **Machine Learning:** Scikit-learn, NLTK, Pandas, NumPy
* **Backend Pipeline:** Python, Flask, SQLite3
* **Frontend:** HTML5, CSS3, Vanilla JavaScript (Fetch API)
* **Model Serialization:** Joblib

## Repository Structure
```text
├── models/
│   ├── random_forest_model.pkl       # Serialized ML model (Ignored in Git)
│   ├── tfidf_vectorizer.pkl          # Serialized vectorizer (Ignored in Git)
├── templates/
│   ├── index.html                    # Main Scanner UI
│   └── admin.html                    # Security Logs Dashboard
├── app.py                            # Flask Server & API routing
├── requirements.txt                  # Dependency list
└── README.md                         # Project documentation

## Installation and Local Setup

To deploy this application locally, follow these steps:

**1. Clone the repository**
''Bash
git clone https://github.com/bhoomiii06/fake_review_detection.git

**2. Activate the virtual environment**
''Bash
env\Scripts\activate

**3. Install required dependencies**
''Bash
pip install -r requirements.txt

**4. Start the application server**
''Bash
python app.py

**5. Access the Web Interface**
Main Scanner: http://127.0.0.1:5000
Admin Security Logs: http://127.0.0.1:5000/admin
