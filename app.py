from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sqlite3
import datetime

# 1. Initialize the Web Server
app = Flask(__name__)

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    # Create a table to store reviews if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            review_text TEXT,
            classification TEXT,
            trust_score REAL,
            behavioral_flag TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Run database setup on startup
init_db()


# 2. Load the AI Brains
print("Loading AI Models...")
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
rf_model = joblib.load('models/random_forest_model.pkl')
user_db = pd.read_csv('models/user_behavior_database.csv')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# 3. Web Pages (Frontend Routes)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/admin')
def admin_dashboard():
    # Fetch all saved reviews from the database to show on the admin page
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM reviews ORDER BY timestamp DESC")
    saved_reviews = c.fetchall()
    conn.close()
    return render_template('admin.html', reviews=saved_reviews)

# 4. API Endpoint (The AI Logic + Saving to Database)
@app.route('/analyze_review', methods=['POST'])
def analyze_review():
    data = request.json
    user_id = data.get('user_id', 'unknown_user')
    review_text = data.get('review_text', '')

    cleaned_review = clean_text(review_text)
    vectorized_text = tfidf.transform([cleaned_review])
    
    ml_confidence = rf_model.predict_proba(vectorized_text)[0][0] * 100
    
    trust_score = ml_confidence
    behavior_warning = "Normal"
    
    user_history = user_db[user_db['user_id'] == user_id]
    if not user_history.empty:
        if user_history['is_extreme_reviewer'].iloc[0] == 1:
            trust_score -= 25  
            behavior_warning = "Suspicious Rating Extremity Detected"
            
    trust_score = max(0, min(100, round(trust_score, 1)))
    classification = "Genuine" if trust_score >= 50 else "Fake/Suspicious"

    # --- SAVE RESULT TO DATABASE ---
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO reviews (user_id, review_text, classification, trust_score, behavioral_flag, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, review_text, classification, trust_score, behavior_warning, timestamp))
    conn.commit()
    conn.close()
    # -------------------------------

    return jsonify({
        'user_id': user_id,
        'classification': classification,
        'trust_score': trust_score,
        'behavioral_flag': behavior_warning
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)