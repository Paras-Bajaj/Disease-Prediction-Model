from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
import warnings
from datetime import datetime
import requests

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Suppress warnings
warnings.filterwarnings('ignore')

# Global variables for model components
rf_model = None
vectorizer = None
label_encoder = None
preprocess_text_func = None
model_ready = False

def create_simple_preprocessing_function():
    """Create a simple text preprocessing function that doesn't require NLTK"""
    
    # Common English stop words
    stop_words = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
        'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once'
    }
    
    def preprocess(text):
        if pd.isna(text) or not text:
            return ""
        
        try:
            # Convert to lowercase and remove special characters
            text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
            text = ' '.join(text.split())  # Remove extra whitespace
            
            # Process words - remove stop words and short words
            words = []
            for word in text.split():
                if word not in stop_words and len(word) > 2:
                    words.append(word)
            
            return ' '.join(words)
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return str(text).lower()
    
    return preprocess

def load_and_train_model():
    """Load dataset and train model"""
    global rf_model, vectorizer, label_encoder, preprocess_text_func, model_ready
    
    try:
        print("Starting model training...")
        
        # Create preprocessing function first
        preprocess_text_func = create_simple_preprocessing_function()
        print("✓ Preprocessing function created")
        
        # Load dataset from URL
        dataset_url = "Symptom.csv"
        print(f"Loading dataset from: {dataset_url}")
        
        try:
            df = pd.read_csv(dataset_url)
        except Exception as e:
            print(f"Failed to load from URL: {e}")
            # Try to create a small sample dataset as fallback
            print("Creating fallback dataset...")
            df = create_fallback_dataset()
        
        print(f"✓ Dataset loaded. Shape: {df.shape}")
        print(f"✓ Columns: {df.columns.tolist()}")
        
        # Check if we have the required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            print("Error: Dataset missing required columns 'text' and 'label'")
            return False
        
        print(f"✓ Sample labels: {df['label'].value_counts().head()}")
        
        # Preprocess data
        print("Preprocessing text data...")
        df['processed_text'] = df['text'].apply(preprocess_text_func)
        df = df[df['processed_text'].str.len() > 0]  # Remove empty texts
        
        print(f"✓ After preprocessing: {df.shape}")
        
        if len(df) < 10:
            print("Error: Not enough data after preprocessing")
            return False
        
        # Prepare features and target
        X = df['processed_text']
        y = df['label']
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        print(f"✓ Number of unique diseases: {len(label_encoder.classes_)}")
        print(f"✓ Diseases: {list(label_encoder.classes_)[:10]}...")  # Show first 10
        
        # Split data
        if len(df) > 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
        else:
            # If dataset is too small, use all data for training
            X_train, y_train = X, y_encoded
            X_test, y_test = X, y_encoded
        
        print(f"✓ Training set size: {len(X_train)}")
        
        # Feature extraction
        vectorizer = TfidfVectorizer(
            max_features=min(1000, len(X_train) * 10),  # Adjust based on data size
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        print("Creating TF-IDF features...")
        X_train_tfidf = vectorizer.fit_transform(X_train)
        print(f"✓ Feature matrix shape: {X_train_tfidf.shape}")
        
        # Train model
        rf_model = RandomForestClassifier(
            n_estimators=min(50, len(X_train)),  # Adjust based on data size
            max_depth=10,
            min_samples_split=2,
            random_state=42,
            n_jobs=1  # Use single thread to avoid issues
        )
        
        print("Training Random Forest model...")
        rf_model.fit(X_train_tfidf, y_train)
        
        # Test accuracy
        X_test_tfidf = vectorizer.transform(X_test)
        accuracy = rf_model.score(X_test_tfidf, y_test)
        print(f"✓ Model accuracy: {accuracy:.3f}")
        
        model_ready = True
        print("✓ Model training completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error in model training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_fallback_dataset():
    """Create a small fallback dataset if the main dataset fails to load"""
    data = {
        'text': [
            "I constantly sneeze and have a dry cough. My infections don't seem to be healing, and I have palpitations. My throat does ache occasionally, but it usually gets better.",
            "I have severe headache with nausea and sensitivity to light that started 2 days ago",
            "I feel very tired, have frequent urination, and excessive thirst for the past week",
            "I have chest pain with shortness of breath and feel dizzy",
            "I have been coughing for weeks with wheezing and difficulty breathing",
            "I have stomach pain with nausea and vomiting after eating",
            "I have joint pain and stiffness especially in the morning",
            "I have skin rash with itching and swelling",
            "I have back pain with muscle spasms that gets worse with movement",
            "I feel sad and hopeless with loss of interest in activities",
            "I feel anxious and worried with rapid heartbeat and sweating",
            "I have runny nose with sore throat and mild fever",
            "I have high fever with body aches and chills",
            "I have blurred vision with frequent urination and increased thirst",
            "I have persistent cough with chest tightness and wheezing at night"
        ],
        'label': [
            'diabetes', 'migraine', 'diabetes', 'heart disease', 'asthma',
            'gastroenteritis', 'arthritis', 'allergic reaction', 'back pain',
            'depression', 'anxiety', 'common cold', 'influenza', 'diabetes', 'asthma'
        ]
    }
    return pd.DataFrame(data)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for disease prediction from symptoms"""
    try:
        # Check if model is ready
        if not model_ready or rf_model is None or vectorizer is None or label_encoder is None or preprocess_text_func is None:
            return jsonify({
                'status': 'error',
                'message': 'AI model is not ready. Please wait for model initialization to complete.',
                'code': 'MODEL_NOT_READY'
            }), 503
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'symptoms' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No symptom data provided',
                'code': 'MISSING_DATA'
            }), 400
        
        symptoms = data['symptoms']
        
        # Validate required fields
        if 'description' not in symptoms or not symptoms['description']:
            return jsonify({
                'status': 'error',
                'message': 'Symptom description is required',
                'code': 'MISSING_DESCRIPTION'
            }), 400
        
        # Combine all symptom-related information into one string
        symptom_text = f"{symptoms.get('description', '')}. "
        if symptoms.get('duration'):
            symptom_text += f"Duration: {symptoms.get('duration', '')}. "
        if symptoms.get('severity'):
            symptom_text += f"Severity: {symptoms.get('severity', '')}. "
        if symptoms.get('triggers'):
            symptom_text += f"Triggers: {symptoms.get('triggers', '')}"
        
        print(f"Processing symptoms: {symptom_text}")
        
        # Get predictions
        prediction_result = predict_disease(symptom_text)
        
        if 'error' in prediction_result:
            return jsonify({
                'status': 'error',
                'message': prediction_result['error'],
                'code': 'PREDICTION_ERROR'
            }), 400
        
        # Generate recommendations based on severity
        recommendations = generate_recommendations(
            prediction_result['primary_prediction'],
            symptoms.get('severity', 'moderate')
        )
        
        # Format response to match frontend expectations
        response = {
            'primaryDiagnosis': {
                'name': prediction_result['primary_prediction'],
                'probability': prediction_result['confidence'],
                'description': get_disease_description(prediction_result['primary_prediction'])
            },
            'differentialDiagnoses': [
                {
                    'name': pred['disease'],
                    'probability': pred['confidence'],
                    'description': get_disease_description(pred['disease'])
                } for pred in prediction_result['top_predictions'][1:] if len(prediction_result['top_predictions']) > 1
            ],
            'recommendedActions': recommendations,
            'urgencyLevel': get_urgency_level(symptoms.get('severity', 'moderate')),
            'processedSymptoms': prediction_result['processed_symptoms'],
            'status': 'success',
            'modelVersion': 'v2.1.0',
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✓ Prediction successful: {prediction_result['primary_prediction']} ({prediction_result['confidence']:.3f})")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"✗ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}',
            'code': 'SERVER_ERROR'
        }), 500

def predict_disease(symptoms_text, top_k=3):
    """
    Predict disease from symptoms text
    """
    try:
        # Preprocess input
        processed_symptoms = preprocess_text_func(symptoms_text)
        
        if not processed_symptoms:
            return {"error": "No valid symptoms found after processing"}
        
        print(f"Processed symptoms: '{processed_symptoms}'")
        
        # Vectorize
        symptoms_tfidf = vectorizer.transform([processed_symptoms])
        
        # Get predictions and probabilities
        probabilities = rf_model.predict_proba(symptoms_tfidf)[0]
        
        # Get top k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        predictions = []
        for idx in top_indices:
            disease = label_encoder.inverse_transform([idx])[0]
            confidence = probabilities[idx]
            predictions.append({
                'disease': disease,
                'confidence': float(confidence),
                'percentage': f"{confidence*100:.1f}%"
            })
        
        return {
            'primary_prediction': predictions[0]['disease'],
            'confidence': predictions[0]['confidence'],
            'top_predictions': predictions,
            'processed_symptoms': processed_symptoms
        }
        
    except Exception as e:
        print(f"✗ Error in predict_disease: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}

def generate_recommendations(disease, severity):
    """Generate recommendations based on disease and severity"""
    base_recommendations = [
        "Rest and stay hydrated",
        "Monitor symptoms for changes"
    ]
    
    # Disease-specific recommendations
    disease_recommendations = {
        'diabetes': [
            "Monitor blood sugar levels regularly",
            "Follow a balanced diet low in sugar",
            "Take prescribed medications as directed",
            "Exercise regularly as recommended by your doctor"
        ],
        'hypertension': [
            "Monitor blood pressure regularly",
            "Reduce sodium intake",
            "Engage in regular physical activity",
            "Manage stress levels"
        ],
        'migraine': [
            "Rest in a quiet, dark room",
            "Apply cold compress to forehead",
            "Consider over-the-counter pain relievers",
            "Avoid known triggers"
        ],
        'asthma': [
            "Use prescribed inhaler as directed",
            "Avoid known triggers",
            "Monitor peak flow if recommended",
            "Keep rescue inhaler accessible"
        ],
        'heart disease': [
            "Take prescribed medications regularly",
            "Follow a heart-healthy diet",
            "Engage in approved physical activity",
            "Monitor symptoms closely"
        ]
    }
    
    severity_recommendations = {
        'mild': ["Schedule a doctor's appointment if symptoms persist"],
        'moderate': ["Consult a healthcare provider within 48 hours"],
        'severe': ["Seek medical attention within 24 hours"],
        'emergency': ["Call emergency services or go to ER immediately"]
    }
    
    recommendations = base_recommendations.copy()
    
    # Add disease-specific recommendations
    disease_lower = disease.lower()
    for key, recs in disease_recommendations.items():
        if key in disease_lower:
            recommendations.extend(recs)
            break
    
    # Add severity-based recommendations
    recommendations.extend(severity_recommendations.get(severity, []))
    
    return recommendations

def get_disease_description(disease_name):
    """Get description for a disease"""
    descriptions = {
        "diabetes": "A group of metabolic disorders characterized by high blood sugar levels over a prolonged period.",
        "hypertension": "High blood pressure that can lead to serious health complications if left untreated.",
        "migraine": "A headache that can cause severe throbbing pain or pulsing sensation, usually on one side of the head.",
        "asthma": "A respiratory condition marked by attacks of spasm in the bronchi, causing difficulty breathing.",
        "arthritis": "Inflammation of one or more joints, causing pain and stiffness that typically worsens with age.",
        "depression": "A mental health disorder characterized by persistent sadness and loss of interest in activities.",
        "anxiety": "A mental health disorder characterized by excessive worry, fear, or nervousness.",
        "heart disease": "A range of conditions that affect the heart's structure and function.",
        "gastroenteritis": "Inflammation of the stomach and intestines, typically from infection or food poisoning.",
        "allergic reaction": "An immune system response to a substance that's usually harmless to most people.",
        "back pain": "Pain in the back that can be caused by muscle strain, injury, or underlying conditions.",
        "common cold": "A viral infection of the upper respiratory tract causing runny nose, sore throat, and cough.",
        "influenza": "A viral infection causing fever, chills, muscle aches, cough, and fatigue."
    }
    
    disease_lower = disease_name.lower()
    for key, desc in descriptions.items():
        if key in disease_lower:
            return desc
    
    return f"A medical condition that requires professional evaluation and appropriate treatment."

def get_urgency_level(severity):
    """Determine urgency level based on symptom severity"""
    levels = {
        'mild': "Self-care recommended - monitor symptoms and consult if they persist",
        'moderate': "Schedule doctor visit within a few days",
        'severe': "Seek urgent medical care within 24 hours",
        'emergency': "Emergency care needed - call 911 immediately"
    }
    return levels.get(severity, "Consult a healthcare provider for proper evaluation")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_ready': model_ready,
        'components_ready': {
            'rf_model': rf_model is not None,
            'vectorizer': vectorizer is not None,
            'label_encoder': label_encoder is not None,
            'preprocess_func': preprocess_text_func is not None
        },
        'version': 'v2.1.0',
        'diseases': list(label_encoder.classes_) if label_encoder else [],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Endpoint to retrain the model"""
    try:
        success = load_and_train_model()
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Model retrained successfully',
                'model_ready': model_ready
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Model retraining failed'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Initialize the model when the app starts
print("🏥 MediScan Pro - AI Medical Diagnosis System")
print("=" * 50)
print("Initializing AI model...")

if load_and_train_model():
    print("✅ AI model ready!")
else:
    print("❌ AI model initialization failed!")
    print("The server will still run, but predictions may not work correctly.")

if __name__ == '__main__':
    print("\n🚀 Starting Flask server...")
    print("📝 API endpoints:")
    print("   - POST /predict - Predict disease from symptoms")
    print("   - GET /health - Health check")
    print("   - POST /retrain - Retrain model")
    print(f"   - Server URL: http://localhost:5000")
    print("\n" + "=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
