from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
import logging
import random
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from functools import wraps
from PIL import Image
import io
import requests
import json
import torch
from torchvision import transforms
import numpy as np
from torchvision import models
import torch.nn as nn

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.chatbot_responses import handle_chat_request

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Range", "X-Content-Range"]
    }
})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = app.logger

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Add secret key for JWT
app.config['SECRET_KEY'] = 'your-secret-key'  # Change this to a secure secret key

# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Fallback responses if Ollama is not available
FALLBACK_RESPONSES = {
    "skin_concern": [
        "Based on your description of {concern}, I recommend using gentle, non-comedogenic products and maintaining a consistent skincare routine.",
        "For {concern}, it's important to keep the affected area clean and avoid touching it frequently. Consider using products with salicylic acid or benzoyl peroxide.",
        "If you're experiencing {concern}, try using a gentle cleanser and moisturizer. If symptoms persist, consult a dermatologist."
    ],
    "product_recommendation": [
        "For {product_type}, I recommend looking for products with ingredients like hyaluronic acid and ceramides.",
        "When choosing a {product_type}, consider your skin type. Look for oil-free formulas if you have oily skin, or richer creams for dry skin.",
        "A good {product_type} should be suitable for your skin type and contain beneficial ingredients like niacinamide or vitamin C."
    ],
    "routine_advice": [
        "A basic skincare routine should include: cleansing, toning (optional), moisturizing, and sunscreen during the day.",
        "For best results, apply products in order of thickness: start with the thinnest (toners) and end with the thickest (oils, creams).",
        "Remember to patch test new products and introduce them one at a time into your routine."
    ],
    "image_analysis": [
        "I can see your skin concern in the image. For this type of {concern}, I recommend keeping the area clean and moisturized.",
        "Based on the image showing {concern}, you might want to try products containing soothing ingredients like aloe vera or chamomile.",
        "The image shows signs of {concern}. Consider using gentle, fragrance-free products and avoid harsh exfoliants."
    ]
}

# Global users dictionary
users = {}
chats = {}

def init_test_users():
    global users
    # Clear existing users
    users.clear()
    
    # Create test users
    users['client@test.com'] = {
        'name': 'Test Client',
        'email': 'client@test.com',
        'password': generate_password_hash('client123'),
        'user_type': 'client'
    }
    
    users['doctor@test.com'] = {
        'name': 'Dr. Smith',
        'email': 'doctor@test.com',
        'password': generate_password_hash('doctor123'),
        'user_type': 'dermatologist'
    }
    
    logger.info("Test users initialized:")
    for email in users:
        logger.info(f"User: {email}, Type: {users[email]['user_type']}")

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'status': 'error', 'message': 'Token is missing'}), 401
        
        try:
            data = jwt.decode(token.split()[1], app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = users.get(data['email'])
        except:
            return jsonify({'status': 'error', 'message': 'Token is invalid'}), 401
            
        return f(current_user, *args, **kwargs)
    return decorated

@app.route('/')
def home():
    return "API Server Running"

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        email = data.get('email')
        
        logger.info(f"Registration attempt - Email: {email}")
        logger.info(f"Registration data: {data}")
        
        if email in users:
            return jsonify({
                'status': 'error',
                'message': 'Email already registered'
            }), 400
        
        users[email] = {
            'name': data.get('name'),
            'email': email,
            'password': generate_password_hash(data.get('password')),
            'user_type': data.get('userType')
        }
        
        logger.info(f"New user registered: {email}")
        
        return jsonify({
            'status': 'success',
            'message': 'Registration successful'
        })
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email', '')
        password = data.get('password', '')
        
        logger.info(f"Login attempt - Email: {email}")
        logger.info(f"Request data: {data}")
        logger.info(f"Available users: {list(users.keys())}")
        
        # Check if user exists
        user = users.get(email)
        if not user:
            logger.warning(f"User not found: {email}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid credentials'
            }), 401
        
        # Verify password
        logger.debug(f"Checking password for user: {email}")
        if not check_password_hash(user['password'], password):
            logger.warning(f"Invalid password for user: {email}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid credentials'
            }), 401
        
        # Generate token
        token = jwt.encode({
            'email': user['email'],
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['SECRET_KEY'])
        
        logger.info(f"Successful login for: {email}")
        
        return jsonify({
            'status': 'success',
            'token': token,
            'user': {
                'email': user['email'],
                'name': user['name'],
                'userType': user['user_type']
            }
        })
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_ollama_available():
    """Check if Ollama service is available"""
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_random_response(category: str, **kwargs) -> str:
    """Get a random response from fallback responses"""
    responses = FALLBACK_RESPONSES.get(category, FALLBACK_RESPONSES["routine_advice"])
    response = random.choice(responses)
    return response.format(**kwargs)

def process_text_only(message: str) -> str:
    """Process text-only input using Tholini if available, otherwise use fallback"""
    try:
        # Check for Ollama availability
        if is_ollama_available():
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": "tholini",
                    "prompt": message,
                    "stream": False
                }
            )
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return get_random_response("routine_advice")

        # Fallback to random responses if Ollama is not available
        message_lower = message.lower()
        if any(word in message_lower for word in ['acne', 'pimple', 'rash', 'irritation', 'dry']):
            concern = next((word for word in ['acne', 'pimple', 'rash', 'irritation', 'dry'] 
                          if word in message_lower), 'skin concern')
            return get_random_response("skin_concern", concern=concern)
        elif any(word in message_lower for word in ['product', 'recommend', 'moisturizer', 'sunscreen']):
            product_type = next((word for word in ['moisturizer', 'sunscreen', 'cleanser', 'toner'] 
                               if word in message_lower), 'skincare product')
            return get_random_response("product_recommendation", product_type=product_type)
        else:
            return get_random_response("routine_advice")

    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        return get_random_response("routine_advice")

# Add model loading and image processing code
class ImageClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self):
        # Modify this part according to your model architecture
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'skin_condition_model.pth')
        
        # Initialize your model architecture
        model = models.resnet50(pretrained=False)
        num_classes = 5  # Adjust based on your number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Load the trained weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        model.to(self.device)
        return model
    
    def predict(self, image_path):
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                
            # Map prediction to class label
            class_labels = ['acne', 'eczema', 'psoriasis', 'rosacea', 'normal']  # Adjust based on your classes
            prediction = class_labels[predicted.item()]
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in image classification: {str(e)}")
            return None

# Initialize the classifier
classifier = ImageClassifier()

# Modify the process_image_and_text function
def process_image_and_text(image_path: str, message: str) -> str:
    """Process both image and text input"""
    try:
        if not image_path:
            return process_text_only(message)

        # Classify the image
        prediction = classifier.predict(image_path)
        
        if prediction:
            # Construct response based on classification
            response = f"I've analyzed the image and detected signs of {prediction}. "
            
            # Add specific advice based on the condition
            condition_advice = {
                'acne': "For acne, I recommend keeping the area clean and using products containing benzoyl peroxide or salicylic acid.",
                'eczema': "For eczema, it's important to keep the skin moisturized and avoid triggers. Consider using a gentle, fragrance-free moisturizer.",
                'psoriasis': "For psoriasis, regular moisturizing and avoiding triggers is key. Topical corticosteroids might be helpful.",
                'rosacea': "For rosacea, avoid triggers like sun exposure and spicy foods. Use gentle, non-irritating skincare products.",
                'normal': "Your skin appears healthy. Continue with your current skincare routine and remember to use sunscreen daily."
            }
            
            response += condition_advice.get(prediction, "")
            
            # Add disclaimer
            response += "\n\nPlease note: This is an AI-based assessment and should not replace professional medical advice. Consider consulting a dermatologist for a proper diagnosis."
            
            return response
        
        # Fallback to random responses if classification fails
        return get_random_response("image_analysis", concern="skin condition")

    except Exception as e:
        logger.error(f"Error processing image and text: {str(e)}")
        return get_random_response("image_analysis", concern="skin condition")

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        logger.info('Received chat request')
        
        # Get the message text
        message = request.form.get('message', '')
        logger.info(f'Message received: {message}')

        # Handle image upload if present
        image_path = None
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename:
                if allowed_file(file.filename):
                    # Create unique filename with timestamp
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = secure_filename(f"{timestamp}_{file.filename}")
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    
                    # Save the file
                    file.save(filepath)
                    image_path = filepath
                    logger.info(f'Image saved to: {filepath}')
                else:
                    return jsonify({
                        'status': 'error',
                        'message': 'Invalid file type'
                    }), 400

        # Process input and generate response
        if image_path:
            response = process_image_and_text(image_path, message)
            relative_path = f'/uploads/{os.path.basename(image_path)}'
        else:
            response = process_text_only(message)
            relative_path = None

        result = {
            'status': 'success',
            'response': response,
            'imagePath': relative_path
        }
        
        logger.info(f'Sending response: {result}')
        return jsonify(result)

    except Exception as e:
        logger.error(f'Error in chat endpoint: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Add routes for dermatologist-client interaction
@app.route('/api/consultations', methods=['GET'])
@token_required
def get_consultations(current_user):
    # Return relevant consultations based on user type
    return jsonify({
        'status': 'success',
        'consultations': chats.get(current_user['email'], [])
    })

@app.route('/api/test-llama', methods=['GET'])
def test_llama():
    """Test endpoint to check Ollama availability and Tholini model"""
    try:
        if not is_ollama_available():
            return jsonify({
                'status': 'warning',
                'message': 'Ollama is not available, using fallback responses'
            })
        
        # Test Tholini model specifically
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": "tholini",
                "prompt": "Hello",
                "stream": False
            }
        )
        
        if response.status_code == 200:
            return jsonify({
                'status': 'success',
                'message': 'Ollama and Tholini model are available'
            })
        else:
            return jsonify({
                'status': 'warning',
                'message': 'Ollama is available but Tholini model could not be loaded'
            })
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error testing Ollama: {str(e)}'
        })

# Enable CORS for file uploads
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Initialize test users when the app starts
init_test_users()

if __name__ == '__main__':
    # Log Ollama availability at startup
    if is_ollama_available():
        logger.info("Ollama service is available")
    else:
        logger.warning("Ollama service is not available, will use fallback responses")
    
    app.run(debug=True, port=5000) 