import os
import io
import re
import base64
import uuid
import numpy as np
import cv2
import tensorflow as tf
import keras
from keras.models import load_model
import matplotlib
matplotlib.use('Agg')  # Set Matplotlib to use a non-GUI backend
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, render_template, redirect, url_for, request, jsonify, send_file
from lime import lime_image
from skimage.segmentation import mark_boundaries
from io import BytesIO
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.scores import CategoricalScore



IMAGE_SIZE=160

app = Flask(__name__)

# make new directory called model and change your model name 
try:
    model = load_model('model/model.h5')
except TypeError as e:
    print("Error Config:", e.args)

label_dict={0:'Glioma Tumor', 1:'Meningioma Tumor',2:'No tumor',3:'Pituitary Tumor'}

# Get Last Convolutional Layer Name
def get_last_convolutional_layer(model):
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
    return last_conv_layer_name

# image process 
def preprocess(img):
	img = np.array(img)
	img = cv2.bilateralFilter(img, d=2, sigmaColor=50, sigmaSpace=50)
	# Apply a colormap (COLORMAP_BONE in this case)
	img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
	# Resize the image to the target size
	img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
	# Convert the image to a Keras-compatible array
	array = keras.utils.img_to_array(img)
	# Expand dimensions to match the input shape required by the model
	array = np.expand_dims(array, axis=0)
	return array

# Grad-CAM Function
def compute_gradcam(model, img_tensor, class_index):
    last_conv_layer_name = get_last_convolutional_layer(model)
    
    # Handle multiple outputs
    model_output = model.output[0] if isinstance(model.output, (list, tuple)) else model.output
    
    # Create Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output, 
            model_output
        ]
    )
    
    # Compute Grad-CAM
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    gradcam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
    for i, w in enumerate(pooled_grads):
        gradcam += w * conv_outputs[:, :, i]

    gradcam = np.maximum(gradcam, 0)
    gradcam = cv2.resize(gradcam, (IMAGE_SIZE,IMAGE_SIZE))
    gradcam -= gradcam.min()
    gradcam /= gradcam.max()
    return gradcam

# Saliency Map Function# Generate Saliency Map
def generate_saliency_map(model, input_tensor, class_index,smooth_samples,smooth_noise):
    score = CategoricalScore([class_index])  # Target the predicted class
    saliency = Saliency(model, clone=False)
    saliency_map = saliency(score, input_tensor, smooth_samples=smooth_samples, smooth_noise=smooth_noise)
    return normalize(saliency_map[0])

def create_visualization(overlay, lime_result, combined_img, saliency_map):
    """Generate visualization plots and save them to a buffer."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(overlay)
    axes[0, 0].set_title("Grad-CAM")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(lime_result)
    axes[0, 1].set_title("LIME Explanation")
    axes[0, 1].axis("off")
    axes[1, 0].imshow(combined_img)
    axes[1, 0].set_title("Combined Grad-CAM & LIME")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(saliency_map, cmap="hot")
    axes[1, 1].set_title("Saliency Map")
    axes[1, 1].axis("off")

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)  # Ensure the figure is closed after saving
    return buf

@app.route("/")
def index():
    upload_predefined_responses()
    return(render_template("index.html"))

@app.route("/predict", methods=["POST"])
def predict():
	data = request.get_json(force=True)
	data_to_print = data.copy()
	image = data['image']
	smooth_samples = data.get('smooth_samples', 50)
	smooth_noise = data.get('smooth_noise', 0.1)
	top_labels = data.get('top_labels', 5)
	hide_color = data.get('hide_color', 0)
	num_samples = data.get('num_samples', 1000)
	num_features = data.get('num_features', 5)
	min_weight = data.get('min_weight', 0.0)
    
	if 'image' in data_to_print:
		del data_to_print['image']

	for key, value in data_to_print.items():
		print(f"{key}: {value}")
    
	decoded = base64.b64decode(image)
	dataBytesIO=io.BytesIO(decoded)
	dataBytesIO.seek(0)
	image = Image.open(dataBytesIO)

	test_image=preprocess(image)

	input_tensor = test_image

	prediction = model.predict(test_image)


	result=np.argmax(prediction,axis=1)[0]
	accuracy=str(np.max(prediction,axis=1)[0])

	label=label_dict[result]

		# Grad-CAM
	predictions = model.predict(input_tensor)
	predicted_class = np.argmax(predictions[0])
	gradcam = compute_gradcam(model, input_tensor, predicted_class)
	# Handle invalid values and normalize gradcam to [0, 1]
	gradcam = np.nan_to_num(gradcam)  # Replace NaN and Inf with 0
	gradcam_min = gradcam.min()
	gradcam_max = gradcam.max()
	if gradcam_max - gradcam_min > 0:
		gradcam = (gradcam - gradcam_min) / (gradcam_max - gradcam_min)  # Normalize
	else:
		gradcam = np.zeros_like(gradcam)  # If gradcam is constant, set all values to 0
	# Convert to uint8 for color mapping
	overlay = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
	overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

	def predict_proba(test_image):
		return model.predict(test_image)

    # LIME
	explainer = lime_image.LimeImageExplainer()
	explanation = explainer.explain_instance(
		image=np.squeeze(input_tensor),
		classifier_fn=predict_proba,
		top_labels=top_labels,
		hide_color=hide_color,
        num_samples=num_samples
	)
	lime_img, mask = explanation.get_image_and_mask(
		label=predicted_class,
		positive_only=True,
		hide_rest=False,
		num_features=num_features,
		min_weight=min_weight
	)
	lime_result = mark_boundaries(lime_img / 255, mask)

	# Saliency Map
	saliency_map = generate_saliency_map(model, input_tensor, predicted_class, smooth_samples, smooth_noise)

	# Combine Grad-CAM and LIME
	gradcam_3d = np.repeat(gradcam[:, :, np.newaxis], 3, axis=2)
	combined_img = (0.5 * gradcam_3d + 0.5 * lime_img / 255).clip(0, 1)

	# Generate visualization
	buf = create_visualization(overlay, lime_result, combined_img, saliency_map)
	encoded_img = base64.b64encode(buf.getvalue()).decode('utf-8')
 
# Convert input_tensor to image
	input_tensor_img = np.squeeze(input_tensor)
	input_tensor_img = (input_tensor_img * 255).astype(np.uint8)
	submit_img_encoded = base64.b64encode(cv2.imencode('.png', input_tensor_img)[1]).decode('utf-8')
 
	# Generate visualization
	gradcam_img = (gradcam_3d * 255).astype(np.uint8)
	lime_explanation = (lime_img * 255).astype(np.uint8)
	lime_explanation_cam = (lime_result * 255).astype(np.uint8)
	combined_gradcam_lime_exp = (combined_img * 255).astype(np.uint8)
	saliency_map_img = (saliency_map * 255).astype(np.uint8)
 
	# Encode images
	gradcam_img_encoded = base64.b64encode(cv2.imencode('.png', gradcam_img)[1]).decode('utf-8')
	lime_explanation_encoded = base64.b64encode(cv2.imencode('.png', lime_explanation)[1]).decode('utf-8')
	combined_gradcam_lime_exp_encoded = base64.b64encode(cv2.imencode('.png', combined_gradcam_lime_exp)[1]).decode('utf-8')
	saliency_map_encoded = base64.b64encode(cv2.imencode('.png', saliency_map_img)[1]).decode('utf-8')
	lime_explanation_cam_encoded = base64.b64encode(cv2.imencode('.png', lime_explanation_cam)[1]).decode('utf-8')

 
	response = {
        'prediction': {'result': label, 'accuracy': accuracy},
        'user_id': str(uuid.uuid4()),
        'img_size': IMAGE_SIZE,
        'submit_img': submit_img_encoded,
        'gradcam_img': gradcam_img_encoded,
        'lime_explanation': lime_explanation_encoded,
        'combined_gradcam_lime_exp': combined_gradcam_lime_exp_encoded,
        'saliency_map': saliency_map_encoded,
        'lime_explanation_cam': lime_explanation_cam_encoded,
        'encoded_img': encoded_img
    }

	return jsonify(response)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200


# for RAG app

import requests
import psycopg2
import uuid
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re

# --- Configuration ---
# Load configuration from environment variables with default values
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")
PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = int(os.environ.get("PG_PORT", 5432))
PG_DATABASE = os.environ.get("PG_DATABASE", "mriDb")
PG_USER = os.environ.get("PG_USER", "user")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "secret")
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", 0.8))

# File to store the TF-IDF vectorizer
VECTORIZER_FILE = "tfidf_vectorizer.pkl"

# --- Database Connection and Table Setup ---
try:
    # Establish PostgreSQL connection
    pg_conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, database=PG_DATABASE, user=PG_USER, password=PG_PASSWORD
    )
    pg_cursor = pg_conn.cursor()
    print("Connected to PostgreSQL successfully!")

    # Create chat_history table if it doesn't exist (idempotent)
    pg_cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL,
            session_id UUID NOT NULL,
            message_text TEXT NOT NULL,
            sender VARCHAR(20) NOT NULL CHECK (sender IN ('user', 'bot')),
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_chat_history_user_session ON chat_history (user_id, session_id);
        CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history (session_id);
    """)
    pg_cursor.execute("""
        CREATE TABLE IF NOT EXISTS predefined_responses (
                id SERIAL PRIMARY KEY,
                query_pattern TEXT NOT NULL,
                response TEXT NOT NULL,
                UNIQUE(query_pattern)
            );
    """)
    pg_conn.commit()

except psycopg2.Error as e:
    print(f"Error connecting to/setting up PostgreSQL: {e}")
    exit()

# --- Load/Initialize TF-IDF Vectorizer and Cached Responses ---
try:
    # Try to load the vectorizer from file
    with open(VECTORIZER_FILE, "rb") as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    # If file not found, create a new vectorizer
    vectorizer = TfidfVectorizer()

cached_responses = {}
try:
    # Load cached responses from the database
    pg_cursor.execute("SELECT message_text, sender FROM chat_history WHERE sender = 'user'")
    user_messages = pg_cursor.fetchall()
    if user_messages:
        texts = [message[0] for message in user_messages]
        vectorizer.fit(texts) # Fit vectorizer to existing user queries
        with open(VECTORIZER_FILE, "wb") as f:
            pickle.dump(vectorizer, f) # Save the vectorizer
        for message in user_messages:
            pg_cursor.execute("SELECT message_text FROM chat_history WHERE sender = 'bot' AND message_text LIKE %s", ('%'+message[0]+'%',))
            bot_message = pg_cursor.fetchone()
            if bot_message:
                cached_responses[message[0]] = bot_message[0]
except psycopg2.Error as e:
    print(f"Error loading cached responses: {e}")



# --- Helper Functions ---
def get_context_from_db(query, session_id):
    """Retrieve context strictly related to specified tumor cases and warn for inappropriate content."""
    relevant_keywords = [
        "MRI", "brain tumor", "tumor diagnosis", "tumor recovery", "tumor identification",
        "oncologist", "tumor examination", "tumor education", "Glioma", "Meningioma", 
        "No tumor", "Pituitary", "tumor symptoms"
    ]
    inappropriate_keywords = ["sex", "child abuse", "violence", "harassment"]

    try:
        # Include the user's query as an additional filter
        sql = """
            SELECT message_text
            FROM chat_history
            WHERE (""" + " OR ".join([f"message_text ILIKE %s" for _ in relevant_keywords]) + """
            OR message_text ILIKE %s)
            ORDER BY timestamp DESC
            LIMIT 10;
        """
        # Add the user's query to the parameters
        parameters = tuple(f"%{keyword}%" for keyword in relevant_keywords) + (f"%{query}%",)
        pg_cursor.execute(sql, parameters)
        results = pg_cursor.fetchall()

        # Prepare context with oldest messages first
        context = "\n".join([f"{row[0]}" for row in results[::-1]])

        # Check for inappropriate content
        if any(re.search(rf"\b{keyword}\b", context, re.IGNORECASE) for keyword in inappropriate_keywords):
            return "Warning: The system detected inappropriate content in the context. Please ensure queries remain professional and relevant."

        return context
    except psycopg2.Error as e:
        print(f"Error querying PostgreSQL: {e}")
        return ""

def store_message(user_id, session_id, message_text, sender):
    """Stores a message in the chat history database."""
    try:
        sql = "INSERT INTO chat_history (user_id, session_id, message_text, sender) VALUES (%s, %s, %s, %s);"
        pg_cursor.execute(sql, (user_id, session_id, message_text, sender))
        pg_conn.commit()
    except psycopg2.Error as e:
        print(f"Error storing message: {e}")

def find_cached_response(query, similarity_threshold=SIMILARITY_THRESHOLD):
    """Finds a cached response for a similar query using TF-IDF and cosine similarity."""
    if not cached_responses:
        return None
    try:
        query_vec = vectorizer.transform([query]) # Vectorize the input query
        similarities = []
        for cached_query in cached_responses:
            cached_query_vec = vectorizer.transform([cached_query]) # Vectorize the cached query
            similarity = cosine_similarity(query_vec, cached_query_vec)[0][0] # Calculate cosine similarity
            similarities.append((cached_query, similarity))

        best_match, max_similarity = max(similarities, key=lambda item: item[1]) if similarities else (None, 0)
        if max_similarity >= similarity_threshold: # Check if similarity is above the threshold
            return cached_responses[best_match] # Return the cached response
        return None
    except Exception as e:
        print(f"Error during similarity check: {e}")
        return None

# --- Flask Routes ---
def upload_predefined_responses():
    """Uploads predefined responses to the database."""
    predefined_responses = [
        {"query_pattern": r"your name|what is your name|name", "response": "I am RadioAI, and I specialize in assisting with the identification and guidance related to 'Glioma,' 'Meningioma,' 'No tumor,' and 'Pituitary' using MRI images."},
        {"query_pattern": r"your purpose|what do you do", "response": "I assist in identifying and providing guidance on brain tumor cases using MRI images."},
    ]
    try:
        # Perform database operations
        for item in predefined_responses:
            sql = "INSERT INTO predefined_responses (query_pattern, response) VALUES (%s, %s) ON CONFLICT DO NOTHING;"
            pg_cursor.execute(sql, (item["query_pattern"], item["response"]))
        pg_conn.commit()
        print("Predefined responses uploaded successfully.")
    except psycopg2.Error as e:
        # Handle database errors
        print(f"Error uploading predefined responses: {e}")
        pg_conn.rollback()  # Rollback in case of an error



# RAG Process
@app.route('/rag', methods=['POST'])
def rag():
    """Handles RAG requests."""
    try:
        data = request.get_json()
        user_query = data.get('query')
        user_id = data.get('user_id')
        session_id = data.get('session_id')

        # Validate input
        if not user_query or not user_id:
            print("Invalid input.")
            return jsonify({"error": "Missing 'query' or 'user_id' parameter"}), 400

        # Check for predefined responses in the database
        sql = "SELECT response FROM predefined_responses WHERE %s ~* query_pattern LIMIT 1;"
        pg_cursor.execute(sql, (user_query,))
        predefined_response = pg_cursor.fetchone()
        if predefined_response:
            return jsonify({
                "response": predefined_response[0],
                "context_used": None,
                "session_id": session_id or str(uuid.uuid4()),
                "cached": False
            })

        # Generate a new session ID if one is not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Store the user's message in the database
        store_message(user_id, session_id, user_query, 'user')

        # Check for cached response
        cached_response = find_cached_response(user_query)
        if cached_response:
            print("Serving cached response.")
            return jsonify({
                "response": cached_response,
                "context_used": "Cached Response",
                "session_id": session_id,
                "cached": True
            })

        # If no cached response, retrieve context and call Ollama
        db_context = get_context_from_db(user_query, session_id)

        prompt = f"""Use the following context to answer the question at the end. 
        If you don't find the answer in the context, say "I couldn't find an answer in the provided context."
        Context:
        {db_context}
        Question: 
        {user_query}
        """
        print(f"Calling Ollama with prompt: {prompt}")
        ollama_request = {
            "model": "llama3.2-vision:latest",
            "prompt": prompt,
            "stream": False
        }
        ollama_response = requests.post(OLLAMA_API_URL, json=ollama_request)
        ollama_response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        ollama_data = ollama_response.json()
        response_text = ollama_data.get("response")

        # Store the bot's response
        store_message(user_id, session_id, response_text, 'bot')

        # Update the cache and vectorizer
        cached_responses[user_query] = response_text
        texts = list(cached_responses.keys())
        vectorizer.fit(texts)
        with open(VECTORIZER_FILE, "wb") as f:
            pickle.dump(vectorizer, f)

        return jsonify({
            "response": response_text,
            "context_used": db_context,
            "session_id": session_id,
            "cached": False
        })

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error communicating with Ollama: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
