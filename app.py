import os
import io
import re
import base64
import numpy as np
import cv2
import requests
import tensorflow as tf
import keras
from keras.models import load_model
import matplotlib
matplotlib.use('Agg')  # Set Matplotlib to use a non-GUI backend
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, render_template, redirect, url_for, request, jsonify, send_file, g
from lime import lime_image
from skimage.segmentation import mark_boundaries
from io import BytesIO
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.scores import CategoricalScore
# for rag app
import psycopg2
from sentence_transformers import SentenceTransformer
import psycopg2
import uuid
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


IMAGE_SIZE=160

app = Flask(__name__)

# make new directory called model and change your model name 
try:
    model = load_model('model/model.h5')
except TypeError as e:
    print("Error Config:", e.args)

label_dict={0:'Glioma', 1:'Meningioma',2:'No tumor',3:'Pituitary'}

# Get Last Convolutional Layer Name
def get_last_convolutional_layer(model):
    layer_name = None
    for layer in reversed(model.layers):
        if 'conv2d' in layer.name.lower():
            layer_name = layer.name
            break
    return layer_name



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
    
    # Check if the layer exists
    if not any(layer.name == last_conv_layer_name for layer in model.layers):
        raise ValueError(f"Layer {last_conv_layer_name} not found in the model.")

    # Create Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output[0] if isinstance(model.output, (list, tuple)) else model.output
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
    gradcam = cv2.resize(gradcam, (IMAGE_SIZE, IMAGE_SIZE))
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

@app.route("/predict", methods=["POST"])
def predict():
	data = request.get_json(force=True)
	image = data['image']
	smooth_samples = data.get('smooth_samples', 50)
	smooth_noise = data.get('smooth_noise', 0.1)
	top_labels = data.get('top_labels', 4)
	hide_color = data.get('hide_color', 0)
	num_samples = data.get('num_samples', 1000)
	num_features = data.get('num_features', 5)
	min_weight = data.get('min_weight', 0.0)
    
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
 
	# print(f'Prediction : {prediction}, Result : {result}, Accuracy : {accuracy}')
	# response = {'prediction': {'result' : label, 'accuracy' : accuracy}}
 
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
    
    
    # Convert input_tensor to image
	input_tensor_img = np.squeeze(input_tensor)
	input_tensor_img = (input_tensor_img * 255).astype(np.uint8)
	submit_img_encoded = base64.b64encode(cv2.imencode('.png', input_tensor_img)[1]).decode('utf-8')
    
	response = {
        'prediction': {'result': label, 'accuracy': accuracy},
        'img_size': IMAGE_SIZE,
        'user_id': str(uuid.uuid4()),
        'submit_img': submit_img_encoded,
        'gradcam_img': gradcam_img_encoded,
        'lime_explanation': lime_explanation_encoded,
        'combined_gradcam_lime_exp': combined_gradcam_lime_exp_encoded,
        'saliency_map': saliency_map_encoded,
        'lime_explanation_cam': lime_explanation_cam_encoded,
        'encoded_img': encoded_img
    }

	return jsonify(response)


# for RAG app

# # PostgreSQL connection
# def get_db_connection():
#     if 'db_conn' not in g:
#         g.db_conn = psycopg2.connect(
#             dbname="mriDb",
#             user="user",
#             password="secret",
#     host="localhost",
#     port="5432"
# )
#     return g.db_conn

# @app.teardown_appcontext
# def close_db_connection(exception):
#     db_conn = g.pop('db_conn', None)
#     if db_conn is not None:
#         db_conn.close()
        
# # SQL to create the table
# CREATE_TABLE_SQL = """
# CREATE TABLE IF NOT EXISTS documents (
#     id SERIAL PRIMARY KEY,
#     content TEXT NOT NULL,
#     embedding vector(768) NOT NULL
# );
# """

# def create_table():
#     print("Creating table...")
#     conn = get_db_connection()
#     try:
#         with conn.cursor() as cursor:
#         # Execute the SQL to create the table
#             cursor.execute(CREATE_TABLE_SQL)
            
#         conn.commit()  # Commit transaction
#         print("Table created successfully (if not exists).")
#     except Exception as e:
#         conn.rollback()  # Rollback transaction in case of error
#         print(f"Error creating table: {e}")

# # Load embedding model
# embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# # Generate embeddings
# def generate_embedding(text):
#     return embedding_model.encode(text).tolist()

# # Search the database
# def search_documents(query_embedding, top_k=5):
#     conn = get_db_connection()
#     try:
#         with conn.cursor() as cursor:
#             embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
#             cursor.execute("""
#                 SELECT id, content, embedding
#                 FROM documents
#                 ORDER BY embedding <-> %s::vector
#                 LIMIT %s;
#             """, (embedding_str, top_k))
#             return cursor.fetchall()
#     except Exception as e:
#         conn.rollback()  # Rollback transaction in case of error
#         raise e  


# # Route to add documents
# @app.route('/add_document', methods=['POST'])
# def add_document():
#     data = request.json
#     content = data.get('content')
#     if not content:
#         return jsonify({'error': 'Content is required'}), 400

#     embedding = generate_embedding(content)
#     conn = get_db_connection()
#     try:
#         with conn.cursor() as cursor:
#             cursor.execute("""
#             INSERT INTO documents (content, embedding)
#             VALUES (%s, %s)
#             RETURNING id;
#         """, (content, embedding))
#             doc_id = cursor.fetchone()[0]
#             print(f"Document added with ID: {doc_id}")
#         conn.commit()
#         return jsonify({'message': 'Document added', 'id': doc_id})
#     except Exception as e:
#         conn.rollback()  # Rollback transaction in case of error
#         raise e
    

# # Configuration
# # LLAMA_API_URL = "http://localhost:11434/api/chat"
# LLAMA_API_URL = "http://localhost:11434/api/generate"
# LLAMA_MODEL = "llama3.2-vision:latest"

# # Query Llama with Conversation History
# def query_llama_with_history(model, history):
#     payload = {
#         "model": model,
#         "prompt": model,
#         # "messages": history,
#         "stream": False
#     }
#     try:
#         # Send request to Llama API
#         response = requests.post(LLAMA_API_URL, json=payload)
        
#         # Validate and parse JSON response
#         try:
#             response_json = response.json()
#         except json.JSONDecodeError as e:
#             print(f"JSONDecodeError: {e}")
#             raise ValueError("Invalid JSON received from Llama API")

#         # Return parsed JSON
#         return response_json
#     except requests.RequestException as e:
#         print(f"RequestException: {e}")
#         raise ValueError("Failed to communicate with Llama API")



# # Flask Route: Query Llama
# @app.route('/query', methods=['POST'])
# def query_llama():
#     try:
#         # Get query from request
#         data = request.json
#         if not data or 'query' not in data:
#             return jsonify({"error": "Invalid input. 'query' is required."}), 400
        
#         # Build conversation history (example with one user input)
#         conversation_history = [
#             {"role": "user", "content": data['query']}
#         ]
        
#         # Call the Llama RAG API
#         llama_response = query_llama_with_history(LLAMA_MODEL, conversation_history)
        
#         # Extract response content
#         response_content = {
#             "query": data['query'],
#             "context": llama_response.get("context", "No context provided"),
#             "response": llama_response.get("message", {}).get("content", "No response provided")
#         }
#         return jsonify(response_content)
#     except ValueError as e:
#         return jsonify({"error": str(e)}), 500
#     except Exception as e:
#         print(f"Unexpected error: {e}")
#         return jsonify({"error": "An unexpected error occurred."}), 500





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
    """Retrieves previous messages from the current session for context."""
    try:
        sql = "SELECT message_text FROM chat_history WHERE session_id = %s ORDER BY timestamp DESC LIMIT 5;"
        pg_cursor.execute(sql, (session_id,))
        results = pg_cursor.fetchall()
        context = "\n".join([f"{row[0]}" for row in results[::-1]]) # Reverse to get correct order (oldest first)
        return context
    except psycopg2.Error as e:
        print(f"Error querying PostgreSQL: {e}")
        return ""

def store_message(user_id, session_id, message_text, sender):
    print(f"Storing message: {message_text}")
    """Stores a message in the chat history database."""
    try:
        sql = "INSERT INTO chat_history (user_id, session_id, message_text, sender) VALUES (%s, %s, %s, %s);"
        pg_cursor.execute(sql, (user_id, session_id, message_text, sender))
        pg_conn.commit()
        print("Message stored successfully.")
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

@app.route('/rag', methods=['POST'])
def rag():
    """Handles RAG requests."""
    try:
        data = request.get_json()
        user_query = data.get('query')
        user_id = data.get('user_id')
        session_id = data.get('session_id')
        print("User Query:", user_query)
        print("Session ID:", session_id)
        print("User ID:", user_id)
        # Validate input
        if not user_query or not user_id:
            print("Invalid input.")
            return jsonify({"error": "Missing 'query' or 'user_id' parameter"}), 400
        # Generate a new session ID if one is not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        # Store the user's message in the database
        store_message(user_id, session_id, user_query, 'user')

        # Check for cached response
        cached_response = find_cached_response(user_query)
        if cached_response:
            print("Serving cached response.")
            return jsonify({"response": cached_response, "context_used": "Cached Response", "session_id": session_id, "cached": True})

        # If no cached response, retrieve context and call Ollama
        db_context = get_context_from_db(user_query, session_id)

        prompt = f"""Use the following context to answer the question at the end. If you don't find the answer in the context, say "I couldn't find an answer in the provided context."

        Context:
        {db_context}

        Question: {user_query}
        """

        ollama_request = {
            "model": "llama2",
            "prompt": prompt,
            "stream": False
        }

        ollama_response = requests.post(OLLAMA_API_URL, json=ollama_request)
        ollama_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        ollama_data = ollama_response.json()
        print(ollama_data)
        response_text = ollama_data.get("response")

        # Store the bot's response
        store_message(user_id, session_id, response_text, 'bot')

        # Update the cache and vectorizer
        cached_responses[user_query] = response_text
        texts = list(cached_responses.keys())
        vectorizer.fit(texts)
        with open(VECTORIZER_FILE, "wb") as f:
            pickle.dump(vectorizer, f)

        return jsonify({"response": response_text, "context_used": db_context, "session_id": session_id, "cached": False})

    except requests.exceptions.RequestException as e:
        return jsonify



@app.route("/")
def index():
    # create_table()
    return(render_template("index.html"))

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(debug=True)
