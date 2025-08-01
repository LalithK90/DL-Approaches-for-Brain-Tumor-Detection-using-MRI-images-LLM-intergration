import base64
import ollama
import logging
import os
from typing import Optional
import chromadb
import uuid
import json
import datetime
from flask_login import current_user
# Setup basic logging for better feedback
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- ChromaDB Client Initialization ---
# Initialize the ChromaDB client once when the module is loaded for efficiency.
# This uses a persistent client, which saves the database to a local directory.
try:
    CHROMA_DATA_PATH = "chroma_data"
    # Ensure the directory exists
    os.makedirs(CHROMA_DATA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    # Get or create the collection. This is where your vectors will be stored.
    collection = client.get_or_create_collection("ollama_rag_store")
    logging.info(
        f"ChromaDB client initialized. Data will be stored in: '{CHROMA_DATA_PATH}'")
except Exception as e:
    logging.error(f"Fatal error initializing ChromaDB client: {e}")
    collection = None  # Set to None to prevent errors in subsequent calls


def _chunk_text(text: str, chunk_size: int = 4000, chunk_overlap: int = 200) -> list[str]:
    """Splits a long text into smaller chunks with overlap."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    # Ensure we process the whole text
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move start for the next chunk
        start += chunk_size - chunk_overlap
    return chunks


MAX_PROMPT_LENGTH = 8000  # A conservative character limit for prompts
MEDGEMMA_MODEL_NAME = "edwardlo12/medgemma-4b-it-Q4_K_M"
LAMMA_MODEL_NAME = 'llama3.2-vision:latest'
DEEPSEEK_MODEL_NAME = 'deepseek-r1:14b'
COMMON_PROMPT_MESSAGE = """**RadioAI**: As an expert oncologist, physician, and radiologist specializing in brain tumors, analyze this medical case and write a comprehensive report for medical practitioners at all levels. Include:

1. **Key Findings:** Detailed observations with precise anatomical references and measurements where possible. Highlight both primary and secondary features visible in the imaging.

2. **Interpretation:** In-depth clinical significance analysis with a ranked list of differential diagnoses. For each potential diagnosis, provide specific imaging markers that support or contradict it.

3. **Diagnosis Confidence Assessment:** Evaluate the certainty of your diagnosis using both qualitative reasoning and quantitative metrics. Explain which features are pathognomonic versus non-specific.

4. **Pathophysiology Insights:** Explain the underlying disease mechanisms, cellular characteristics, and progression patterns of the identified condition to enhance practitioner understanding.

5. **Treatment Considerations:** Evidence-based therapeutic options with their mechanisms of action, expected outcomes, and potential complications. Include recent advances in treatment modalities.

6. **Clinical Correlation:** Connect imaging findings to expected clinical presentations, laboratory findings, and genetic markers when applicable.

7. **Educational Pearls:** Include 3-5 high-yield learning points that highlight critical diagnostic features, common pitfalls, and distinguishing characteristics from similar conditions. Use clear comparisons and analogies where helpful.

8. **References to Literature:** Mention relevant clinical guidelines or landmark studies that inform the diagnostic and treatment approach.

Use precise medical terminology while providing explanations for complex concepts. Structure your report with clear headings and bullet points for readability. Your goal is to both accurately diagnose the current case and enhance the medical practitioner's knowledge for future cases."""


def store_in_vector_db(model_name: str, prompt: str, response: dict, image_path: Optional[str] = None, user_id: Optional[int] = None) -> Optional[str]:
    if not collection:
        logging.warning("ChromaDB collection not available. Skipping storage.")
        return None

    try:
        doc_id = str(uuid.uuid4())
        # Ensure document is a valid JSON string
        document_content = json.dumps({
            "prompt": prompt,
            "response": response
        })
        image_name = os.path.basename(image_path).replace(
            " ", "") if image_path else "none"
        metadata = {
            "model": model_name,
            "image": image_name,
            "user_id": str(user_id) if user_id else "unknown",
            "timestamp": datetime.datetime.now().isoformat()
        }

        # Add to ChromaDB collection
        collection.add(documents=[document_content],
                       metadatas=[metadata], ids=[doc_id])
        logging.info(f"Stored interaction in ChromaDB with ID: {doc_id}")
        return doc_id
    except Exception as e:
        logging.error(f"Failed to store interaction in ChromaDB: {e}")
        return None


def _query_vector_db(
    image_name: Optional[str] = None,
    user_id: Optional[int] = None,
    query_text: Optional[str] = None,
    n_results: int = 3
) -> Optional[list[str]]:

    if not collection:
        logging.warning("ChromaDB collection not available. Skipping query.")
        return None

    try:
        # Build the where clause
        where_clause = {}
        if image_name and user_id:
            where_clause = {
                "$and": [{"image": image_name}, {"user_id": str(user_id)}]}
        elif image_name:
            where_clause = {"image": image_name}
        elif user_id:
            where_clause = {"user_id": str(user_id)}

        logging.debug(
            f"Querying ChromaDB with where_clause: {where_clause}, query_text: {query_text}"
        )

        # Run the actual query or metadata-based get
        if query_text:
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
        else:
            results = collection.get(
                where=where_clause if where_clause else None,
                limit=n_results
            )

        if not results or not results.get('documents'):
            logging.info(
                f"No documents found in ChromaDB for image: '{image_name}', user_id: '{user_id}', query: '{query_text}'"
            )
            return None

        # Handle possible nested list structure
        docs_raw = results.get("documents", [[]])
        metas_raw = results.get("metadatas", [[]])

        docs = docs_raw[0] if isinstance(docs_raw[0], list) else docs_raw
        metas = metas_raw[0] if isinstance(metas_raw[0], list) else metas_raw

        parsed_strings = []

        for doc, meta in zip(docs, metas):
            if not isinstance(doc, str):
                logging.error(
                    f"Invalid document format, expected string, got: {type(doc)} - {doc}"
                )
                continue
            try:
                parsed_doc = json.loads(doc)
                parsed_doc['metadata'] = meta

                if "response" in parsed_doc:
                    parsed_strings.append(parsed_doc["response"])
                elif "prompt" in parsed_doc:
                    parsed_strings.append(parsed_doc["prompt"])
                else:
                    parsed_strings.append(json.dumps(parsed_doc))  # fallback

            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON: {doc} - Error: {e}")
                continue

        if not parsed_strings:
            logging.info(
                f"No valid documents parsed for image: '{image_name}', user_id: '{user_id}"
            )
            return None

        logging.info(
            f"Found {len(parsed_strings)} valid documents in ChromaDB for image: '{image_name}', user_id: '{user_id}'"
        )

        return parsed_strings

    except Exception as e:
        logging.error(f"An error occurred while querying ChromaDB: {e}")
        return None



def _call_ollama_model(model_name: str, message: str, image_path: Optional[str] = None) -> Optional[str]:
    try:
        messages = [{'role': 'user', 'content': message}]

        if image_path:
            if not os.path.exists(image_path):
                logging.error(f"Image not found at path: {image_path}")
                return None
            messages[0]['images'] = [image_path]

        logging.info(f"Calling model '{model_name}'...")
        response = ollama.chat(model=model_name, messages=messages)

        response_content = response['message']['content']
        logging.info(f"Model '{model_name}' responded successfully.")

        return response_content

    except Exception as e:
        logging.error(
            f"An error occurred while calling the Ollama model '{model_name}': {e}")
        return None


def get_text_reasoning(message: str, image_path: str) -> Optional[str]:

    logging.info("Performing RAG query for 'get_text_reasoning'...")
    user_id = current_user.id if current_user.is_authenticated else None

    retrieved_context = _query_vector_db(
        image_name=image_path, user_id=user_id, query_text=message, n_results=5)

    augmented_prompt = message
    if retrieved_context:
        # Each item in retrieved_context is a dict. We need to format it into a string.
        context_str = "\n\n".join([json.dumps(item)
                                  for item in retrieved_context])
        augmented_prompt = (
            f"{COMMON_PROMPT_MESSAGE}" +
            "--- CONTEXT ---\n"
            f"{context_str}\n"
            "--- END CONTEXT ---\n\n"
            f"USER QUESTION: {message}"
        )
        logging.info("Augmented prompt with retrieved context.")
    else:
        logging.info("No context found. Proceeding with the original prompt.")

    final_response = None
    # Check if the prompt is too long
    if len(augmented_prompt) <= MAX_PROMPT_LENGTH:
        logging.info("Prompt is within length limits. Calling model directly.")
        final_response = _call_ollama_model(
            model_name='deepseek-r1:14b', message=augmented_prompt)
    else:
        # Handle long prompt with chunking and refining
        logging.info(
            f"Prompt is too long ({len(augmented_prompt)} chars). Chunking and using refine strategy.")
        chunks = _chunk_text(augmented_prompt)

        refined_answer = ""

        # Process the first chunk
        first_chunk_prompt = (
            "This is the first part of a long document. Please summarize the key points from this section.\n\n"
            "--- DOCUMENT PART 1 ---\n"
            f"{chunks[0]}"
        )
        logging.info(f"Processing chunk 1 of {len(chunks)}...")
        # We don't store intermediate results of the refine process
        refined_answer = _call_ollama_model(
            model_name=DEEPSEEK_MODEL_NAME, message=first_chunk_prompt)

        if not refined_answer:
            logging.error("Failed to process the first chunk. Aborting.")
            return None

        # Process subsequent chunks
        for i, chunk in enumerate(chunks[1:], start=2):
            refine_prompt = (
                "You have provided a summary of the previous parts of a document. "
                "Your existing summary is:\n\n"
                "--- EXISTING SUMMARY ---\n"
                f"{refined_answer}\n\n"
                "--- END EXISTING SUMMARY ---\n\n"
                "Now, using the new information from the next part of the document below, refine and extend your summary. "
                "Provide a complete, updated summary that incorporates all information so far.\n\n"
                f"--- NEW DOCUMENT PART {i} ---\n"
                f"{chunk}"
            )
            logging.info(f"Processing chunk {i} of {len(chunks)}...")
            # We don't store intermediate results of the refine process
            refined_answer = _call_ollama_model(
                model_name=DEEPSEEK_MODEL_NAME, message=refine_prompt)
            if not refined_answer:
                logging.warning(
                    f"Failed to process chunk {i}. Continuing with previous summary.")
                # Continue with the last good answer

        final_response = refined_answer

    # Store the final result in the vector DB
    if final_response:
        store_in_vector_db(
            model_name='deepseek-r1:14b',
            prompt=message,  # Always store the original, clean user message
            response=final_response,
            image_path=image_path,
            user_id=current_user.id if current_user.is_authenticated else None
        )

    return final_response


def get_medical_report_from_image_medgemma(message: str, image_path: str) -> Optional[str]:
    prompt = (
        f"{COMMON_PROMPT_MESSAGE}" +
        f"USER QUESTION: {message}"
    )
    logging.info(f"Calling MedGemma model with image")
    response = _call_ollama_model(
        model_name=MEDGEMMA_MODEL_NAME, message=prompt, image_path=image_path)
    if response:
        store_in_vector_db(model_name=MEDGEMMA_MODEL_NAME,
                           prompt=message,
                           response=response,
                           image_path=image_path,
                           user_id=current_user.id if current_user.is_authenticated else None)
    return response


def get_medical_report_from_text_medgemma(message: str, image_path: str) -> Optional[str]:
    prompt = (
        f"{COMMON_PROMPT_MESSAGE}" +
        f"USER QUESTION: {message}"
    )
    logging.info(f"Calling MedGemma model with text")
    response = _call_ollama_model(
        model_name=MEDGEMMA_MODEL_NAME, message=prompt)
    if response:
        store_in_vector_db(model_name=MEDGEMMA_MODEL_NAME,
                           prompt=message,
                           response=response,
                           image_path=image_path,
                           user_id=current_user.id if current_user.is_authenticated else None)
    return response


def get_image_description_llama3_vision(message: str, image_path: str) -> Optional[str]:
    prompt = (
        f"{COMMON_PROMPT_MESSAGE}" +
        f"USER QUESTION: {message}"
    )
    logging.info(f"Calling Llama3.2 Vision model with image")
    response = _call_ollama_model(
        model_name=LAMMA_MODEL_NAME, message=prompt, image_path=image_path)
    if response:
        store_in_vector_db(model_name=LAMMA_MODEL_NAME,
                           prompt=message,
                           response=response,
                           image_path=image_path,
                           user_id=current_user.id if current_user.is_authenticated else None)
    return response
