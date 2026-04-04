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
import requests
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv('api_key.env')
# Replace these w
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


MAX_PROMPT_LENGTH = 24000  # Default fallback value

# Model-specific context window limits (in characters)
# Optimized to reduce chunking and improve response time
MODEL_MAX_LENGTHS = {
    # ~6K tokens (75% of 8K capacity)
    "edwardlo12/medgemma-4b-it-Q4_K_M": 24000,
    # ~50K tokens (conservatively ~40% of 128K capacity)
    "llama3.2-vision:latest": 200000,
    # ~25K tokens (conservatively ~40% of 64K capacity)
    "deepseek-r1:14b": 100000,
    # OpenRouter DeepSeek (~25K tokens)
    "deepseek/deepseek-chat-v3-0324:free": 100000,
    "deepseek-r1-distill-llama-70b": 100000,  # Groq DeepSeek (~25K tokens)
    "qwen/qwen3-32b": 80000,  # Groq Qwen (~20K tokens, ~40% of 32K capacity)
}


def get_max_prompt_length(model_name: str) -> int:
    """
    Returns the maximum prompt length for a given model.
    Uses model-specific limits to optimize performance and reduce chunking.
    
    Args:
        model_name: Name of the model (e.g., 'deepseek-r1:14b', 'llama3.2-vision:latest')
    
    Returns:
        Maximum prompt length in characters
    """
    return MODEL_MAX_LENGTHS.get(model_name, MAX_PROMPT_LENGTH)

MEDGEMMA_MODEL_NAME = "dcarrascosa/medgemma-1.5-4b-it:F16"
DEEPSEEK_MODEL_NAME = 'deepseek-r1:14b'
COMMON_PROMPT_MESSAGE = """
You are an expert oncologist and neuroradiologist. Generate a structured brain tumor diagnostic report.

**MANDATORY: Output ALL 12 sections with EXACT headings below. Each section must have ≥20 characters of content. Do NOT skip, rename, or merge any section.**

## QUICK CLINICAL DECISION
## 1. Executive Summary
## 2. Clinical Presentation
## 3. Imaging Findings
## 4. Differential Diagnosis
## 5. Pathophysiology
## 6. Tumor Grading & Classification
## 7. Clinical Implications
## 8. Management Plan
## 9. Next Steps
## 10. Educational Pearls
## 11. References
## 12. TECHNICAL APPENDIX

---

**SECTION GUIDANCE:**

## QUICK CLINICAL DECISION
- **Urgency:** STAT (glioma/malignant) | Urgent (meningioma/pituitary) | Routine (no tumor)
- **Confidence:** High (≥90%) | Medium (70–89%) | Low (<70%) — use the DL score provided
- **Key Finding:** one-line diagnosis summary
- **Action Required:** immediate clinical action

## 1. Executive Summary — 2–3 sentences: diagnosis, confidence level, key clinical concern

## 2. Clinical Presentation — typical symptoms, patient profile, symptom onset/duration

## 3. Imaging Findings — location, size (cm), margins, enhancement pattern, mass effect (midline shift mm), edema, necrosis

## 4. Differential Diagnosis — markdown table: top 3 diagnoses with likelihood %, supporting features, distinguishing factors

## 5. Pathophysiology — cell origin, invasion pattern, molecular markers:
- Glioma: IDH mutation status, MGMT methylation, WHO grade classification
- Meningioma: dural attachment, grade, benign vs atypical
- Pituitary adenoma: hormone axis involvement, sellar/suprasellar extent
- No Tumor: describe normal brain parenchyma findings

## 6. Tumor Grading & Classification — WHO grade (I–IV), molecular subtype, expected survival data

## 7. Clinical Implications — symptom-imaging correlation, prognosis with/without treatment, quality of life impact

## 8. Management Plan — evidence-based:
- Glioma: maximal safe resection + temozolomide chemotherapy + radiotherapy (Stupp protocol)
- Meningioma: surgical resection planning, grade-based adjuvant therapy, seizure prophylaxis
- Pituitary: hormone evaluation (prolactin, cortisol, GH), vision assessment, sellar surgery options (transsphenoidal)
- No Tumor: routine follow-up, patient reassurance, no intervention required

## 9. Next Steps — referrals (neurosurgery/neuro-oncology urgency), additional imaging (MR spectroscopy, perfusion), tumor board

## 10. Educational Pearls — key diagnostic features, common pitfalls, clinical decision logic

## 11. References — NCCN CNS guidelines, WHO CNS tumor classification, 2–3 landmark studies

## 12. TECHNICAL APPENDIX — AI model prediction confidence, XAI metrics (Grad-CAM regions, comprehensiveness, sufficiency, Dice/IoU scores)

---

**AUDIENCE:** Write for two readers simultaneously:
- **Medical students** — explain clinical reasoning, define key terms (IDH, MGMT, WHO grade), include 1–2 teaching points per section (e.g., "Why does ring enhancement suggest high-grade glioma?")
- **Expert clinicians** — use precise terminology, evidence-based protocols, and searchable clinical phrases (tumor class, grade, molecular markers, urgency level) so the report supports specialist queries

**RULES:** Medical content (sections 1–11) BEFORE technical AI details (section 12). Use markdown tables for differential diagnosis. Be concise and clinically precise.
"""

MEDGEMMA_IMAGE_PROMPT = """
You are an expert neuroradiologist analyzing a brain MRI image. Describe what you observe visually, then provide clinical synthesis.

**MANDATORY: Your response MUST include ALL 12 sections with EXACT headings (≥20 characters each). Do NOT skip any.**

## QUICK CLINICAL DECISION
## 1. Executive Summary
## 2. Clinical Presentation
## 3. Imaging Findings
## 4. Differential Diagnosis
## 5. Pathophysiology
## 6. Tumor Grading & Classification
## 7. Clinical Implications
## 8. Management Plan
## 9. Next Steps
## 10. Educational Pearls
## 11. References
## 12. TECHNICAL APPENDIX

---

**IMAGE READING PROTOCOL:**

**Step 1 — Describe what you SEE:**
- Location: lobe, hemisphere, deep vs superficial
- Size: estimate in cm (length × width)
- Margins: sharp vs ill-defined vs infiltrative
- Signal: T1/T2/FLAIR characteristics
- Enhancement: none | solid | ring | heterogeneous
- Mass effect: midline shift (mm), ventricular compression, edema (mild/moderate/severe)
- Additional: necrosis, hemorrhage, calcification, cystic components

**Step 2 — Diagnosis and Confidence:**
- Primary impression with confidence: High (≥90%) | Medium (70–89%) | Low (<70%)
- Urgency: STAT (glioma/malignant) | Urgent (meningioma/pituitary) | Routine (no tumor)

**Step 3 — Use class-specific clinical terms:**
- Glioma: IDH mutation, MGMT methylation, WHO grade, maximal safe resection, temozolomide
- Meningioma: dural attachment, surgical resection, grade, seizure prophylaxis
- Pituitary adenoma: hormone dysfunction, vision impairment, sellar/suprasellar, prolactin, cortisol
- No Tumor: normal brain parenchyma, no evidence of mass, routine follow-up

**Step 4 — Differential diagnosis table** (top 3: likelihood %, supporting features, distinguishing factors)

**Step 5 — Management implications:** surgical accessibility, eloquent cortex involvement, urgency assessment, next referrals

**Step 6 — Educational pearl:** most diagnostic visual feature, one common pitfall to avoid

---

**AUDIENCE:** Write for two readers:
- **Students** — explain what each imaging feature means clinically (e.g., "ring enhancement indicates central necrosis typical of high-grade glioma"), include one teaching point per key finding
- **Expert clinicians** — use precise radiology terminology and searchable clinical phrases (tumor class, urgency, molecular markers) to support specialist queries and RAG retrieval

**FORMAT:** Markdown with ## headings. Visual findings FIRST (## 3. Imaging Findings), AI technical details LAST (## 12. TECHNICAL APPENDIX).
"""


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
        # Build the where clause - USER_ID IS PRIMARY FILTER (for privacy/isolation)
        where_clause = {}
        if user_id:
            where_clause = {"user_id": str(user_id)}
            if image_name:
                # If both provided, filter by both (user first for security)
                where_clause = {
                    "$and": [{"user_id": str(user_id)}, {"image": image_name}]
                }
        elif image_name:
            # Only use image_name if user_id not available
            where_clause = {"image": image_name}

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



def _call_ollama_model(model_name: str, message: str, image_path: Optional[str] = None, system: Optional[str] = None) -> Optional[str]:
    try:
        messages = []

        # System prompt sent separately — not prepended to user message.
        # This reduces per-request token prefill cost significantly.
        if system:
            messages.append({'role': 'system', 'content': system})

        user_msg = {'role': 'user', 'content': message}
        if image_path:
            if not os.path.exists(image_path):
                logging.error(f"Image not found at path: {image_path}")
                return None
            user_msg['images'] = [image_path]
        messages.append(user_msg)

        logging.info(f"Calling model '{model_name}'...")
        response = ollama.chat(model=model_name, messages=messages)

        response_content = response['message']['content']
        logging.info(f"Model '{model_name}' responded successfully.")

        return response_content

    except Exception as e:
        logging.error(
            f"An error occurred while calling the Ollama model '{model_name}': {e}")
        return None



OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY') 
GROQ_API_KEY = os.getenv('GROQ_API_KEY')


def get_text_reasoning(message: str, image_path: str) -> Optional[str]:

    logging.info(f"Performing RAG query for 'get_text_reasoning'...")
    user_id = current_user.id if current_user.is_authenticated else None

    retrieved_context = _query_vector_db(
        image_name=image_path, user_id=user_id, query_text=message, n_results=5)

    augmented_prompt = message
    if retrieved_context:
        # Each item in retrieved_context is a dict. We need to format it into a string.
        context_str = "\n\n".join([json.dumps(item) for item in retrieved_context])
        augmented_prompt = (
            "--- CONTEXT ---\n"
            f"{context_str}\n"
            "--- END CONTEXT ---\n\n"
            f"USER QUESTION: {message}"
        )
        logging.info("Augmented prompt with retrieved context.")
    else:
        logging.info("No context found. Proceeding with the original prompt.")

    # Determine which model will be used
    if OPENROUTER_API_KEY:
        model_name = "deepseek/deepseek-chat-v3-0324:free"
    else:
        model_name = DEEPSEEK_MODEL_NAME

    # Get model-specific max prompt length
    max_length = get_max_prompt_length(model_name)
    logging.info(
        f"Using model '{model_name}' with max prompt length: {max_length} chars")

    final_response = None
    # Check if the prompt is too long
    if len(augmented_prompt) <= max_length:
        logging.info("Prompt is within length limits. Calling model directly.")
        if OPENROUTER_API_KEY:
            final_response = _call_openrouter_model(augmented_prompt)
        else:
            final_response = _call_ollama_model(
                model_name=model_name, message=augmented_prompt)
    else:
        # Handle long prompt with chunking and refining
        logging.info(f"Prompt is too long ({len(augmented_prompt)} chars). Chunking and using refine strategy.")
        chunks = _chunk_text(augmented_prompt)

        refined_answer = ""

        # Process the first chunk
        first_chunk_prompt = (
            f"Summarize key points from first document part:\n--- PART 1 ---\n{chunks[0]}"
        )
        logging.info(f"Processing chunk 1 of {len(chunks)}...")
        if OPENROUTER_API_KEY:
            refined_answer = _call_openrouter_model(first_chunk_prompt)
        else:
            refined_answer = _call_ollama_model(
                model_name=model_name, message=first_chunk_prompt)

        if not refined_answer:
            logging.error("Failed to process the first chunk. Aborting.")
            return None

        # Process subsequent chunks
        for i, chunk in enumerate(chunks[1:], start=2):
            refine_prompt = (
                f"""Refine existing summary with new part:
                \n--- EXISTING ---\n{refined_answer}\n--- NEW PART {i} ---\n{chunk}\n
                Provide updated full summary."""
            )
            logging.info(f"Processing chunk {i} of {len(chunks)}...")

            if OPENROUTER_API_KEY:
                refined_answer = _call_openrouter_model(refine_prompt)
            else:
                refined_answer = _call_ollama_model(
                    model_name=model_name, message=refine_prompt)
            if not refined_answer:
                logging.warning(f"Failed to process chunk {i}. Continuing with previous summary.")
                # Continue with the last good answer

        final_response = refined_answer

    # Store the final result in the vector DB
    if final_response:
        model_name = 'deepseek/deepseek-chat-v3-0324:free' if OPENROUTER_API_KEY else 'deepseek-r1:14b'
        store_in_vector_db(
            model_name=model_name,
            prompt=message,  # Always store the original, clean user message
            response=final_response,
            image_path=image_path,
            user_id=current_user.id if current_user.is_authenticated else None
        )
    logging.info(f"Final response from model with deepseek: {final_response}")
    return final_response


OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')


def _call_openrouter_model(prompt: str) -> Optional[str]:
    logging.info("Calling OpenRouter model...")
    if not OPENROUTER_API_KEY:
        logging.error("OpenRouter API key is not set. Cannot call model.")
        return _call_groq_model(prompt)
    try:
        # Try OpenRouter first
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "deepseek/deepseek-chat-v3-0324:free",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
            })
        )
        response.raise_for_status()
        response_data = response.json()
        content = response_data.get("choices", [{}])[
            0].get("message", {}).get("content")
        if content:
            return content
        else:
            raise ValueError("OpenRouter response missing content.")

    except Exception as e:
        logging.warning(f"OpenRouter failed, falling back to Groq: {e}")
        return _call_groq_model(prompt)


def _call_groq_model(prompt: str) -> Optional[str]:
    logging.info("Calling Groq model...")
    if not GROQ_API_KEY:
        logging.error("GROQ API key is not set. Cannot call model.")
        return _call_ollama_model(model_name='deepseek-r1:14b', message=prompt)

    # List of models to try in order
    models_to_try = ["deepseek-r1-distill-llama-70b", "qwen/qwen3-32b"]

    for model_name in models_to_try:
        try:
            logging.info(f"Attempting Groq API with model: {model_name}")
            response = requests.post(
                url="https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "user",
                         "content": prompt}
                    ],
                    "temperature": 0.7
                }
            )
            response.raise_for_status()
            response_data = response.json()
            result = response_data.get("choices", [{}])[0].get(
                "message", {}).get("content")
            if result:
                logging.info(
                    f"Successfully received response from Groq model: {model_name}")
                return result

        except requests.exceptions.RequestException as e:
            logging.warning(
                f"Error calling Groq API with model {model_name}: {e}")
            if model_name == models_to_try[-1]:
                # Last model failed, fall back to Ollama
                logging.error(
                    f"All Groq models failed. Falling back to Ollama.")
                return _call_ollama_model(model_name='deepseek-r1:14b', message=prompt)
            # Try next model in list
            continue


def get_medical_report_from_image_medgemma(message: str, image_path: str) -> Optional[str]:
    logging.info(f"Calling MedGemma model with image {image_path}")
    response = _call_ollama_model(
        model_name=MEDGEMMA_MODEL_NAME,
        message=f"USER QUESTION: {message}",
        image_path=image_path,
        system=MEDGEMMA_IMAGE_PROMPT,
    )
    if response:
        store_in_vector_db(model_name=MEDGEMMA_MODEL_NAME,
                           prompt=message,
                           response=response,
                           image_path=image_path,
                           user_id=current_user.id if current_user.is_authenticated else None)
    logging.info(f"MedGemma model response to image: {response}")
    return response


def get_medical_report_from_text_medgemma(message: str, image_path: str) -> Optional[str]:
    logging.info(f"Calling MedGemma model with text")
    response = _call_ollama_model(
        model_name=MEDGEMMA_MODEL_NAME,
        message=f"USER QUESTION: {message}",
        system=COMMON_PROMPT_MESSAGE,
    )
    if response:
        store_in_vector_db(model_name=MEDGEMMA_MODEL_NAME,
                           prompt=message,
                           response=response,
                           image_path=image_path,
                           user_id=current_user.id if current_user.is_authenticated else None)
    logging.info(f"MedGemma model response to text: {response}")
    return response


def get_image_description_llama3_vision(message: str, image_path: str) -> Optional[str]:
    logging.info(f"Calling Llama3.2 Vision model with image")
    response = _call_ollama_model(
        model_name=MEDGEMMA_MODEL_NAME,
        message=f"USER QUESTION: {message}",
        image_path=image_path,
        system=COMMON_PROMPT_MESSAGE,
    )
    if response:
        store_in_vector_db(model_name=MEDGEMMA_MODEL_NAME,
                           prompt=message,
                           response=response,
                           image_path=image_path,
                           user_id=current_user.id if current_user.is_authenticated else None)
    return response
