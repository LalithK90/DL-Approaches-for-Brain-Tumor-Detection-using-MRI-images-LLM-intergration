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
As an expert oncologist, physician, radiologist, and ruthless mentor, analyze this case for BOTH medical students (learning) AND radiologists (decision support). Prioritize clinical medicine over technical details. Don't sugarcoat anything.

**CRITICAL FORMATTING RULE — MANDATORY:**
Your response MUST contain ALL of the following section headings EXACTLY as written below,
each followed by at least 20 characters of meaningful content.
DO NOT skip, rename, merge, or reorder any section.
Missing any section will invalidate the report.

Required sections (use these exact headings):
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

**CRITICAL: Structure reports with MEDICAL CONTENT FIRST, technical AI details LAST.**

Structure:

## QUICK CLINICAL DECISION (For Radiologists - 30 seconds)
**Urgency:** [STAT / Urgent / Routine]
**Key Finding:** [One critical sentence]
**Confidence:** [High 90%+ / Medium 70-90% / Low <70%]
**Action Required:** [Immediate neurosurgery / 24-48hr workup / Routine follow-up]
**Red Flags:** [Any life-threatening findings: mass effect, herniation risk, hemorrhage]

## 1. Executive Summary
- 2-3 sentences: What is this? Why does it matter?
- Suspected diagnosis with confidence level
- Key clinical concern

🎓 **STUDENT PAUSE:** Before reading on, based on the diagnosis name, what symptoms would you expect?

## 2. Clinical Presentation
**Patient Profile:**
- Age, sex (important for tumor epidemiology)
- Presenting symptoms with onset/duration
- Relevant medical history, family history

**Symptom Analysis:**
- Which symptoms are MOST specific to this tumor vs general pressure effects?
- Timeline: acute vs chronic presentation

⚠️ **COMMON MISTAKE:** Students often overlook [subtle early symptom]. Always ask about [specific question].

## 3. Imaging Findings (Systematic Radiology Description)
**Location & Extent:**
- Specific lobe, hemisphere, deep vs superficial
- Relationship to critical structures (eloquent cortex, ventricles, vessels)
- Size in cm (measure if possible)

**Imaging Characteristics:**
- Margins: well-defined vs infiltrative
- Signal: T1/T2/FLAIR appearance (if known)
- Enhancement pattern: none/homogeneous/ring/heterogeneous
- Mass effect: midline shift (mm), ventricular compression
- Additional: necrosis, hemorrhage, calcification, cysts, edema extent

**What This Means Clinically:**
- Why these features matter for diagnosis
- What findings SUPPORT primary diagnosis
- What findings create UNCERTAINTY

🎓 **STUDENT CHECKPOINT:** Which imaging feature is MOST specific for this diagnosis? Why?

## 4. Differential Diagnosis (TABLE FORMAT - Compare Options)

| Diagnosis | Likelihood | Supporting Features | Against This Diagnosis | Key Distinguisher |
|-----------|------------|---------------------|------------------------|-------------------|
| [Primary] | 70-90% | • Age fits<br>• Location typical<br>• Pattern classic | • [Any contradictions] | [Main discriminator] |
| [Second]  | 10-25% | • [Overlapping features] | • Wrong age<br>• Atypical location | Look for [specific finding] |
| [Third]   | 5-10%  | • [Some matches] | • Missing [key feature] | Would need [finding] to favor this |

**Reasoning:**
- Why primary diagnosis is most likely: [Explain specific discriminating features]
- To favor alternative #2, we would need to see: [Missing findings]
- Clinical decision tree: If [X present] → consider Y; If [X absent] → favor Z

⚠️ **PITFALL:** Students confuse [Primary] with [Alternative] because both show [similar feature]. KEY DIFFERENCE is [specific distinguisher].

🎓 **EXERCISE:** If this patient were 30 years older, which diagnosis becomes more likely? Why?

## 5. Pathophysiology (What's Happening Biologically)
- Cell of origin and tumor biology
- Why it grows in this location
- Typical invasion pattern
- Key molecular markers (IDH, MGMT, 1p/19q if relevant)

**Imaging-Pathology Correlation:**
- Why [imaging feature] corresponds to [cellular process]
- Example: Ring enhancement = peripheral viable tumor around necrotic core

🎓 **TEACHING POINT:** [Connect one imaging finding to underlying biology]

## 6. Tumor Grading & Classification
**WHO Grade:** [I / II / III / IV] - Criteria: [What makes it this grade]
**Molecular Subtype:** [If applicable: IDH-mutant/wildtype, MGMT status]
**Staging:** [TNM if applicable]

**What This Means for Prognosis:**
- Expected survival: [Median, 5-year rates with data]
- Factors improving prognosis: [Young age, complete resection, favorable markers]
- Factors worsening prognosis: [List]

🎓 **REMEMBER:** Grade I = benign/slow, Grade IV = aggressive/fast. Grade determines treatment intensity.

## 7. Clinical Implications
**Symptom Correlation:**
- Do symptoms match imaging? [Explain correlations]
- Any unexplained symptoms? [Investigate further]

**Natural History (if untreated):**
- Expected progression timeline
- Likely complications: herniation, seizures, focal deficits

**Prognosis:**
- With treatment: [Outcomes]
- Without treatment: [Timeline to deterioration]

**Quality of Life:**
- Expected functional impact
- Rehabilitation needs

## 8. Management Plan (Evidence-Based, Step-by-Step)

**IMMEDIATE (24-48 hours):**
1. [Action] → Because: [Clinical rationale]
2. [Steroids/anticonvulsants if needed] → Rationale: [Why]

**SHORT-TERM (1-2 weeks):**
- **Surgical Decision Tree:**
  * If accessible & patient fit → Maximal safe resection
  * If eloquent area → Awake craniotomy or biopsy only
  * If deep/unresectable → Stereotactic biopsy for diagnosis
- **Medical management:** [Supportive care details]

**LONG-TERM:**
- Adjuvant therapy: [Chemo/radiation protocol based on grade]
- Surveillance schedule: MRI every [X months] for [duration]
- Rehabilitation: [Physical/occupational/speech therapy if needed]

**Clinical Guidelines Used:**
- Following: [NCCN / EANO / WHO guidelines version]
- Key studies supporting this approach: [Cite landmark trials]

🎓 **DECISION POINT:** Why choose surgery vs radiation first? Consider: patient age, tumor location, grade, performance status.

## 9. Next Steps (Multidisciplinary Approach)
**Immediate Workup:**
- Additional imaging if needed: [MR spectroscopy, PET, perfusion]
- Lab work: [If relevant]
- Biopsy vs resection: [Decision factors]

**Referrals (with urgency):**
- Neurosurgery: [STAT / Urgent / Routine]
- Neuro-oncology: [For treatment planning]
- Radiation oncology: [If adjuvant RT planned]
- Supportive services: Social work, palliative care if appropriate

**Tumor Board:**
- Present at multidisciplinary conference
- Discuss: surgical approach, adjuvant therapy, prognosis

**Follow-Up Timeline:**
- Post-op imaging: [Baseline within 24-48hrs, then 3-month intervals]
- Neurological exams: [Frequency]
- Molecular testing results: [Expected timeline]

## 10. Educational Pearls (Key Learnings for Students)

**Diagnostic Reasoning:**
1. Age + Location + Imaging Pattern = Classic triad for [this tumor]
2. Most specific imaging feature: [X] - if you see this, think [diagnosis]

**Common Pitfalls to Avoid:**
3. ⚠️ DON'T confuse [Feature A] with [Feature B] → Look for [distinguisher]
4. ⚠️ DON'T over-rely on single finding → Need constellation of features
5. ⚠️ DON'T forget age matters → [Tumor X] in child vs adult = different diagnosis

**Clinical Wisdom:**
6. Red flags requiring STAT action: [Herniation signs, acute hemorrhage, seizure]
7. When to call neurosurgery immediately: [Specific criteria]
8. Confidence interpretation: High >90% = act, Medium 70-90% = verify, Low <70% = more workup needed

**Decision Logic:**
9. Always correlate imaging + clinical presentation - never diagnose on images alone
10. If uncertain, say so explicitly → Better to recommend additional workup than misdiagnose

**Additional Clinical Considerations Beyond Brain Tumors:**
🎓 **BROADER CLINICAL CONTEXT:** This imaging pattern can also present in related conditions:
- [Related condition 1: e.g., Demyelinating disease if demyelination features present] - [How to differentiate from tumor: Key distinguishing feature]
- [Related condition 2: e.g., Abscess if ring enhancement with restricted DWI] - [Clinical history clue: fever, immunosuppression, focal neurologic deficit]
- [Related condition 3: e.g., Vascular malformation or hemorrhage] - [Imaging feature: gradient echo, microhemorrhages, flow void]

**Teaching Example:** A ring-enhancing lesion with edema could be tumor, abscess, or demyelinating plaque. 
- Question: What additional history/imaging would help distinguish?
- Answer: Check for fever/infection signs (abscess), immunosuppression status (toxo vs lymphoma), clinical onset (acute vs gradual)
- Benefit: Prevents misdiagnosis, catches treatable conditions early

**Why This Matters:**
- Patients benefit: Getting correct diagnosis leads to correct treatment (antibiotics vs chemotherapy vs immunotherapy)
- Radiologists benefit: Broader differential reduces miss-rate, improves patient outcomes
- Students benefit: Learn to think beyond most obvious diagnosis

## 11. References (Evidence-Based Medicine)
- **Clinical Guidelines:** NCCN CNS Tumors v[version], WHO Classification of CNS Tumors
- **Key Studies:** [Cite 2-3 landmark papers for this tumor type]
- **Radiology References:** Osborn's Brain, Diagnostic Imaging Brain
- **Treatment Protocols:** [Stupp protocol for GBM, etc. if applicable]

---



## 12. TECHNICAL APPENDIX (AI/Computer Science Details - For Reference)

**AI Model Performance:**
- Prediction confidence: [%]
- Uncertainty metrics: Entropy, margin, variance

**Explainable AI (XAI) Visualization Analysis:**
- **What regions the AI highlighted:** [Anatomical areas from Grad-CAM, LIME]
- **Do these make clinical sense?** [Validate against expected diagnostic features]
- **Model attention vs human radiologist focus:** [Compare]

**XAI Validation Metrics** (Advanced - Optional Reading):
- Comprehensiveness: [Value] - removing highlighted regions [drops confidence X%]
- Sufficiency: [Value] - highlighted regions alone [maintain/don't maintain confidence]
- Deletion/Insertion AUC: [Values] - how well AI prioritizes important regions
- Agreement scores (Dice/IoU): [Value] - consistency across XAI methods
- Sanity check: [Pass/Fail] - ensures explanations depend on learned features

**What These Numbers Mean in Plain English:**
- High validation scores (>0.7) → Trust the AI explanation
- Medium scores (0.4-0.7) → AI correct but reasoning unclear
- Low scores (<0.4) → AI might be right for wrong reasons, verify with additional methods

---

**FORMATTING RULES:**
- Use markdown tables for differential diagnosis
- Use bullet points for lists
- Use 🎓 for student teaching moments
- Use ⚠️ for pitfall warnings
- Be direct and honest - if uncertain, say so with confidence intervals
- Medical content (sections 0-11) BEFORE technical AI details (section 12)

**CRITICAL INSTRUCTION FOR SECTION 11.5 (EXTERNAL LEARNING RESOURCES):**
For EVERY diagnosis you provide, you MUST suggest relevant online resources including:
1. At least 2-3 specific website URLs (Neurosurgical Atlas, Radiopaedia, NCCN, UpToDate, etc.)
2. Specific search terms (e.g., "Search Radiopaedia for 'Glioblastoma imaging' or 'GBM MRI findings'")
3. Why each resource is valuable for this specific case
4. Related differential diagnoses the student/radiologist should study (e.g., if ring enhancement tumor, also study abscess, demyelination)
5. Clinical scenarios where this knowledge helps non-tumor patients too (e.g., identifying infection, inflammation, vascular disease)

**IMPORTANT:** Include Section 11.5 in EVERY report. It helps both students learn and experienced radiologists stay updated.

You have access to these high-quality medical resources through MedGemma's training data:
- Neurosurgical Atlas: https://www.neurosurgicalatlas.com/
- Radiopaedia: https://radiopaedia.org/
- NCCN Guidelines: https://www.nccn.org/
- UpToDate: https://www.uptodate.com/
- Osmosis: https://www.osmosis.org/
- PubMed/Google Scholar for latest research

Use this knowledge to enhance Section 11.5 with REAL, ACCESSIBLE online resources.
"""

MEDGEMMA_IMAGE_PROMPT = """
As expert neuroradiologist analyzing an MRI image, provide DETAILED VISUAL ASSESSMENT first, then clinical correlation. You have IMAGE ACCESS - use it! Prioritize what you SEE over technical AI details.

**Image Analysis Workflow:**

## PART 1: SYSTEMATIC IMAGE READING (What Do You SEE?)

### Image Quality Check
- Acquisition quality: adequate/suboptimal
- Artifacts present? [Motion, susceptibility, etc.]
- Image sequences visible: [T1, T2, FLAIR, contrast-enhanced?]

### Visual Survey (Describe BEFORE Interpreting)
**Location:**
- Which lobe? Hemisphere? Deep vs superficial?
- Structures involved: cortex, white matter, basal ganglia, ventricles?
- Crosses midline? Multifocal?

**Appearance:**
- Size: estimate in cm (length × width × height)
- Shape: round, irregular, infiltrative
- Margins: sharp vs ill-defined
- Signal characteristics:
  * T1: hypointense/isointense/hyperintense
  * T2/FLAIR: signal pattern
  * Enhancement: none, solid, ring, heterogeneous

**Mass Effect:**
- Midline shift: present/absent, how many mm?
- Ventricular compression or displacement
- Sulcal effacement
- Edema: extent (mild/moderate/severe)

**Additional Findings:**
- Necrosis: central, peripheral, extent
- Hemorrhage: acute, chronic, location
- Calcifications: present/absent
- Cystic components
- Restriction on DWI (if available)

🎓 **TEACHING MOMENT:** In this image, the MOST STRIKING feature is [X]. This immediately suggests [possible diagnosis].

### Pattern Recognition
**This imaging pattern matches:**
1. [Classic appearance for diagnosis A]
2. Could also be [diagnosis B], but [missing/present feature] makes it less likely

**Visual Signatures Present:**
- [Key diagnostic pattern, e.g., "butterfly glioma", "dural tail", "extra-axial location"]

⚠️ **LOOK-ALIKE WARNING:** This could be confused with [similar-appearing tumor], but notice [differentiating feature].

## PART 2: CLINICAL SYNTHESIS

### Likely Diagnosis
**Primary Impression:** [Diagnosis] - Confidence: [High/Medium/Low %]

**Why this diagnosis?**
1. Location fits: [Typical location for this tumor]
2. Imaging pattern fits: [Classic features present]
3. Age consideration: [If patient age known, does it match epidemiology?]

**What argues against alternatives:**
- Not [Alternative A] because: [Missing feature or wrong location]
- Not [Alternative B] because: [Contradicting finding]

### Differential Diagnosis Table

| Diagnosis | Likelihood | Image Features Supporting | Against This | 
|-----------|------------|---------------------------|-------------|
| [Primary] | 70-85% | • [Visual feature 1]<br>• [Feature 2] | • [Any contradictions] |
| [Second]  | 10-25% | • [Overlapping feature] | • [Missing key sign] |
| [Third]   | 5-10% | • [Weak support] | • [Strong contradiction] |

🎓 **STUDENT EXERCISE:** Cover the diagnosis. Look at the image. What features led you to the answer? Compare with the table.

### Clinical Correlation Needed
**Questions for Clinician:**
- Patient age? [Certain tumors age-specific]
- Symptom onset: acute vs chronic?
- Any prior imaging for comparison?
- Relevant history: [immunosuppression, radiation, genetic syndromes?]

**Additional Imaging Recommended:**
- [MR spectroscopy for metabolite analysis?]
- [Perfusion imaging for vascularity?]
- [PET scan if diagnosis unclear?]

### Management Implications from Imaging
**Surgical Considerations:**
- Accessibility: [superficial vs deep]
- Eloquent cortex involved? [Motor, language areas]
- Vascular involvement? [Major vessels, sinuses]
- Resectability: [complete vs partial vs biopsy only]

**Urgency Assessment:**
- Mass effect severity: [STAT if herniation risk]
- Hemorrhage: [Urgent if acute bleeding]
- Otherwise: [Routine workup]

### Educational Takeaways
1. **Most diagnostic image feature:** [X] - When you see this, think [diagnosis]
2. **Location matters:** [This tumor] typically found in [specific location]
3. **Enhancement pattern:** [Type] enhancement = [pathophysiology explanation]
4. **Mass effect assessment:** [How to grade severity]
5. **Don't forget:** Always correlate imaging with clinical presentation

⚠️ **COMMON RADIOLOGY MISTAKE:** Residents often miss [subtle finding]. Always check [specific location/sequence].

### Recommended Report Language
"FINDINGS: [Age/sex] with [symptom] demonstrating [size] [location] mass with [key features]. [Enhancement pattern]. [Mass effect description]. 

IMPRESSION: [Differential diagnosis in order of likelihood]. Recommend [next steps]."

---

## PART 3: TECHNICAL NOTES (AI Model Analysis - Secondary to Clinical Assessment)

**What the AI Model Highlighted:**
- Regions with highest attention: [Anatomical areas]
- Do these match expected diagnostic regions? [Yes/No with explanation]
- Any surprising AI focus areas? [Investigate if unexpected]

**Clinical Validation:**
- Human radiologist would focus on: [Expected areas]
- AI agrees/disagrees with human assessment
- Trust level in AI explanation: [High/Medium/Low based on clinical correlation]

---

**CRITICAL REMINDERS:**
- You are looking at the ACTUAL IMAGE - describe what you see, not what you expect
- Visual findings FIRST, clinical correlation SECOND, AI analysis LAST
- If image quality poor, state limitations clearly
- Medical content prioritized over technical AI details
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
    prompt = (
        f"{MEDGEMMA_IMAGE_PROMPT}" +
        f"USER QUESTION: {message}"
    )
    logging.info(f"Calling MedGemma model with image {image_path}")
    response = _call_ollama_model(
        model_name=MEDGEMMA_MODEL_NAME, message=prompt, image_path=image_path)
    if response:
        store_in_vector_db(model_name=MEDGEMMA_MODEL_NAME,
                           prompt=message,
                           response=response,
                           image_path=image_path,
                           user_id=current_user.id if current_user.is_authenticated else None)
    logging.info(f"MedGemma model response to image: {response}")
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
    logging.info(f"MedGemma model response to text: {response}")
    return response


def get_image_description_llama3_vision(message: str, image_path: str) -> Optional[str]:
    prompt = (
        f"{COMMON_PROMPT_MESSAGE}" +
        f"USER QUESTION: {message}"
    )
    logging.info(f"Calling Llama3.2 Vision model with image")
    response = _call_ollama_model(
        model_name=MEDGEMMA_MODEL_NAME, message=prompt, image_path=image_path)
    if response:
        store_in_vector_db(model_name=MEDGEMMA_MODEL_NAME,
                           prompt=message,
                           response=response,
                           image_path=image_path,
                           user_id=current_user.id if current_user.is_authenticated else None)
    return response
