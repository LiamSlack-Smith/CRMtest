# app.py
import os
import glob
import logging
import urllib.parse
import io
import json
import yaml
import re
import time
import datetime
import functools
# Removed crm_logic imports and threading
from flask import Flask, request, jsonify, render_template, abort, Response, stream_with_context
from werkzeug.utils import secure_filename

# --- REMOVED CRM Logic Imports ---
# Removed custom exceptions related to CRM

import ollama # Keep ollama check for now, though unused
import mammoth

# --- Google GenAI Imports & Setup ---
genai = None
types = None
google_genai_client = None

try:
    from google import genai
    from google.genai import types
    logging.info("Google GenAI SDK imported successfully.")

    # --- IMPORTANT: Google GenAI API Key ---
    # !! SECURITY WARNING: DO NOT HARDCODE KEYS IN PRODUCTION !!
    GOOGLE_GENAI_API_KEY = "AIzaSyCUZaott1f-ES2PZES7CZeh1pxYSbN74wg" # Example key - REPLACE

    if GOOGLE_GENAI_API_KEY == "YOUR_GOOGLE_GENAI_API_KEY" or not GOOGLE_GENAI_API_KEY:
        logging.warning("GOOGLE_GENAI_API_KEY not set. Google GenAI functionality will be disabled.")
        google_genai_client = None
    else:
        try:
            google_genai_client = genai.Client(api_key=GOOGLE_GENAI_API_KEY)
            logging.info("Google GenAI client initialized.")
        except Exception as e:
             logging.error(f"Failed to initialize Google GenAI client: {e}. Google GenAI functionality will be disabled.", exc_info=True)
             google_genai_client = None

except ImportError:
    logging.warning("google-genai not found. Install with 'pip install google-genai'. Google GenAI functionality will be disabled.")


# --- Configuration ---
GUIDES_DIR = "guides"
GEMINI_MODEL_NAME = 'gemini-2.5-flash-preview-04-17' # Or a model known to support streaming well
THEMES_FILE = "themes.yaml"
PROFILES_FILE = "profiles.yaml" # New profiles file
SUPPORT_LOG_FILE = "support_requests.log"
EMBEDDING_MODEL_NAME = 'text-embedding-004' # NEW: Embedding model name

# REMOVED CRM_DATA_FILE

# --- Global Theme Data & Loading ---
THEMES_DATA = {}
DEFAULT_THEME_NAME = 'default'
THEME_COLOR_KEYS = [
    'bg-color', 'secondary-bg', 'gradient-bg-start', 'gradient-bg-end', 'gradient-direction',
    'text-color', 'heading-color', 'muted-text-color',
    'accent-color', 'accent-hover', 'link-color', 'error-color',
    'border-color', 'input-bg', 'button-text-color',
    'user-msg-bg', 'assistant-msg-bg', 'message-text-color', 'message-border-color',
    'scrollbar-track-color', 'scrollbar-thumb-color',
    'menu-text-color', 'menu-hover-bg', 'menu-button-bg', 'menu-button-hover-bg',
    'menu-button-active-bg', 'menu-button-text-color', 'menu-button-active_text_color',
    # Removed CRM specific theme keys
    'viewer-html-bg', 'viewer-html-text', 'viewer-html-heading',
    'viewer-html-para', 'viewer-html-code-bg', 'viewer-html-code-text'
]
HARDCODED_FALLBACK_THEME = {
    'default': { 'name': "Default Dark (Fallback)", 'bg-color': "#1a1d24", 'secondary_bg': "#282c34", 'gradient_bg_start': "#1a1d24", 'gradient_bg_end': "#282c34", 'gradient_direction': "to bottom right", 'text_color': "#abb2bf", 'heading_color': "#ffffff", 'muted_text_color': "#5c6370", 'accent_color': "#61afef", 'accent_hover': "#528bce", 'link_color': "#61afef", 'error_color': "#e06c75", 'border_color': "#3b4048", 'input_bg': "#21252b", 'button_text_color': "#ffffff", 'user_msg_bg': "#0a3d62", 'assistant_msg_bg': "#3a3f4b", 'message_text_color': "#dcdfe4", 'message_border_color': "transparent", 'scrollbar_track_color': "#21252b", 'scrollbar_thumb_color': "#4b5263", 'menu_text_color': "#abb2bf", 'menu_hover_bg': "rgba(97, 175, 239, 0.1)", 'menu_button_bg': "transparent", 'menu_button_hover_bg': "rgba(255, 255, 255, 0.08)", 'menu_button_active_bg': "#61afef", 'menu_button_text_color': "#abb2bf", 'menu_button_active_text_color': "#ffffff", 'viewer_html_bg': "#f8f9fa", 'viewer_html_text': "#212529", 'viewer_html_heading': "#111", 'viewer_html_para': "#343a40", 'viewer_html_code_bg': "#e9ecef", 'viewer_html_code_text': "#333" }
}

def load_themes():
    global THEMES_DATA, DEFAULT_THEME_NAME
    try:
        with open(THEMES_FILE, 'r', encoding='utf-8') as f:
            loaded_data = yaml.safe_load(f)
        if not isinstance(loaded_data, dict) or not loaded_data:
            raise ValueError("Invalid structure")
        if DEFAULT_THEME_NAME not in loaded_data:
             first_key = next(iter(loaded_data), None)
             if not first_key:
                 raise ValueError("No themes defined")
             logging.warning(f"Default theme '{DEFAULT_THEME_NAME}' not found. Using '{first_key}'.")
             DEFAULT_THEME_NAME = first_key
        THEMES_DATA = loaded_data
        logging.info(f"Loaded {len(THEMES_DATA)} themes. Default: '{DEFAULT_THEME_NAME}'")
    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        logging.error(f"Error loading themes file '{THEMES_FILE}': {e}. Using hardcoded default.")
        THEMES_DATA = HARDCODED_FALLBACK_THEME.copy()
        DEFAULT_THEME_NAME = list(THEMES_DATA.keys())[0]
load_themes()

# --- Global Profile Data & Loading ---
PROFILES_DATA = {}
DEFAULT_PROFILE_KEY = 'default'

def load_profiles():
    global PROFILES_DATA, DEFAULT_PROFILE_KEY
    try:
        with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
            loaded_data = yaml.safe_load(f)
        if not isinstance(loaded_data, dict) or 'profiles' not in loaded_data or not isinstance(loaded_data['profiles'], dict):
            raise ValueError("Invalid structure in profiles.yaml. Expected top-level 'profiles' dictionary.")

        profiles = loaded_data['profiles']
        if DEFAULT_PROFILE_KEY not in profiles:
            first_key = next(iter(profiles), None)
            if not first_key:
                raise ValueError("No profiles defined in profiles.yaml")
            logging.warning(f"Default profile key '{DEFAULT_PROFILE_KEY}' not found in profiles.yaml. Using '{first_key}'.")
            DEFAULT_PROFILE_KEY = first_key

        # Validate profile structure (basic check)
        valid_profiles = {}
        for key, profile in profiles.items():
            if not isinstance(profile, dict) or 'name' not in profile or 'system_prompt' not in profile or 'available_guides' not in profile:
                 logging.warning(f"Profile '{key}' in profiles.yaml has invalid structure. Skipping.")
                 continue
            if not isinstance(profile['name'], str) or not isinstance(profile['system_prompt'], str) or not isinstance(profile['available_guides'], list):
                 logging.warning(f"Profile '{key}' in profiles.yaml has invalid field types. Skipping.")
                 continue
            valid_profiles[key] = profile # Add valid profile


        PROFILES_DATA = valid_profiles
        logging.info(f"Loaded {len(PROFILES_DATA)} profiles. Default: '{DEFAULT_PROFILE_KEY}'")

    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        logging.error(f"Error loading profiles file '{PROFILES_FILE}': {e}. Using hardcoded default profile.", exc_info=True)
        # Define a basic default profile if loading fails
        PROFILES_DATA = {
            'default': {
                'name': "General Assistant (Fallback)",
                'system_prompt': "You are a helpful AI assistant. Answer questions based on provided context.",
                'available_guides': [] # All guides
            }
        }
        DEFAULT_PROFILE_KEY = 'default'

load_profiles()


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- Initialize Flask App ---
app = Flask(__name__)

# --- RAG Data Storage (In-Memory) ---
# Structure: [{"text": "chunk text", "source": "filename", "embedding": [...float values...]}]
document_chunks = []
# Store a mapping of filename to its loaded status/error during embedding
loaded_guide_info = {}


# --- Helper Function for Gemini Calls with Retry ---
# CORRECTED: Use generate_content_stream for streaming calls
def generate_with_retry(client, model_name, contents, config=None, stream=False, max_retries=3, initial_delay=1):
    """Calls Google GenAI generate_content or embed_content with retry logic."""
    if client is None:
        raise ConnectionError("Google GenAI client is not initialized.")

    retries = 0
    delay = initial_delay
    last_exception = None

    while retries < max_retries:
        try:
            logging.debug(f"Attempt {retries + 1}/{max_retries} to call Gemini model '{model_name}' (stream={stream}).")

            kwargs = {
                'model': model_name,
                'contents': contents,
            }
            # Only add config if it's not None
            if config:
                 kwargs['config'] = config


            # Call the appropriate method based on whether streaming is requested
            if stream:
                # CORRECTED: Use generate_content_stream for streaming
                response = client.models.generate_content_stream(**kwargs)
            elif model_name == EMBEDDING_MODEL_NAME:
                 # Use embed_content for embeddings
                 response = client.models.embed_content(**kwargs)
            else:
                # Use generate_content for non-streaming text responses
                response = client.models.generate_content(**kwargs)


            logging.debug(f"Gemini call attempt {retries + 1} successful.")
            return response # Return the response (or iterator if stream=True)

        except Exception as e:
            last_exception = e
            retries += 1
            logging.warning(f"Gemini call failed (attempt {retries}/{max_retries}): {e}")
            if retries < max_retries:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2 # Exponential backoff
            else:
                logging.error(f"Gemini call failed after {max_retries} attempts.")

    # If loop finishes without success, raise the last exception
    raise ConnectionError(f"Failed to connect to Gemini after {max_retries} attempts: {last_exception}") from last_exception


# --- Helper to get safe file path ---
def get_safe_filepath(filename):
    """Validates filename and returns the absolute path if safe and exists."""
    if not filename or not isinstance(filename, str):
        logging.warning(f"Invalid filename type received: {type(filename)}")
        return None

    base_path = os.path.abspath(GUIDES_DIR)
    clean_filename = filename.strip()

    if '..' in clean_filename or '/' in clean_filename or '\\' in clean_filename:
         logging.warning(f"Potentially unsafe characters found in filename: '{filename}'")
         return None

    requested_path = os.path.normpath(os.path.join(base_path, clean_filename))

    if os.path.commonprefix([requested_path, base_path]) != base_path:
        logging.warning(f"Traversal attempt detected: {filename} -> {requested_path}")
        return None

    if not os.path.isfile(requested_path):
        logging.warning(f"File not found or is not a file at path: {requested_path} (requested: '{filename}')")
        return None

    if os.path.basename(requested_path).startswith('~$'):
         logging.debug(f"Skipping temporary file: {requested_path}")
         return None

    logging.debug(f"Validated path for '{filename}': {requested_path}")
    return requested_path

# --- RAG Implementation ---

# Simple chunking function (e.g., by paragraphs)
def chunk_text(text, source_filename, max_chars=1000):
    """Simple text chunking by paragraphs or sentences, with overlap."""
    chunks = []
    # Attempt to split by paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    current_chunk = ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 <= max_chars: # +2 for potential space/newline
            if current_chunk: current_chunk += "\n\n"
            current_chunk += paragraph
        else:
            if current_chunk:
                chunks.append({"text": current_chunk, "source": source_filename})
            current_chunk = paragraph # Start new chunk with the current paragraph

    if current_chunk: # Add the last chunk
        chunks.append({"text": current_chunk, "source": source_filename})

    # If chunks are still too large (e.g., very long paragraphs), fall back to splitting by sentences or fixed size
    final_chunks = []
    for chunk in chunks:
        if len(chunk["text"]) > max_chars:
            # Fallback: simple split by sentence (more complex regex needed for robust sentence splitting)
            sentences = re.split(r'(?<=[.!?])\s+', chunk["text"])
            sub_chunk = ""
            for sentence in sentences:
                 if len(sub_chunk) + len(sentence) + 1 <= max_chars:
                     if sub_chunk: sub_chunk += " "
                     sub_chunk += sentence
                 else:
                     if sub_chunk: final_chunks.append({"text": sub_chunk, "source": source_filename})
                     sub_chunk = sentence
            if sub_chunk: final_chunks.append({"text": sub_chunk, "source": source_filename})
        else:
            final_chunks.append(chunk)

    # Add overlap (simple approach: add last sentence/paragraph of previous chunk to next)
    overlapped_chunks = []
    for i in range(len(final_chunks)):
        chunk_text = final_chunks[i]["text"]
        overlap_text = ""
        if i > 0:
            # Get last sentence/paragraph of previous chunk (simplified)
            prev_text = final_chunks[i-1]["text"]
            overlap_candidate = prev_text.split('\n\n')[-1] # Last paragraph
            if len(overlap_candidate) > 50: # Or split by last sentence if paragraph too long
                 last_sentence_match = re.search(r'[^.!?]*[.!?]\s*$', overlap_candidate)
                 if last_sentence_match: overlap_candidate = last_sentence_match.group(0).strip()
                 else: overlap_candidate = overlap_candidate[-50:] # Fallback to last N chars

            if overlap_candidate and len(overlap_candidate) < max_chars * 0.2: # Limit overlap size
                 overlap_text = overlap_candidate + "\n\n"

        overlapped_chunks.append({
            "text": overlap_text + chunk_text,
            "source": final_chunks[i]["source"]
        })

    return overlapped_chunks if overlapped_chunks else [{"text": text, "source": source_filename}] # Ensure at least one chunk

async def load_and_embed_guides():
    """Loads text from guide files, chunks them, and generates embeddings."""
    global document_chunks, loaded_guide_info
    document_chunks = []
    loaded_guide_info = {} # Reset info on reload

    if google_genai_client is None:
        logging.warning("Google GenAI client not available. Skipping guide embedding.")
        return
    if not os.path.isdir(GUIDES_DIR):
        logging.warning(f"Guides directory '{GUIDES_DIR}' not found. Cannot load guides.")
        return

    logging.info(f"Loading and embedding guides from '{GUIDES_DIR}'...")
    files = [f for f in os.listdir(GUIDES_DIR) if os.path.isfile(os.path.join(GUIDES_DIR, f)) and not f.startswith('~$')]
    total_files = len(files)
    if total_files == 0:
        logging.info("No guide files found to embed.")
        return

    embedded_count = 0
    for filename in files:
        safe_path = get_safe_filepath(filename)
        if not safe_path:
            loaded_guide_info[filename] = {"status": "skipped", "reason": "Invalid path or temporary file"}
            continue

        content = None
        try:
            if filename.lower().endswith('.docx'):
                 with open(safe_path, "rb") as docx_file: result = mammoth.extract_raw_text(docx_file); content = result.value
            else:
                 try:
                     with open(safe_path, 'r', encoding='utf-8') as f: content = f.read()
                 except UnicodeDecodeError:
                     with open(safe_path, 'r', encoding='cp1252', errors='ignore') as f: content = f.read()
        except Exception as e:
            logging.error(f"Error reading file '{filename}' for embedding: {e}", exc_info=True)
            loaded_guide_info[filename] = {"status": "error", "reason": f"Read error: {e}"}
            continue

        if content and content.strip():
            chunks = chunk_text(content, filename)
            logging.info(f"Chunked '{filename}' into {len(chunks)} chunks.")
            chunk_texts = [chunk["text"] for chunk in chunks]
            try:
                # Updated: Use types.Content directly for batching
                content_list = [types.Content(parts=[types.Part(text=text)]) for text in chunk_texts]
                embedding_config = types.EmbedContentConfig(task_type="retrieval_document")

                embed_response = generate_with_retry(
                    client=google_genai_client,
                    model_name=EMBEDDING_MODEL_NAME,
                    contents=content_list, # Pass the list of Content objects
                    config=embedding_config,
                    stream=False # Embedding is not streamed
                )

                # --- CORRECTED EMBEDDING EXTRACTION for BATCH ---
                if embed_response and hasattr(embed_response, 'embeddings') and embed_response.embeddings and len(embed_response.embeddings) == len(chunks):
                    successful_chunk_embeddings = 0
                    for i, chunk in enumerate(chunks):
                        # Check if the individual embedding object has 'values'
                        if hasattr(embed_response.embeddings[i], 'values') and embed_response.embeddings[i].values:
                            chunk["embedding"] = embed_response.embeddings[i].values
                            document_chunks.append(chunk)
                            successful_chunk_embeddings += 1
                        else:
                            logging.warning(f"Chunk {i+1} from '{filename}' did not contain embedding values.")
                    if successful_chunk_embeddings > 0:
                        embedded_count += 1
                        loaded_guide_info[filename] = {"status": "success", "chunk_count": successful_chunk_embeddings}
                        logging.info(f"Successfully processed embeddings for {successful_chunk_embeddings}/{len(chunks)} chunks from '{filename}'.")
                    else:
                         loaded_guide_info[filename] = {"status": "warning", "reason": "No embeddings generated for any chunks"}
                         logging.warning(f"No embeddings generated for any chunks from '{filename}'.")
                # --- END CORRECTION ---
                else:
                    loaded_guide_info[filename] = {"status": "warning", "reason": "Embedding response had unexpected structure or count"}
                    logging.warning(f"Embedding response for '{filename}' had unexpected structure or count: {embed_response}")

            except (ConnectionError, Exception) as e:
                logging.error(f"Error generating embeddings for '{filename}' (retries exhausted?): {e}", exc_info=True)
                loaded_guide_info[filename] = {"status": "error", "reason": f"Embedding error: {e}"}

        else:
            logging.warning(f"Skipping empty or whitespace-only file: '{filename}'.")
            loaded_guide_info[filename] = {"status": "skipped", "reason": "Empty content"}


    logging.info(f"Finished embedding guides. Total files processed: {total_files}, files with >=1 chunk embedded: {embedded_count}, total chunks embedded: {len(document_chunks)}.")
    logging.debug(f"Guide loading summary: {loaded_guide_info}")


# --- Similarity Search Function ---
def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude1 = sum(v**2 for v in vec1)**0.5
    magnitude2 = sum(v**2 for v in vec2)**0.5
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

def find_relevant_chunks(query_embedding, top_k=5, allowed_guides=None):
    """Finds the top_k most similar chunks to the query embedding,
       optionally filtering by allowed guide filenames."""
    if not query_embedding or not document_chunks:
        return []

    # Filter chunks based on allowed_guides if the list is not empty
    if allowed_guides is not None and len(allowed_guides) > 0:
        filterable_chunks = [
            chunk for chunk in document_chunks
            if chunk.get("source") in allowed_guides
        ]
        logging.debug(f"Filtered RAG search to {len(filterable_chunks)} chunks based on allowed guides.")
    else:
        # If allowed_guides is None or empty, use all chunks
        filterable_chunks = document_chunks
        logging.debug("RAG search using all available chunks.")


    similarities = []
    for chunk in filterable_chunks:
        if "embedding" in chunk and chunk["embedding"]:
            similarity = cosine_similarity(query_embedding, chunk["embedding"])
            similarities.append((similarity, chunk))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[0], reverse=True)

    # Return the top_k chunks (excluding similarity score)
    return [chunk for similarity, chunk in similarities[:top_k]]


# --- Tool Definitions and Prompts ---
# No changes needed in prompts
def construct_gemini_tool_selection_prompt(query):
    available_files_str = "None"
    try:
        if os.path.isdir(GUIDES_DIR):
            # Use loaded_guide_info keys for the list of available files
            available_files = [fname for fname, info in loaded_guide_info.items() if info.get('status') != 'error']
            if available_files: available_files_str = ", ".join([f"`{f}`" for f in available_files])
        else: logging.warning(f"Guides directory '{GUIDES_DIR}' not found."); available_files_str = "(Guides directory not found)"
    except Exception as e: logging.error(f"Error listing files in {GUIDES_DIR}: {e}"); available_files_str = "(Error listing files)"

    prompt = f"""
You are an AI assistant that selects the appropriate tool and arguments to respond to a user query. Your primary goal is to provide accurate answers based on available internal documents when possible, or facilitate user actions like theme creation or support requests.
Based on the user's query, you MUST choose one of the following tools and respond ONLY with a valid JSON object matching the tool's schema. Do NOT add any explanation before or after the JSON.

Available Tools:

1.  **`read_files`**: Use this tool **FIRST** if the user's question likely requires information contained within specific internal guide files to provide an accurate answer (e.g., "how do I...", "explain concept X..."). Analyze the query and choose the most relevant-sounding file(s) from the list provided below. If unsure which file, select the one(s) that seem most likely related. Use 'answer_question' only when you've sourced the answer from files.
    Schema:
    ```json
    {{
      "tool": "read_files",
      "arguments": {{
        "filenames": ["filename1.ext", "filename2.ext"] // List of relevant filenames from the available list
      }}
    }}
    ```

2.  **`answer_question`**: Use this tool ONLY if the user's question is very general (e.g., "hello", "what can you do?"), does not seem to require specific internal knowledge, OR if you previously used `read_files` and the information needed should now be available in the context passed to the next step. Do NOT use this if the question likely requires info from the available files unless `read_files` was already used or no files seem relevant.
    Schema:
    ```json
    {{
      "tool": "answer_question",
      "arguments": {{}} // No arguments needed for this tool selection step
    }}
    ```

3.  **`create_theme`**: Use this tool ONLY if the user explicitly asks to create a new visual theme (e.g., "create a solarized theme", "make a theme with green accents").
    Schema:
    ```json
    {{
      "tool": "create_theme",
      "arguments": {{}} // No arguments needed here, another AI call will generate them
    }}
    ```

4.  **`request_support`**: Use this tool ONLY if the user is explicitly asking for help *with the application itself*, reporting a bug, requesting a feature, or indicating they are stuck due to a technical problem (e.g., "the save button isn't working", "I need help logging in", "can you add X feature?", "getting an error message"). Do NOT use this for questions about *how to use* the application features (use `read_files` or `answer_question` for those). Capture the user's description of the issue.
    Schema:
    ```json
    {{
      "tool": "request_support",
      "arguments": {{
        "request_description": "A summary of the user's problem or request for help." // Extract this from the user's query
      }}
    }}
    ```

**Analysis Steps:**
1. Analyze the user's query: `{query}`
2. Determine the user's intent: Answer question needing file info? Answer general question? Create theme? Request technical support/report issue?
3. **Prioritize `read_files`:** If the question asks about procedures, specific functionalities, how-tos, or mentions topics likely covered in guides, choose `read_files` and identify relevant filename(s).
4. If the intent is clearly theme creation, choose `create_theme`.
5. **If the intent is clearly a request for technical help with the application or reporting an issue/bug**, choose `request_support` and extract the description of the problem into `request_description`.
6. If the question is general conversation OR no available files seem relevant AND it's not a support request, choose `answer_question`.
7. Output ONLY the chosen tool's JSON structure with necessary arguments.

**Available Guide Files:**
{available_files_str}

**User Query:** {query}

**Chosen Tool (JSON Output Only):**
"""
    return prompt

# CORRECTED: construct_google_answer_prompt now takes system_prompt as an argument
# REINFORCED: Added explicit Markdown formatting instruction back into the main prompt body
def construct_google_answer_prompt(query, file_content_map, system_prompt):
    """Constructs the prompt for Google GenAI to generate the answer (now expects streamed text, not JSON).
       Includes cited sources IN THE PROMPT, not expecting them back in JSON."""

    context_str = "No specific file content was provided or read for this query."
    filenames_provided = list(file_content_map.keys())
    if file_content_map:
        context_parts = []
        # Use the generic key from the RAG result
        context_key = next(iter(file_content_map), None)
        if context_key:
            content = file_content_map[context_key]
            max_len = 10000 # Limit context length per file
            truncated_content = content[:max_len] + ("..." if len(content) > max_len else "")
            context_parts.append(f"--- START OF RELEVANT CONTEXT ---\n{truncated_content}\n--- END OF RELEVANT CONTEXT ---")
        context_str = "\n\n".join(context_parts)

    prompt = f"""
{system_prompt}

Retrieved Information (Context):
{context_str}

User Question: {query}

Instructions for Answer:
- Answer the user's question based *only* on the provided context above.
- If the context doesn't contain the answer, state that the information is unavailable based on the provided documents.
- **Format the entire answer using Markdown** (e.g., use headings, lists, bold text, inline markdown and any other markdown where appropriate). Ensure that the markdown is extensive and professional.
- Respond directly with the answer text. Do not include introductory phrases like "Here is the answer:" unless it flows naturally.

Answer (Markdown Formatted Text Only):
"""
    return prompt

def construct_google_theme_generation_prompt(user_query):
    """Constructs the prompt for Google GenAI to generate theme JSON."""
    theme_keys_string = ", ".join([f"`{key}`" for key in THEME_COLOR_KEYS])
    prompt = f"""
You are an AI assistant skilled in UI/UX design and color theory. Your task is to generate a new theme based on the user's request.
Output ONLY a valid JSON object containing the key "arguments" which itself contains ALL the required theme properties with appropriate values based on the user's description.
Do NOT add any explanation before or after the JSON.

User's Theme Request: "{user_query}"

Required Theme Properties (provide a value for ALL of these within the 'arguments' object):
{theme_keys_string}

Guidelines:
- Generate a creative and descriptive `theme_name`.
- Ensure all color values are valid 6-digit hex codes (`#xxxxxx`) unless the key specifically allows `rgba()` or `transparent`.
- Choose colors that are aesthetically pleasing and provide good contrast for readability based on the user's request.
- For `gradient-direction`, use standard CSS values.
- For hover/transparent backgrounds, use `rgba()` with low alpha or `transparent`.
- Ensure all {len(THEME_COLOR_KEYS)} keys are present in the JSON output under the 'arguments' key.

JSON Output Only (containing 'arguments' key):
"""
    return prompt


# --- Tool Execution Functions (AI/File Related Only) ---
def is_valid_hex_color(color_code):
    if not isinstance(color_code, str):
        return False
    return bool(re.match(r'^#[0-9a-fA-F]{6}$', color_code))

def is_valid_rgba_or_transparent(color_code):
     if not isinstance(color_code, str):
         return False
     if color_code.lower() == 'transparent':
         return True
     return bool(re.match(r'^rgba?\(\s*\d+%?\s*,\s*\d+%?\s*,\s*\d+%?\s*(?:,\s*(?:0|1|0?\.\d+)\s*)?\)$', color_code.lower()))

def handle_create_theme(arguments):
    """Validates theme args (generated by Google GenAI), updates YAML, reloads themes."""
    global THEMES_DATA
    if not isinstance(arguments, dict):
        return {"status": "error", "message": "Invalid args format received from GenAI for theme creation."}
    theme_name = arguments.get("theme_name", f"Custom Theme {len(THEMES_DATA) + 1}")
    if not isinstance(theme_name, str) or len(theme_name) > 50:
        theme_name = f"Custom Theme {len(THEMES_DATA) + 1}"
    theme_key = secure_filename(theme_name.lower().replace(' ', '_')) or f"custom_theme_{int(time.time())}"
    if theme_key in THEMES_DATA:
        theme_key = f"{theme_key}_{int(time.time())}"
    new_theme_data = {"name": theme_name}
    missing_keys, invalid_values = [], []
    for key in THEME_COLOR_KEYS:
        value = arguments.get(key)
        if value is None:
            missing_keys.append(key)
            continue
        is_valid = False
        if key in ['menu-hover-bg', 'message-border-color', 'menu-button-bg', 'menu-button-hover-bg']:
            is_valid = is_valid_rgba_or_transparent(value) or is_valid_hex_color(value)
        elif key == 'gradient-direction':
            is_valid = isinstance(value, str) and value.strip() != ''
        else:
            is_valid = is_valid_hex_color(value)
        if is_valid:
            new_theme_data[key] = value
        else:
            invalid_values.append(f"{key} ('{value}')")
    if missing_keys:
        return {"status": "error", "message": f"Google GenAI failed to provide required theme keys: {', '.join(missing_keys)}"}
    if invalid_values:
        return {"status": "error", "message": f"Google GenAI provided invalid values for theme keys: {', '.join(invalid_values)}"}
    try:
        current_themes = {}
        if os.path.exists(THEMES_FILE):
             with open(THEMES_FILE, 'r', encoding='utf-8') as f_read:
                 loaded_content = yaml.safe_load(f_read);
                 if isinstance(loaded_content, dict):
                    current_themes = loaded_content
                 elif loaded_content is not None:
                    logging.warning(f"Overwriting non-dict content in {THEMES_FILE}.")
        current_themes[theme_key] = new_theme_data
        with open(THEMES_FILE, 'w', encoding='utf-8') as f_write:
            yaml.dump(current_themes, f_write, default_flow_style=False, sort_keys=False, allow_unicode=True)
        logging.info(f"Added theme '{theme_name}' (key: '{theme_key}')")
        load_themes()
        return {"status": "success", "message": f"Theme '{theme_name}' created!", "new_theme_key": theme_key, "themes": THEMES_DATA}
    except Exception as e:
        logging.error(f"Failed to write theme: {e}", exc_info=True)
        # Use a generic error type here if StorageError was removed
        return {"status": "error", "message": "Failed to save the new theme."}

def handle_read_files(arguments):
    """Reads content of requested files using get_safe_filepath. (Used for RAG context)."""
    # This function remains as it's needed by the RAG flow, even though the LLM
    # doesn't directly "read" files in the old sense. It identifies files for RAG.
    if not isinstance(arguments, dict):
        return None, {"error": "Invalid arguments format for read_files."}, []

    filenames = arguments.get("filenames")
    if not filenames or not isinstance(filenames, list):
        return None, {"error": "Missing or invalid 'filenames' list for read_files."}, []

    # The RAG implementation now handles the actual reading and embedding.
    # This function, when called by the LLM tool selection, now primarily serves
    # as a signal that RAG should be attempted, possibly using these filenames
    # as a hint (though the current RAG uses similarity search on the query).
    # We'll return an empty content map but log the filenames.
    logging.info(f"LLM suggested reading files (for RAG context): {filenames}")
    errors = []
    valid_filenames_for_rag_hint = []
    for fname in filenames:
        if isinstance(fname, str):
            valid_filenames_for_rag_hint.append(fname)
        else:
             errors.append(f"Invalid filename type: {type(fname)}")

    # Return empty content, no errors (unless filename type was wrong),
    # and the list of filenames the LLM thought were relevant.
    return {}, {"errors": errors} if errors else None, valid_filenames_for_rag_hint


def handle_request_support(arguments):
    """Logs the user's support request to a file."""
    if not isinstance(arguments, dict):
        return {"status": "error", "message": "Invalid arguments format received for support request."}

    request_description = arguments.get("request_description")
    if not request_description or not isinstance(request_description, str) or not request_description.strip():
        logging.warning("Received support request tool call without a valid 'request_description'.")
        return {"status": "error", "message": "Could not log support request. Please describe the issue clearly."}

    request_description = request_description.strip()
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
    log_entry = f"{timestamp} - Request: {request_description}\n"

    try:
        with open(SUPPORT_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)

        logging.info(f"Logged support request: {request_description}")
        return {
            "status": "success",
            "message": "Your support request has been logged successfully. Our team will review it shortly."
        }
    except IOError as e:
        logging.error(f"Failed to write to support log file '{SUPPORT_LOG_FILE}': {e}", exc_info=True)
        # Use generic error if StorageError removed
        return {"status": "error", "message": "Sorry, there was an internal error logging your support request."}
    except Exception as e:
        logging.error(f"Unexpected error handling support request: {e}", exc_info=True)
        return {"status": "error", "message": "An unexpected error occurred while logging your request."}


# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page, passing theme, profile, and guide data."""
    themes_json = json.dumps(THEMES_DATA)
    profiles_json = json.dumps(PROFILES_DATA) # Pass profiles data
    # Get list of successfully processed guide filenames
    all_guides_list = sorted([fname for fname, info in loaded_guide_info.items() if info.get('status') == 'success'])
    all_guides_json = json.dumps(all_guides_list)
    return render_template('index.html',
                           themes_json=themes_json,
                           default_theme_name=DEFAULT_THEME_NAME,
                           profiles_json=profiles_json,
                           default_profile_key=DEFAULT_PROFILE_KEY,
                           all_guides_json=all_guides_json) # Pass all guides

@app.route('/get_guide_content')
def get_guide_content():
    """Securely fetches and returns the content of a requested guide file."""
    filename_encoded = request.args.get('filename')
    if not filename_encoded:
        abort(400, "Missing 'filename'.")
    filename_decoded = urllib.parse.unquote(filename_encoded)
    safe_path = get_safe_filepath(filename_decoded)
    if not safe_path:
        abort(404, "File not found or invalid.")
    actual_filename = os.path.basename(safe_path)
    if actual_filename.lower().endswith('.docx'):
        try:
            with open(safe_path, "rb") as docx_file:
                result = mammoth.convert_to_html(docx_file)
            logging.info(f"Converted DOCX '{actual_filename}'.")
            return jsonify({"filename": actual_filename, "html_content": result.value, "is_html": True})
        except Exception as e:
            logging.error(f"Error converting DOCX '{actual_filename}': {e}", exc_info=True)
            abort(500, f"Error processing DOCX: {actual_filename}")
    else:
        content = None
        encoding = 'utf-8'
        try:
            with open(safe_path, 'r', encoding=encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            encoding = 'cp1252'
            try:
                with open(safe_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                logging.error(f"Read error '{actual_filename}' ({encoding}): {e}", exc_info=True)
                abort(500, f"Error reading: {actual_filename}")
        except Exception as e:
            logging.error(f"Read error '{actual_filename}' ({encoding}): {e}", exc_info=True)
            abort(500, f"Error reading: {actual_filename}")
        if content is not None:
            logging.info(f"Read '{actual_filename}' ({encoding}).")
            return jsonify({"filename": actual_filename, "text_content": content, "is_html": False})
        else:
            abort(500, f"Failed read: {actual_filename}")


# --- /ask ROUTE (AI Interaction) ---
@app.route('/ask', methods=['POST'])
def handle_ask():
    """Handles user query, uses Gemini for tool choice, integrates RAG for answers."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    query = data.get('query')
    profile_key = data.get('profile_key', DEFAULT_PROFILE_KEY) # Get selected profile key

    if not query or not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Missing/invalid 'query'"}), 400
    query = query.strip()

    # Get the selected profile data
    selected_profile = PROFILES_DATA.get(profile_key, PROFILES_DATA.get(DEFAULT_PROFILE_KEY))
    if not selected_profile:
         logging.error(f"Selected profile key '{profile_key}' not found and default '{DEFAULT_PROFILE_KEY}' is also missing.")
         return jsonify({"error": "Selected AI profile not found."}), 500

    profile_system_prompt = selected_profile.get('system_prompt', PROFILES_DATA[DEFAULT_PROFILE_KEY]['system_prompt'])
    profile_allowed_guides = selected_profile.get('available_guides', []) # Use empty list if not specified

    if google_genai_client is None or types is None:
        logging.error("Google GenAI client not initialized. Cannot proceed.")
        return jsonify({"error": "Primary AI service unavailable."}), 503

    try:
        # 1. Tool Selection (Gemini still decides the high-level intent)
        # Tool selection prompt is general, not profile-specific
        tool_selection_prompt = construct_gemini_tool_selection_prompt(query)
        logging.info(f"Asking Gemini model '{GEMINI_MODEL_NAME}' to choose a tool...")
        llm_content = None; tool_name_decision = "answer_question"; tool_arguments = {}
        try:
            # Use a non-streaming call for tool selection
            tool_select_config = types.GenerateContentConfig(
                # Ensure JSON output for tool selection if model supports it
                 response_mime_type="application/json",
                 thinking_config=types.ThinkingConfig(thinking_budget=0) if GEMINI_MODEL_NAME == "gemini-2.5-flash-preview-04-17" else None
            )
            # CORRECTED: Use generate_content for non-streaming tool selection
            google_response = generate_with_retry(
                client=google_genai_client,
                model_name=GEMINI_MODEL_NAME,
                contents=[{"role": "user", "parts": [{"text": tool_selection_prompt}]}],
                config=tool_select_config,
                stream=False # Tool selection is NOT streamed
            )

            if google_response.candidates and google_response.candidates[0].content and google_response.candidates[0].content.parts:
                llm_content = google_response.candidates[0].content.parts[0].text
            # Handle potential direct text response if parts structure isn't as expected
            elif hasattr(google_response, 'text'):
                 llm_content = google_response.text
            else:
                raise ValueError("Gemini tool selection returned no usable content.")

            if not llm_content:
                raise ValueError("Gemini tool selection returned empty content.")

            logging.debug(f"Gemini raw tool selection response: {llm_content}")
            # Attempt to parse JSON directly (assuming response_mime_type worked)
            tool_call_data = json.loads(llm_content)

            if isinstance(tool_call_data, dict) and "tool" in tool_call_data:
                tool_name_decision = tool_call_data.get("tool")
                tool_arguments = tool_call_data.get("arguments", {})
            else:
                logging.warning(f"Gemini JSON missing 'tool' or invalid format. Raw: {llm_content}. Assuming 'answer_question'.")
                tool_name_decision = "answer_question"
                tool_arguments = {}
        except (json.JSONDecodeError, ValueError, ConnectionError, Exception) as e:
            logging.error(f"Error calling/parsing Gemini tool selection (retries exhausted?): {e}. Raw response: '{llm_content if llm_content else 'N/A'}'", exc_info=True)
            logging.warning("Gemini tool selection failed/invalid. Defaulting to 'answer_question'.")
            tool_name_decision = "answer_question"
            tool_arguments = {}


        # 2. Execute based on tool decision
        # ai_cited_sources will store sources identified during RAG
        ai_cited_sources = []
        should_stream_answer = False

        if tool_name_decision == "create_theme":
            logging.info("Gemini chose create_theme. Asking Gemini to generate theme details...")
            theme_gen_prompt = construct_google_theme_generation_prompt(query)
            try:
                # Ensure JSON output for theme generation
                theme_gen_config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=0) if GEMINI_MODEL_NAME == "gemini-2.5-flash-preview-04-17" else None
                )
                # CORRECTED: Use generate_content for non-streaming theme generation
                google_response = generate_with_retry(
                    client=google_genai_client,
                    model_name=GEMINI_MODEL_NAME,
                    contents=[{"role": "user", "parts": [{"text": theme_gen_prompt}]}],
                    config=theme_gen_config,
                    stream=False # Theme generation is not streamed
                )
                theme_gen_content = None
                if google_response.candidates and google_response.candidates[0].content and google_response.candidates[0].content.parts:
                    theme_gen_content = google_response.candidates[0].content.parts[0].text
                elif hasattr(google_response, 'text'):
                     theme_gen_content = google_response.text
                else: raise ValueError("Theme generation returned no usable content.")

                if not theme_gen_content: raise ValueError("Theme generation returned empty content.")

                parsed_theme_response = json.loads(theme_gen_content)
                theme_arguments = parsed_theme_response.get("arguments", {}) if isinstance(parsed_theme_response, dict) else {}
                result = handle_create_theme(theme_arguments)
                status_code = 200 if result.get("status") == "success" else 400
                return jsonify(result), status_code
            except (json.JSONDecodeError, ValueError, ConnectionError, Exception) as e:
                 logging.error(f"Error generating/parsing theme details from Gemini (retries exhausted?): {e}", exc_info=True)
                 return jsonify({"status": "error", "message": f"Sorry, I couldn't create the theme due to an internal error processing the AI response."}), 500

        elif tool_name_decision == "request_support":
             logging.info("Gemini chose request_support. Logging request...")
             result = handle_request_support(tool_arguments)
             status_code = 200 if result.get("status") == "success" else 400
             if result.get("status") != "success" and ("internal error" in result.get("message", "").lower() or "unexpected error" in result.get("message", "").lower()): status_code = 500
             return jsonify(result), status_code

        # --- RAG INTEGRATION & STREAMING ANSWER ---
        elif tool_name_decision in ["read_files", "answer_question"]:
            logging.info(f"Processing query with RAG/Answer logic (Tool: {tool_name_decision}).")
            if not document_chunks: logging.warning("Document chunks not available for RAG. Answering without context.")
            relevant_context = {}
            source_info = [] # Will hold {"source": filename} dicts

            if document_chunks:
                try:
                    logging.info("Generating embedding for user query...")
                    query_content = types.Content(parts=[types.Part(text=query)])
                    query_embedding_response = generate_with_retry(
                        client=google_genai_client, model_name=EMBEDDING_MODEL_NAME, contents=query_content,
                        config=types.EmbedContentConfig(task_type="retrieval_query"), stream=False
                    )
                    query_embedding = None
                    if query_embedding_response and hasattr(query_embedding_response, 'embeddings') and query_embedding_response.embeddings and query_embedding_response.embeddings[0].values:
                         query_embedding = query_embedding_response.embeddings[0].values
                         logging.info("Query embedding generated successfully.")
                    else: logging.warning("Query embedding generation returned no usable content.")
                except (ConnectionError, Exception) as e:
                    logging.error(f"Error generating embedding for query (retries exhausted?): {e}", exc_info=True)
                    logging.warning("Query embedding failed. Answering without RAG context.")
                    query_embedding = None

                if query_embedding:
                    logging.info(f"Finding relevant document chunks for profile '{profile_key}'...")
                    # CORRECTED: Pass allowed_guides to find_relevant_chunks
                    relevant_chunks = find_relevant_chunks(query_embedding, top_k=5, allowed_guides=profile_allowed_guides)
                    logging.info(f"Found {len(relevant_chunks)} relevant chunks.")
                    if relevant_chunks:
                        context_parts = []
                        for chunk in relevant_chunks:
                            context_parts.append(chunk["text"])
                            # Store source info for sending to client
                            if {"source": chunk["source"]} not in source_info:
                                source_info.append({"source": chunk["source"]})
                        relevant_context = {"Relevant Context": "\n\n---\n\n".join(context_parts)}
                        logging.debug(f"Context sent to LLM: {relevant_context}")
                        # Set ai_cited_sources based on RAG results *before* calling LLM
                        ai_cited_sources = source_info

            # --- Prepare for Streaming ---
            # CORRECTED: Pass the profile's system prompt to the answer prompt
            answer_gen_prompt = construct_google_answer_prompt(query, relevant_context, profile_system_prompt)
            should_stream_answer = True # Always stream for answer_question/read_files

            # Define the actual streaming generator function
            def stream_answer_generator(stream_iterator, sources_to_cite):
                start_time = time.time()
                sent_chars = 0
                try:
                    # 1. Send sources first
                    yield f"event: sources\ndata: {json.dumps(sources_to_cite)}\n\n".encode('utf-8')
                    logging.info(f"Sent sources event: {sources_to_cite}")

                    # 2. Stream Gemini response chunks
                    for chunk in stream_iterator:
                        # Check if the chunk has candidates, content, and parts
                        if chunk.candidates and chunk.candidates[0].content.parts:
                            # CORRECTED: Access the text from the first part in the parts list
                            token = chunk.candidates[0].content.parts[0].text
                            if token: # Only yield if token is not empty
                                sent_chars += len(token)
                                # Escape newlines and wrap in JSON for SSE data field
                                json_data = json.dumps({"token": token})
                                yield f"event: token\ndata: {json_data}\n\n".encode('utf-8')
                                # Small sleep to prevent overwhelming the client? Optional.
                                # time.sleep(0.001)
                        else:
                             # Handle potential empty chunks or different structures gracefully
                             logging.debug("Stream chunk did not contain expected text part.")


                    # 3. Send end event
                    yield f"event: end\ndata: {{}}\n\n".encode('utf-8')
                    end_time = time.time()
                    logging.info(f"Sent end event. Stream finished in {end_time - start_time:.2f}s. Chars: {sent_chars}")

                except Exception as e:
                    logging.error(f"Stream error during generation: {e}", exc_info=True)
                    yield f"event: error\ndata: {json.dumps({'error': f'Stream generation error: {e}'})}\n\n".encode('utf-8')

            # --- Call Gemini with stream=True ---
            try:
                common_gen_config = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0) if GEMINI_MODEL_NAME == "gemini-2.5-flash-preview-04-17" else None)
                # CORRECTED: Use generate_content_stream for streaming
                gemini_stream_iterator = generate_with_retry(
                    client=google_genai_client,
                    model_name=GEMINI_MODEL_NAME,
                    contents=[{"role": "user", "parts": [{"text": answer_gen_prompt}]}],
                    config=common_gen_config,
                    stream=True # <<<--- IMPORTANT: Enable streaming
                )

                # Create the Flask response with the generator
                generator_instance = stream_answer_generator(gemini_stream_iterator, ai_cited_sources)
                return Response(stream_with_context(generator_instance), mimetype='text/event-stream')

            except (ConnectionError, ValueError, Exception) as e:
                 logging.error(f"Error initiating Gemini stream or during streaming: {e}", exc_info=True)
                 # Return a JSON error if streaming setup fails
                 return jsonify({"error": f"Sorry, error generating answer stream: {e}"}), 500

        else: # Fallback for unexpected tool name (Should not stream this ideally, but keep simple for now)
             logging.warning(f"Unexpected tool decision: {tool_name_decision}. Defaulting to non-streaming answer_question...")
             # CORRECTED: Use the default profile's system prompt for fallback
             fallback_system_prompt = PROFILES_DATA.get(DEFAULT_PROFILE_KEY, {}).get('system_prompt', "You are a helpful AI assistant. Answer questions.")
             answer_gen_prompt = construct_google_answer_prompt(query, {}, fallback_system_prompt)
             try:
                 # Non-streaming call for fallback
                 google_response = generate_with_retry(
                     client=google_genai_client,
                     model_name=GEMINI_MODEL_NAME,
                     contents=[{"role": "user", "parts": [{"text": answer_gen_prompt}]}],
                     config=None, # Use default config for simple fallback
                     stream=False
                 )
                 answer_text = "(No answer provided.)" # Default
                 if google_response.candidates and google_response.candidates[0].content and google_response.candidates[0].content.parts:
                     answer_text = google_response.candidates[0].content.parts[0].text
                 elif hasattr(google_response, 'text'):
                      answer_text = google_response.text

                 answer_markdown = f"*(Note: An unexpected tool '{tool_name_decision}' was suggested.)*\n\n" + answer_text.strip()
                 ai_cited_sources = [] # No sources for fallback

                 # Simulate stream for this fallback case for consistency on client
                 def fallback_stream_generator(full_answer_md, sources_cited_by_ai):
                     yield f"event: sources\ndata: {json.dumps(sources_cited_by_ai)}\n\n".encode('utf-8')
                     chunk_size = 1
                     for i in range(0, len(full_answer_md), chunk_size):
                         chunk = full_answer_md[i:i+chunk_size]
                         yield f"event: token\ndata: {json.dumps({'token': chunk})}\n\n".encode('utf-8')
                         time.sleep(0.0067)
                     yield f"event: end\ndata: {{}}\n\n".encode('utf-8')
                     logging.info("Fallback stream finished.")

                 generator_instance = fallback_stream_generator(answer_markdown, ai_cited_sources)
                 return Response(stream_with_context(generator_instance), mimetype='text/event-stream')

             except (ConnectionError, ValueError, Exception) as e:
                 logging.error(f"Error generating fallback answer: {e}", exc_info=True)
                 return jsonify({"error": f"Sorry, error generating fallback response: {e}"}), 500

        # This part should not be reachable if streaming is handled correctly
        # logging.error("Reached end of /ask logic unexpectedly without streaming or returning a specific JSON response.")
        # return jsonify({"error": "An unexpected internal error occurred after tool selection."}), 500

    except Exception as e:
        logging.error(f"Unhandled error in /ask endpoint: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500


# --- REMOVED CRM API Endpoints ---


# --- Main Execution ---
if __name__ == '__main__':
    # Removed crm_logic.initialize_crm_storage()
    logging.info("Starting Flask development server...")
    try:
        try:
            ollama.list()
            logging.info("Ollama service detected.")
        except Exception as ollama_err:
            logging.warning(f"Ollama connection check failed: {ollama_err}.")
        if google_genai_client is None:
            logging.critical("Google GenAI client is NOT available. AI features DISABLED.")
        else:
            logging.info("Google GenAI client appears available.")
            import asyncio
            async def embed_guides_async():
                 await load_and_embed_guides()
            try:
                # Run embedding in the current event loop if available, otherwise create a new one
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(embed_guides_async())
                    logging.info("Scheduled guide embedding in existing event loop.")
                except RuntimeError: # No running event loop
                    logging.info("No running event loop, running guide embedding synchronously.")
                    asyncio.run(embed_guides_async())

            except Exception as e:
                logging.error(f"Error during initial guide embedding: {e}", exc_info=True)

        app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
    except Exception as e:
        logging.critical(f"Failed to start Flask server: {e}", exc_info=True)
