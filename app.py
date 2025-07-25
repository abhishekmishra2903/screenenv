# pip install screenenv chromadb langchain pillow pytesseract
# sudo port install tesseract
# sudo port install tesseract-eng

# --- Imports ---
import os
from screenenv import sandbox
from screenenv.sandbox import Sandbox as OriginalSandbox
from PIL import ImageGrab, Image
import pytesseract
import platform
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.core.settings import Settings 
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Patch pytesseract to handle non-UTF-8 error output ---
def safe_get_errors(error_string):
    """Safely decode Tesseract error output, replacing undecodable bytes."""
    try:
        return [
            line
            for line in error_string.decode("utf-8", errors="replace").splitlines()
            if line
        ]
    except Exception:
        return ["[pytesseract] Could not decode error output."]

# Apply the patch
pytesseract.pytesseract.get_errors = safe_get_errors

# --- Patched Sandbox for macOS screenshots ---
class PatchedSandbox(OriginalSandbox):
    def screenshot(self):
        # Use PIL's ImageGrab for screenshots on macOS
        if platform.system() == "Darwin":
            return ImageGrab.grab()
        return super().screenshot()

# --- OCR and Context Extraction ---
def get_current_context():
    """
    Capture a screenshot using PatchedSandbox, save it, and extract text using Tesseract OCR.
    Returns the extracted text, or None if OCR fails.
    """
    with PatchedSandbox() as s:
        raw_img = s.screenshot()

        # Prepare temp directory for saving screenshots
        temp_dir = os.path.expanduser("~/screenenv_temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_ocr_path = os.path.join(temp_dir, "temp_ocr.png")

        # Save the screenshot directly (raw_img is a PIL Image)
        raw_img.save(temp_ocr_path)

        # Run OCR on the saved image
        try:
            ocr_text = pytesseract.image_to_string(temp_ocr_path, lang='eng')
            return ocr_text
        except pytesseract.TesseractError as e:
            print(f"[❌] Tesseract OCR failed:\n{e}")
            return None

# --- LlamaIndex Setup ---
# Load environment variables (for Gemini API key)
load_dotenv()
llm = Gemini(model="models/gemini-1.5-flash", api_key=os.getenv("GEMINI_API_KEY"))

# Set LLM and embedding model for LlamaIndex
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding()

def create_index_from_screen(ocr_text):
    """
    Create a vector index from the OCR text.
    """
    doc = Document(text=ocr_text)
    index = VectorStoreIndex.from_documents([doc])
    return index

def ask_agent_question(index, question: str):
    """
    Query the index with a question and print the response.
    """
    agent = index.as_query_engine(similarity_top_k=3)
    response = agent.query(question)
    print(f"\n[🔍] Q: {question}\n\n🧠 A: {response}")

# --- Main Execution ---
if __name__ == "__main__":
    # Step 1: Capture screen and extract text
    ocr_text = get_current_context()
    if not ocr_text or not ocr_text.strip():
        print("No text found in screenshot, skipping indexing and querying.")
        exit()

    # Step 2: Create index from OCR text
    index = create_index_from_screen(ocr_text)

    # Step 3: Ask questions about the screen content
    ask_agent_question(index, "Summarize what I’m reading right now.")
    ask_agent_question(index, "What does this document relate to?")
    ask_agent_question(index, "List 3 advanced terms and explain them.")
