import hashlib
import os

# --- Configuration ---
# These paths correspond to your project structure
DATA_DIR = "data"
INDEX_DIR = "./faiss_index"
HASH_FILE = os.path.join(INDEX_DIR, "data_master_hash.txt")
# ---------------------

def calculate_directory_hash(directory: str = DATA_DIR) -> str:
    """
    Calculates a deterministic SHA256 hash for all files in a directory
    based on their content and path.
    """
    hasher = hashlib.sha256()
    
    # 1. Collect all file paths in a sorted list
    file_paths = []
    for root, _, files in os.walk(directory):
        for filename in files:
            # Skip hidden files and the hash file itself
            if filename.startswith('.') or filename == os.path.basename(HASH_FILE):
                continue
            file_paths.append(os.path.join(root, filename))
    
    file_paths.sort() 

    # 2. Update the master hash with file paths and content
    for filepath in file_paths:
        # Include the relative path/name in the hash (sensitive to renames/moves)
        relative_path = os.path.relpath(filepath, directory)
        hasher.update(relative_path.encode('utf-8')) 
        
        # Include file content in chunks for efficiency with large files
        try:
            with open(filepath, 'rb') as f:
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    hasher.update(chunk)
        except Exception as e:
            # Handle potential file access errors gracefully
            print(f"Warning: Could not read file {filepath}. Skipping. Error: {e}")
            
    return hasher.hexdigest()

def get_saved_hash() -> str | None:
    """Retrieves the previously saved hash from the manifest file."""
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            return f.read().strip()
    return None

def save_current_hash(current_hash: str):
    """Saves the calculated hash to a manifest file."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(HASH_FILE, "w") as f:
        f.write(current_hash)
    
def is_rebuild_required() -> tuple[bool, str]:
    """
    Compares the current data hash to the saved hash.
    Returns (True/False, current_hash)
    """
    current_hash = calculate_directory_hash()
    saved_hash = get_saved_hash()
    
    # Check if the FAISS index files exist
    faiss_index_exists = os.path.exists(os.path.join(INDEX_DIR, "index.faiss")) and \
                         os.path.exists(os.path.join(INDEX_DIR, "index.pkl"))

    if not faiss_index_exists:
        print("FAISS index files not found. Index rebuild required.")
        return True, current_hash
    
    if saved_hash is None:
        print("Hash manifest not found. Index rebuild required.")
        return True, current_hash
    
    if current_hash != saved_hash:
        print("Hash mismatch detected! Source data has changed. Index rebuild required.")
        return True, current_hash
    
    print("Hashes match. Saved FAISS index is up-to-date.")
    return False, current_hash