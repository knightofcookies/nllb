"""
Python script to run a Flask server for translation using CTranslate2 
and Hugging Face Transformers. This script initializes a SQLite database 
for caching translations, downloads a CTranslate model from Hugging Face,
and sets up a Flask web server with CORS support. It provides an endpoint for 
translation requests, checking the cache first before performing the translation.
The script also includes error handling for database operations and translation processes.
"""

import traceback
import time
from datetime import datetime
import sqlite3
import threading
import os
from flask import Flask, request, jsonify, render_template
import ctranslate2
import transformers
from flask_cors import CORS
from huggingface_hub import snapshot_download

DATABASE_FILE = 'translation_cache_ctranslate.db'
db_lock = threading.Lock()

startup_start = datetime.now()
app = Flask(__name__)
CORS(app)

def init_db():
    """Initializes the SQLite database for CTranslate cache."""
    print(f"Initializing CTranslate database at: {os.path.abspath(DATABASE_FILE)}")
    with db_lock:
        conn = None
        try:
            conn = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    text TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    translation TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (text, direction)
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON cache (timestamp);')
            conn.commit()
            print("CTranslate database initialized successfully.")
        except sqlite3.Error as e:
            print(f"!!! CTranslate Database Error during initialization: {e}")
        finally:
            if conn:
                conn.close()

init_db()

ctranslate_model_name = "user10383/nllb-600M-finetune-en-kha-ct2"
local_ctranslate_model_path = "ctranslate_model"  # Define a local directory

print(f"Downloading CTranslate model from Hugging Face: {ctranslate_model_name} to {local_ctranslate_model_path}")
try:
    snapshot_download(repo_id=ctranslate_model_name, local_dir=local_ctranslate_model_path, revision="main") # Or specify a specific revision if needed
    print(f"CTranslate model downloaded successfully to: {local_ctranslate_model_path}")
    translator = ctranslate2.Translator(local_ctranslate_model_path, device="cpu") # Or "cuda" if you have GPU
    print("CTranslate model loaded successfully!")
except Exception as e:
    print(f"!!! Error downloading or loading CTranslate model: {e}")
    translator = None # Handle the case where loading fails

tokenizer_name = "user10383/nllb-600M-finetune-en-kha"
print("Loading tokenizer...")
tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
print("Tokenizer loaded successfully!")

startup_end = datetime.now()
print(f"Startup time: {startup_end - startup_start}")

@app.route('/')
def home():
    """ Renders the home page. """
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    """ Handles translation requests. It checks the cache first and if not found,
    it performs the translation using CTranslate2. """
    data = request.get_json()
    text = data.get('text', '')
    direction = data.get('direction', 'en-kha')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    if translator is None or tokenizer is None:
        return jsonify({'error': 'Model or tokenizer not loaded'}), 500

    start_time = time.time()
    cache_key_tuple = (text, direction)

    # Check cache (unchanged)
    cached_result = None
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT translation FROM cache WHERE text = ? AND direction = ?", (text, direction))
        row = cursor.fetchone()
        if row:
            cached_result = row[0]
    except sqlite3.Error as e:
        print(f"!!! CTranslate Database Error during cache read: {e}")
    finally:
        if conn:
            conn.close()

    if cached_result is not None:
        processing_time = time.time() - start_time
        print(f"Cache hit (SQLite - CTranslate) for: {cache_key_tuple}")
        return jsonify({
            'translation': cached_result,
            'processing_time': f"{processing_time:.4f} seconds",
            'cached': True
        })

    print(f"Cache miss for: {cache_key_tuple}. Translating with CTranslate...")
    try:
        if direction == "en-kha":
            src_lang = "eng_Latn"
            tgt_lang = "vie_Latn"
        elif direction == "kha-en":
            src_lang = "vie_Latn"
            tgt_lang = "eng_Latn"
        else:
            try:
                src_code, tgt_code = direction.split('-')
                lang_map = {"en": "eng_Latn", "kha": "vie_Latn"}
                src_lang = lang_map.get(src_code)
                tgt_lang = lang_map.get(tgt_code)
                if not src_lang or not tgt_lang:
                    return jsonify({'error': f'Unsupported language code in direction: {direction}'}), 400
            except ValueError:
                return jsonify({'error': f'Invalid direction format: {direction}. Use "src-tgt" like "en-kha".'}), 400

        source_sentences = [text]
        inputs = tokenizer(source_sentences, padding=True, truncation=True)
        source_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in inputs.input_ids]
        target_prefix = [[tgt_lang]] * len(source_sentences)

        results = translator.translate_batch(source_tokens, target_prefix=target_prefix)
        
        # Convert string tokens back to IDs for decoding
        translated_tokens = []
        for result in results:
            token_ids = [tokenizer.convert_tokens_to_ids(token) for token in result.hypotheses[0][1:]]
            translated_tokens.append(token_ids)

        translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        processing_time = time.time() - start_time

        # Store in cache (unchanged)
        conn_write = None
        try:
            with db_lock:
                conn_write = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
                cursor = conn_write.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO cache (text, direction, translation, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (text, direction, translation, datetime.now()))
                conn_write.commit()
                print(f"Stored translation in SQLite cache (CTranslate) for: {cache_key_tuple}")
        except sqlite3.Error as e:
            print(f"!!! CTranslate Database Error during cache write: {e}")
            if conn_write:
                conn_write.rollback()
        finally:
            if conn_write:
                conn_write.close()

        return jsonify({
            'translation': translation,
            'processing_time': f"{processing_time:.2f} seconds",
            'cached': False
        })

    except Exception as e:
        print(f"Error during CTranslate translation: {e}")
        print(traceback.format_exc())
        return jsonify({'error': 'Translation failed with CTranslate', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=False)
