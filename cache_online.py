import pandas as pd
import pyarrow 
import sys

parquet_file = "NEWS_20240101-142500_20251101-232422.parquet"

df = pd.read_parquet(parquet_file)


df_sorted = df.sort_values(by='time_published_ts', ascending=True)

print(df_sorted["url"].head())

import re
import os
import requests # <-- ADDED: For fetching the URL content
import nltk
from nltk.corpus import words

try:
    nltk.data.find('corpora/words')
except nltk.downloader.DownloadError:
    print("Downloading 'words' corpus for dictionary check...")
    nltk.download('words')

# Convert the list of words to a set for fast lookup (O(1) complexity)
ENGLISH_WORDS = set(words.words())
TICKER_SYMBOLS = {"meta", "amd", "aapl", "msft", "tsla", "nvda"}
HTML_ATTRIBUTE_STOP_WORDS = {
    # Layout & Structure
    "div", "span", "row", "col", "container", "wrapper", "header", "footer", 
    "nav", "sidebar", "main", "content", "section", "article",
    
    # State & Visibility
    "active", "current", "hidden", "show", "toggle", "open", "close", 
    "disabled", "selected", "has", 
    
    # Position & Spacing
    "top", "bottom", "left", "right", "center", "float", "margin", 
    "padding", "space", "block", "inline", "pull", 
    
    # Visual & Theme
    "default", "primary", "secondary", "light", "dark", "white", "black", 
    "gray", "red", "blue", "green", "transparent", 
    
    # Responsiveness
    "sm", "md", "lg", "xl", "mobile", "desktop", "tablet", "screen", "print", 
    
    # Utility & Icons
    "icon", "button", "link", "img", "src", "href", "data", "aria", "role", 
    "svg", "xml", "lazy", "load", "loader", 
    
    # Scripting & Events
    "function", "click", "event", "on", "handler", "widget", "js", "trigger", 
    "track", "analytics", 
    "name", "description", 
    "type",
"head",
"title",

"tag",
"extracted", 
"the", 
"com"
"a", 
"https",
"and",
"blockname", 
"tag", 
# "strong", 
"display", 
"content", 
"select", 
"async", 
"target", 
"blank", 
"lang", 
"width", 
"height", 
"schema", 
"embed", 
"quote",
"text", 
"title", 
"null", 
"logo", 
"article", 
"context", 
"publisher", 
"symbol",

"true",
"page",
"query",
"false",
"true",
"nid",
"author",
"channel",
"best",
"buy",
"summary",
"large",
"image",
"term",
"string",
"search", 
"subscribe",

"sell",
"personal",
"privacy",
"policy",
"disclaimer",
"service",
"status",
"all",
"reserved",


}


def retrieve_url_content(url, output_file_path):
    """
    Fetches the raw HTML content from the given URL and saves it to a file.
    Returns True on success, False on failure.
    """
    print(f"Attempting to retrieve content from: {url}")
    try:
        # Set a User-Agent to mimic a browser, which helps prevent blocks
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        raw_html = response.text
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(raw_html)
        
        print(f"Successfully retrieved content and saved to: {output_file_path}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error retrieving URL content: {e}")
        return False


def clean_html_tags(input_file_path, output_file_path):
    """
    Reads HTML, applies targeted CSS removal, tag removal, filtering, 
    writes the result, and prints metrics.
    """

    # 1. Read the raw HTML content
    with open(input_file_path, 'r', encoding='utf-8') as f:
        raw_html = f.read()

    initial_tokens = raw_html.split()
    initial_count = len(initial_tokens)

    # -----------------------------------------------------------------
    # NEW STEP 1: Targeted CSS Removal (The "Dynamic/CFG" Way)
    # -----------------------------------------------------------------
    # Pattern 1: Remove everything inside <style>...</style> blocks (non-greedy)
    # re.DOTALL ensures '.' matches newlines, capturing multi-line CSS blocks.
    style_block_pattern = re.compile(r'<style.*?>(.*?)</style>', re.DOTALL | re.IGNORECASE)
    intermediate_html = re.sub(style_block_pattern, ' ', raw_html)
    
    # Pattern 2: Remove inline style attributes (non-greedy)
    # This targets attributes like style="..." inside any tag.
    style_attr_pattern = re.compile(r'style=".*?"', re.IGNORECASE)
    intermediate_html = re.sub(style_attr_pattern, ' ', intermediate_html)
    
    # -----------------------------------------------------------------
    # 2. STEP 2 (Original Step 1): Remove remaining HTML/XML/SVG Tags
    # -----------------------------------------------------------------
    tag_pattern = re.compile('<.*?>')
    cleaned_text_no_tags = re.sub(tag_pattern, ' ', intermediate_html)

    # 3. STEP 3 (Original Step 2): Remove Punctuation/Numbers and Lowercase
    cleaned_text_final_str = re.sub(r'[^a-zA-Z\s]', ' ', cleaned_text_no_tags).lower()
    
    # 4. STEP 4 (Original Step 3): Apply Word-Level Filtering
    cleaned_text_final_str = re.sub(r'[^a-zA-Z\s]', ' ', cleaned_text_no_tags).lower()
    potential_words = cleaned_text_final_str.split()

    cleaned_text_list = []
    # Initialize a variable to track the last word that was successfully added
    previous_word = None 

    for word in potential_words:
        # Rule: Remove words that are too short (<= 2) or too long (>= 17)
        if not (len(word) <= 2 or len(word) >= 17):
            
            # Combined Filtering Logic
            is_english_word = word in ENGLISH_WORDS
            is_ticker_symbol = word in TICKER_SYMBOLS
            is_html_symbol = word in HTML_ATTRIBUTE_STOP_WORDS
            # Check 1: Must be a meaningful word (English OR Ticker) AND not HTML noise
            if (is_english_word or is_ticker_symbol) and not is_html_symbol:
                if word != previous_word:
                    cleaned_text_list.append(word)
        
                    previous_word = word
        
    # 5. Write the cleaned list to the final output file
    output_content = '\n'.join(cleaned_text_list)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(output_content)

    # 6. Calculate and Print Metrics
    final_count = len(cleaned_text_list)
    removed_count = initial_count - final_count

    print("-" * 40)
    print(f"Metrics for file: {input_file_path}")
    print(f"1. Total items (tokens) before all masking: {initial_count}")
    print(f"2. Items (approx.) removed by all masks: {removed_count}")
    print(f"3. Total items (pure words, filtered) after masking: {final_count}")
    print(f"Cleaned output saved to: {output_file_path}")
    print("-" * 40)

# --- Main Execution ---
INPUT_FILE = 'raw_html_cache.txt' # Renamed for clarity: this holds the raw HTML
OUTPUT_FILE = 'clean_text.txt'
URL = "https://www.zacks.com/stock/news/2759433/is-flexshares-us-quality-large-cap-etf-qlc-a-strong-etf-right-now"

# 1. Retrieve the URL content and save it to the input file
if retrieve_url_content(URL, INPUT_FILE):
    clean_html_tags(INPUT_FILE, OUTPUT_FILE)
