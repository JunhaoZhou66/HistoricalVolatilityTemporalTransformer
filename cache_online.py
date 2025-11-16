import pandas as pd
import pyarrow 
import sys
import json
import re
import os
import requests 
import nltk
from nltk.corpus import words
from collections import Counter # <-- 1. Import Counter
try:
    nltk.data.find('corpora/words')
except:
    print("Downloading 'words' corpus for dictionary check...")
    nltk.download('words')

ENGLISH_WORDS = set(words.words())
TICKER_SYMBOLS = {"meta", "amd", "aapl", "msft", "tsla", "nvda"}


with open("stopwords.json") as f:
    data = json.load(f)


HTML_ATTRIBUTE_STOP_WORDS = data['naive_stopwords']





def retrieve_url_content(url, output_file_path):
    print(f"Attempting to retrieve content from: {url}")
    try:
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
    with open(input_file_path, 'r', encoding='utf-8') as f:
        raw_html = f.read()

    initial_tokens = raw_html.split()
    initial_count = len(initial_tokens)
    
    # 1. STEP 1: Remove Style blocks and attributes
    style_block_pattern = re.compile(r'<style.*?>(.*?)</style>', re.DOTALL | re.IGNORECASE)
    intermediate_html = re.sub(style_block_pattern, ' ', raw_html)
    
    style_attr_pattern = re.compile(r'style=".*?"', re.IGNORECASE)
    intermediate_html = re.sub(style_attr_pattern, ' ', intermediate_html)
    
    # 2. STEP 2: Remove remaining HTML/XML/SVG Tags
    tag_pattern = re.compile('<.*?>')
    cleaned_text_no_tags = re.sub(tag_pattern, ' ', intermediate_html)

    # 3. STEP 3: Remove Punctuation/Numbers and Lowercase
    cleaned_text_final_str = re.sub(r'[^a-zA-Z\s]', ' ', cleaned_text_no_tags).lower()
    
    # 4. STEP 4: Apply Word-Level Filtering
    potential_words = cleaned_text_final_str.split()

    cleaned_text_list = []
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
                    
    # 5. STEP 5: Write output
    output_content = '\n'.join(cleaned_text_list)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(output_content)
        
    # 6. STEP 6: Report Metrics
    final_count = len(cleaned_text_list)
    removed_count = initial_count - final_count

    print("-" * 40)
    print(f"Metrics for file: {input_file_path}")
    print(f"1. Total items (tokens) before all masking: {initial_count}")
    print(f"2. Items (approx.) removed by all masks: {removed_count}")
    print(f"3. Total items (pure words, filtered) after masking: {final_count}")
    print(f"Cleaned output saved to: {output_file_path}")

    # --- 7. NEW: Percent Frequency Counting ---
    if final_count > 0: # Avoid division by zero
        print("\n--- Top 50 Word Frequencies ---")
        
        # Create the frequency counter
        word_counts = Counter(cleaned_text_list)
        
        # Get the 50 most common
        top_50_words = word_counts.most_common(50)
        
        # Print them in a formatted table
        print(f"{'Rank':<5} {'Word':<20} {'Count':<8} {'Frequency':<10}")
        print(f"{'-'*4:<5} {'-'*19:<20} {'-'*7:<8} {'-'*9:<10}")
        
        for i, (word, count) in enumerate(top_50_words, 1):
            percent = (count / final_count) * 100
            print(f"{i:<5} {word:<20} {count:<8} {percent:>9.2f}%")
            
    print("-" * 40)


if __name__ == "__main__":


        

    parquet_file = "NEWS_20240101-142500_20251101-232422.parquet"

    df = pd.read_parquet(parquet_file)


    df_sorted = df.sort_values(by='time_published_ts', ascending=True)

    print(df_sorted["url"].head())

    INPUT_FILE = 'raw_html_cache.txt' 
    OUTPUT_FILE = 'clean_text.txt'
    URL = df_sorted["url"][0]

    if retrieve_url_content(URL, INPUT_FILE):
        clean_html_tags(INPUT_FILE, OUTPUT_FILE)



