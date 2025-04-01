import os
import argparse
import pandas as pd
import logging
import requests
import time
from sentence_transformers import SentenceTransformer, util
import numpy as np
from abbreviation_rag_query import AbbreviationRAG
from column_memory import ColumnMatchMemory
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Predefined dictionary of known expansions for validation
KNOWN_EXPANSIONS = {
    "financial": {
        "net_ret": "Net Return",
        "net_rtn": "Net Return",
        "cct": "Credit Card Type",
        "card_type": "Card Type"
    },
    "general": {
        "net_ret": "Net Return",
        "net_rtn": "Net Return",
        "cct": "Customer Contact",
        "card_type": "Card Type"
    }
}

# Add a global variable for the RAG system
abbreviation_rag = None

# Initialize the RAG system if available
def init_rag(rag_dir="abbreviation_rag"):
    """Initialize the RAG system if available."""
    global abbreviation_rag
    try:
        if os.path.exists(rag_dir):
            abbreviation_rag = AbbreviationRAG(rag_dir)
            return True
        return False
    except Exception as e:
        print(f"Error initializing RAG system: {str(e)}")
        return False

# DeepSeek integration using direct API calls
class DeepSeekLLM:
    def __init__(self, api_key=None):
        """Initialize the DeepSeek LLM client."""
        # Get API key from secrets or environment variables
        try:
            # First try to get from directly provided parameter
            if api_key:
                self.api_key = api_key
            # Then try Streamlit secrets
            elif hasattr(st, 'secrets') and 'DEEPSEEK_API_KEY' in st.secrets:
                self.api_key = st.secrets['DEEPSEEK_API_KEY']
            # Finally try environment variables
            else:
                self.api_key = os.environ.get('DEEPSEEK_API_KEY')
            
            # Check if we got a key
            if not self.api_key:
                st.error("DeepSeek API key not found. Please contact the app administrator.")
            
        except Exception as e:
            st.error(f"Error accessing API key: {str(e)}")
            self.api_key = None
        
        if not self.api_key:
            raise ValueError("DeepSeek API key not found. Please set in Streamlit Secrets.")
        self.api_base = "https://api.deepseek.com/v1"
        self.model_name = "deepseek-coder"
        logger.info(f"Initialized DeepSeek LLM with model: {self.model_name}")
        
    def __call__(self, prompt, temperature=0.0, max_tokens=1024):
        """Call the model with the given prompt using direct API requests."""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Make the API request
            response = requests.post(
                f"{self.api_base}/chat/completions", 
                headers=headers,
                json=data
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                error_msg = f"API request failed with status code {response.status_code}: {response.text}"
                logger.error(error_msg)
                return f"Error: {error_msg}"
                
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {str(e)}")
            return f"Error: {str(e)}"

def extract_answer(raw_answer_str: str, sep_token: str, abbreviations: list):
    """Process the raw model output into a list of expanded column names."""
    try:
        # Log the raw answer for debugging
        logger.info(f"Raw answer from LLM: {raw_answer_str}")
        
        # Clean the answer string
        raw_answer_str = raw_answer_str.strip()
        
        # First attempt: Split on sep_token (expected format)
        # Try to find the actual list of expansions - most direct format
        answer_list = [_ans.strip() for _ans in raw_answer_str.split(sep_token)]
        if len(answer_list) == len(abbreviations):
            logger.info(f"Successfully parsed with separator token: {answer_list}")
            return answer_list
        
        # Second attempt: Look for a clear list-like format with the same number of items
        import re
        list_pattern = r"(?:^|\n)(?:\d+[\.\)]\s*)?(.+?)(?:$|\n)"
        matches = re.findall(list_pattern, raw_answer_str)
        if len(matches) == len(abbreviations):
            logger.info(f"Successfully parsed with list pattern: {matches}")
            return [match.strip() for match in matches]
        
        # Third attempt: Look for direct mappings in the text
        predictions = []
        for abbr in abbreviations:
            # Try multiple patterns to capture different response formats
            patterns = [
                rf"`{abbr}`[^-]*stands for \"([^\"]+)\"",  # `BP` stands for "Blood Pressure"
                rf"{abbr}[^-]*means \"([^\"]+)\"",         # BP means "Blood Pressure"
                rf"{abbr}[^-]*:[ \t]*([^\n\.,]+)",         # BP: Blood Pressure
                rf"{abbr}[ \t]+->[ \t]+([^\n\.,]+)"        # BP -> Blood Pressure
            ]
            
            for pattern in patterns:
                match = re.search(pattern, raw_answer_str, re.IGNORECASE)
                if match:
                    expanded_name = match.group(1).strip()
                    predictions.append(expanded_name)
                    logger.info(f"Matched {abbr} to {expanded_name} using pattern {pattern}")
                    break
            else:
                # If no match found, try to find any mention of this abbreviation followed by text
                general_pattern = rf"(?:^|\n|\s){re.escape(abbr)}(?:\s|:)+(.*?)(?:\n|$|\.|,)"
                match = re.search(general_pattern, raw_answer_str, re.IGNORECASE)
                if match:
                    expanded_name = match.group(1).strip()
                    predictions.append(expanded_name)
                    logger.info(f"Matched {abbr} to {expanded_name} using general pattern")
                else:
                    predictions.append(abbr)  # Fall back to the abbreviation itself
                    logger.warning(f"Could not find expansion for {abbr}")
        
        if len(predictions) == len(abbreviations):
            logger.info(f"Successfully parsed with direct mapping patterns: {predictions}")
            return predictions
            
        # If we still don't have enough predictions, fill with empty strings
        if len(predictions) < len(abbreviations):
            logger.warning(f"Not enough predictions ({len(predictions)}) for abbreviations ({len(abbreviations)})")
            predictions.extend([" "] * (len(abbreviations) - len(predictions)))
        
        return predictions[:len(abbreviations)]  # Ensure correct length
        
    except Exception as e:
        logger.error(f"Error extracting answer: {str(e)}")
        return [" "] * len(abbreviations)

class PromptTemplate:
    @property
    def demos(self):
        _demo = (
            "As abbreviations of column names from a table, "
            "c_name | pCd | dt stand for Customer Name | Product Code | Date. "
        )
        return _demo

    @property
    def sep_token(self):
        _sep_token = " | "
        return _sep_token

def expand_abbreviations(abbreviations: list, context: str, model: DeepSeekLLM, 
                        prompt_template: PromptTemplate, verbose: bool = False):
    """Expand abbreviations using the LLM."""
    global abbreviation_rag
    
    # Initialize RAG if not already done
    if abbreviation_rag is None:
        init_rag()
    
    # Construct prompt
    query = f"Expand: {' | '.join(abbreviations)} "
    context_part = f"Context: {context}. " if context else ""
    
    # Add RAG-enhanced context if available
    rag_context = ""
    if abbreviation_rag is not None:
        for abbr in abbreviations:
            abbr_context = abbreviation_rag.get_context_for_llm(abbr, top_k=5)
            if abbr_context and "No abbreviation information found" not in abbr_context:
                rag_context += f"For abbreviation '{abbr}':\n{abbr_context}\n"
    
    if rag_context:
        rag_context = f"Reference information from medical abbreviation database:\n{rag_context}\n"
        if verbose:
            print("\nDebug: RAG Context:")
            print(rag_context)
    
    prompt = (
        f"{context_part}{rag_context}{prompt_template.demos}{query}"
        "Provide the expanded names as a list separated by ' | ', e.g., 'Expanded Name 1 | Expanded Name 2'. "
        "Do not include additional explanations or examples."
    )

    # Debug: Print the prompt
    if verbose:
        print("\nDebug: Prompt sent to the model:")
        print(prompt)

    # Get model prediction
    raw_answer = model(prompt)
    
    # Add a small delay to avoid rate limiting
    time.sleep(1)
    
    predictions = extract_answer(raw_answer, prompt_template.sep_token, abbreviations)
    if verbose:
        print(f"Debug: Raw answer: {raw_answer}")
        print(f"Debug: Predictions: {predictions}")

    # Handle mismatched prediction lengths
    if len(predictions) != len(abbreviations):
        print(f"Error! Got {len(predictions)} predictions, expected {len(abbreviations)}")
        print(f"Raw answer: {raw_answer}")
        predictions = [" "] * len(abbreviations)

    # Validate expansions using the predefined dictionary
    domain = "medical" if "medical" in context.lower() or "clinical" in context.lower() else ("financial" if "financial" in context.lower() else "general")
    validated_predictions = []
    for abbr, pred in zip(abbreviations, predictions):
        known_expansion = KNOWN_EXPANSIONS.get(domain, {}).get(abbr)
        if known_expansion and pred != known_expansion:
            if verbose:
                print(f"Debug: Overriding expansion for {abbr}: {pred} -> {known_expansion}")
            validated_predictions.append(known_expansion)
        else:
            validated_predictions.append(pred)

    return validated_predictions

@st.cache_resource
def load_models():
    """Cached model loading"""
    deepseek_model = DeepSeekLLM()
    prompt_template = PromptTemplate()
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    return deepseek_model, prompt_template, semantic_model

def match_columns(source_cols, dest_cols, source_expanded, dest_expanded, 
                 semantic_model, verbose=False, memory_matches=None, 
                 similarity_threshold=0.5):
    """Match columns between source and destination tables using semantic similarity and memory."""
    if memory_matches is None:
        memory_matches = {}
    
    # For columns with memory matches, we'll use those directly
    # For others, we'll use semantic matching
    
    # Compute embeddings only for columns that need semantic matching
    semantic_source_cols = []
    semantic_source_expanded = []
    semantic_indices = []
    
    for i, (col, expanded) in enumerate(zip(source_cols, source_expanded)):
        if col in memory_matches:
            continue  # Skip columns with memory matches
        semantic_source_cols.append(col)
        semantic_source_expanded.append(expanded)
        semantic_indices.append(i)
    
    # Initialize all_comparisons list to store all pair comparisons
    all_comparisons = []
    
    # Skip semantic matching entirely if all columns have memory matches
    if semantic_source_cols:
        # Ensure all expanded names are strings and handle empty expansions
        semantic_source_expanded = [
            str(exp) if exp and exp.strip() and not exp.startswith("MEMORY_MATCH:")
            else f"Column {src}" 
            for src, exp in zip(semantic_source_cols, semantic_source_expanded)
        ]
        
        dest_expanded = [
            str(exp) if exp and exp.strip() else f"Column {dst}" 
            for dst, exp in zip(dest_cols, dest_expanded)
        ]
        
        # Compute embeddings and similarities as before
        source_embeddings = semantic_model.encode(semantic_source_expanded, convert_to_tensor=True)
        dest_embeddings = semantic_model.encode(dest_expanded, convert_to_tensor=True)
        similarity_matrix = util.cos_sim(source_embeddings, dest_embeddings)
        
        # Generate all comparisons for semantic columns
        for i, (src_col, src_exp, orig_idx) in enumerate(zip(semantic_source_cols, semantic_source_expanded, semantic_indices)):
            for j, (dst_col, dst_exp) in enumerate(zip(dest_cols, dest_expanded)):
                score = similarity_matrix[i][j].item()
                all_comparisons.append({
                    "source_column": src_col,
                    "source_expanded": src_exp,
                    "dest_column": dst_col,
                    "dest_expanded": dst_exp,
                    "similarity_score": score
                })
    
    # Find the best match for each source column, or mark as NO_MATCH if below threshold
    matches = []
    for i, (src_col, src_exp) in enumerate(zip(source_cols, source_expanded)):
        if src_col in memory_matches:
            # Use memory match
            dest_col = memory_matches[src_col]
            dest_idx = dest_cols.index(dest_col)
            dest_exp = dest_expanded[dest_idx]
            
            match_entry = {
                "source_column": src_col,
                "source_expanded": src_exp,
                "dest_column": dest_col,
                "dest_expanded": dest_exp,
                "similarity_score": 1.0,  # Perfect score for memory matches
                "from_memory": True
            }
            
            # Add this to all_comparisons too for memory matches
            for j, (dst_col, dst_exp) in enumerate(zip(dest_cols, dest_expanded)):
                score = 1.0 if dst_col == dest_col else 0.0
                all_comparisons.append({
                    "source_column": src_col,
                    "source_expanded": src_exp,
                    "dest_column": dst_col,
                    "dest_expanded": dst_exp,
                    "similarity_score": score
                })
                
        else:
            # Use semantic matching
            semantic_idx = semantic_indices.index(i)
            similarities = similarity_matrix[semantic_idx]
            best_match_idx = similarities.argmax().item()
            best_score = similarities[best_match_idx].item()
            
            # If best match is below threshold, mark as NO_MATCH
            if best_score < similarity_threshold:
                match_entry = {
                    "source_column": src_col,
                    "source_expanded": src_exp,
                    "dest_column": "NO_MATCH",
                    "dest_expanded": "NO_MATCH",
                    "similarity_score": best_score,
                    "from_memory": False
                }
            else:
                dest_col = dest_cols[best_match_idx]
                dest_exp = dest_expanded[best_match_idx]
                
                match_entry = {
                    "source_column": src_col,
                    "source_expanded": src_exp,
                    "dest_column": dest_col,
                    "dest_expanded": dest_exp,
                    "similarity_score": best_score,
                    "from_memory": False
                }
        
        matches.append(match_entry)
    
    # Find the pair with the highest similarity score across all comparisons
    if all_comparisons:
        highest_similarity_pair = max(all_comparisons, key=lambda x: x["similarity_score"])
    else:
        # Default if no comparisons were made
        highest_similarity_pair = {
            "source_column": source_cols[0] if source_cols else "",
            "dest_column": dest_cols[0] if dest_cols else "",
            "similarity_score": 0
        }
    
    return matches, all_comparisons, highest_similarity_pair

def process_files(source_file, dest_file, context=None, use_memory=True, memory_file="data/column_matches.json", save_new_matches=True):
    """Process source and destination files to match columns."""
    # Initialize our column memory
    if use_memory:
        # Ensure data directory exists
        os.makedirs(os.path.dirname(os.path.abspath(memory_file)), exist_ok=True)
        column_memory = ColumnMatchMemory(memory_file)
    else:
        column_memory = None
    
    # ... existing code to load files and get columns ...
    
    # Check memory for existing matches
    memory_matches = {}
    if use_memory and column_memory:
        memory_matches = column_memory.find_matches(source_cols, dest_cols)
        
    # Determine which columns need expansion through the LLM
    columns_to_expand = [col for col in source_cols if col not in memory_matches]
    
    # Only process columns that need expansion
    if columns_to_expand:
        # ... existing code for abbreviation expansion ...
        source_expanded = expand_abbreviations(columns_to_expand, context, deepseek_model, prompt_template, verbose)
        # Create full source_expanded list by combining memory and new expansions
        all_expanded = []
        expansion_index = 0
        for col in source_cols:
            if col in memory_matches:
                # Use a placeholder - we'll use memory for matching directly
                all_expanded.append(f"MEMORY_MATCH:{col}")
            else:
                all_expanded.append(source_expanded[expansion_index])
                expansion_index += 1
    else:
        # If all columns are in memory, we still need a placeholder list
        all_expanded = [f"MEMORY_MATCH:{col}" for col in source_cols]
    
    # Expand destination columns as normal
    dest_expanded = expand_abbreviations(dest_cols, context, deepseek_model, prompt_template, verbose)
    
    # Modify the match_columns function call
    matches, all_comparisons, highest_pair = match_columns(
        source_cols, dest_cols, all_expanded, dest_expanded, 
        semantic_model, memory_matches=memory_matches
    )
    
    # Save new matches if requested
    if use_memory and save_new_matches and column_memory:
        for match in matches:
            source_col = match["source_column"]
            dest_col = match["dest_column"]
            # Only save matches with high confidence
            if match["similarity_score"] > 0.85:  # Threshold can be adjusted
                column_memory.add_match(source_col, dest_col)
    
    return matches, all_comparisons, highest_pair

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-cols", type=str, required=True,
        help="Pipe-separated list of source table column names (e.g., 'net_ret | cct')"
    )
    parser.add_argument(
        "--dest-cols", type=str, required=True,
        help="Pipe-separated list of destination table column names (e.g., 'net_rtn | card_type')"
    )
    parser.add_argument(
        "--context", type=str, default="",
        help="Optional context for the LLM prompt (e.g., 'financial data')"
    )
    parser.add_argument(
        "--model_name", type=str, default="deepseek-coder",
        help="DeepSeek model to use (default: deepseek-coder)"
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="DeepSeek API key (if not provided, will use environment variable or default)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed debug information"
    )
    args = parser.parse_args()

    # Process input column names
    source_cols = [col.strip() for col in args.source_cols.split("|")]
    dest_cols = [col.strip() for col in args.dest_cols.split("|")]
    if not source_cols or not dest_cols:
        print("No columns provided for source or destination table. Exiting.")
        exit(1)

    # Debug: Print the columns
    if args.verbose:
        print("\nDebug: Source columns:")
        print(source_cols)
        print("Debug: Destination columns:")
        print(dest_cols)

    # Initialize models
    deepseek_model, prompt_template, semantic_model = load_models()

    print(f"Using model: {args.model_name}")
    print("Expanding source table columns...")
    matches, all_comparisons, highest_pair = process_files(source_cols, dest_cols, args.context, args.verbose)

    # Print all comparisons
    print("\nAll Comparisons:")
    for match in all_comparisons:
        print(f"{match['source_column']} ({match['source_expanded']}) vs. {match['dest_column']} ({match['dest_expanded']}), Similarity = {match['similarity_score']:.3f}")

    # Print best matches for each source column
    print("\nBest Matches for Each Source Column:")
    for match in matches:
        print(f"{match['source_column']} ({match['source_expanded']}) --> {match['dest_column']} ({match['dest_expanded']}), Similarity = {match['similarity_score']:.3f}")

    # Print the pair with the highest similarity
    print("\nPair with Highest Similarity:")
    print(f"{highest_pair['source_column']} ({highest_pair['source_expanded']}) --> {highest_pair['dest_column']} ({highest_pair['dest_expanded']}), Similarity = {highest_pair['similarity_score']:.3f}")

    # Save best matches to CSV
    results_df = pd.DataFrame(matches)
    output_dir = "outputs"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "matched_columns.csv")
    if os.path.exists(output_file):
        os.remove(output_file)
    results_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nBest matches saved to {output_file}")

    # Save all comparisons to CSV
    all_comparisons_df = pd.DataFrame(all_comparisons)
    comparisons_file = os.path.join(output_dir, "all_comparisons.csv")
    if os.path.exists(comparisons_file):
        os.remove(comparisons_file)
    all_comparisons_df.to_csv(comparisons_file, index=False, encoding='utf-8')
    print(f"All comparisons saved to {comparisons_file}")
    print("Done!") 