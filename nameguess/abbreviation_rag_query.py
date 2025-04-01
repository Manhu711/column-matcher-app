import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import os
import streamlit as st

class AbbreviationRAG:
    def __init__(self, rag_dir=None):
        """
        Initialize the Abbreviation RAG system.
        
        Args:
            rag_dir: Directory containing the RAG system files
        """
        if rag_dir is None:
            # Automatically find the RAG directory relative to this file
            rag_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "abbreviation_rag")
        
        self.rag_dir = rag_dir
        if not os.path.exists(rag_dir):
            raise ValueError(f"RAG directory {rag_dir} not found!")
        
        # Load the embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load the FAISS index
        index_path = os.path.join(rag_dir, "abbreviation_index.faiss")
        self.index = faiss.read_index(index_path)
        
        # Load the data
        data_path = os.path.join(rag_dir, "abbreviation_data.pkl")
        self.df = pd.read_pickle(data_path)
        
        print(f"Loaded abbreviation RAG with {len(self.df)} entries")
    
    def query(self, query_text, top_k=5):
        """
        Query the RAG system for abbreviation expansions.
        
        Args:
            query_text: The abbreviation or text to query
            top_k: Number of results to return
            
        Returns:
            A list of dictionaries with matched abbreviations and expansions
        """
        # Convert query to embedding
        query_embedding = self.model.encode([query_text])[0].reshape(1, -1).astype('float32')
        
        # Search the index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get the results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.df) and idx >= 0:
                entry = self.df.iloc[idx]
                results.append({
                    'short_form': entry['SF'],
                    'normalized_short_form': entry['NormSF'],
                    'long_form': entry['LF'],
                    'normalized_long_form': entry['NormLF'],
                    'source': entry['Source'],
                    'similarity_score': 1.0 - distances[0][i] / 100.0  # Convert distance to similarity
                })
        
        return results
    
    def get_context_for_llm(self, query_text, top_k=5):
        """
        Generate context for an LLM prompt based on RAG query results.
        Ensures only unique interpretations are included, handling cases with fewer
        than top_k unique results.
        
        Args:
            query_text: The abbreviation or text to query
            top_k: Maximum number of results to include
            
        Returns:
            A formatted string with relevant abbreviation information
        """
        # Request twice as many results to account for duplicates
        results = self.query(query_text, top_k=max(10, top_k*2))
        
        if not results:
            return "No abbreviation information found."
        
        # Filter to keep only unique long forms
        unique_results = []
        seen_long_forms = set()
        
        for result in results:
            # Normalize the long form for deduplication
            norm_long_form = result['normalized_long_form'].lower() if result['normalized_long_form'] else ''
            if norm_long_form and norm_long_form not in seen_long_forms:
                seen_long_forms.add(norm_long_form)
                unique_results.append(result)
                # Stop once we have enough unique results
                if len(unique_results) >= top_k:
                    break
        
        # If we have no results after filtering, return the original results
        if not unique_results and results:
            unique_results = results[:min(len(results), top_k)]
        
        context = "Relevant abbreviation information:\n\n"
        for i, result in enumerate(unique_results):
            context += f"{i+1}. '{result['short_form']}' means '{result['long_form']}' "
            context += f"(normalized as '{result['normalized_short_form']}' -> '{result['normalized_long_form']}')\n"
            context += f"   Source: {result['source']}, Confidence: {result['similarity_score']:.2f}\n\n"
        
        return context

def init_rag(rag_dir="abbreviation_rag"):
    """Initialize RAG system, downloading data if needed"""
    if not os.path.exists(rag_dir):
        st.info("Downloading and processing abbreviation data...")
        # Add code to download and process data
        download_and_process_abbreviations(rag_dir)
    return True

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Query the abbreviation RAG system")
    parser.add_argument("--query", required=True, help="Abbreviation to query")
    parser.add_argument("--rag_dir", default="abbreviation_rag", help="RAG directory")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()
    
    rag = AbbreviationRAG(args.rag_dir)
    results = rag.query(args.query, args.top_k)
    
    print(f"Results for '{args.query}':")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['short_form']} -> {result['long_form']} (Score: {result['similarity_score']:.2f})")
    
    print("\nLLM Context:")
    print(rag.get_context_for_llm(args.query, args.top_k)) 