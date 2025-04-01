import streamlit as st
import pandas as pd
import os

# Define memory path - ensures it works both locally and in cloud
def get_memory_path():
    # Check if running on Streamlit Cloud
    if os.environ.get('STREAMLIT_DEPLOYMENT') == 'cloud':
        # In cloud, use a path that's within the app's deployed directory
        return os.path.join(os.path.dirname(__file__), "..", "data", "column_matches.json")
    else:
        # Locally, use the same path as before
        return os.path.join("data", "column_matches.json")

MEMORY_FILE_PATH = get_memory_path()

from match_columns_deepseek import DeepSeekLLM, PromptTemplate, expand_abbreviations, match_columns, init_rag
from sentence_transformers import SentenceTransformer
import re

st.set_page_config(page_title="Column Matcher", layout="wide")

def create_aligned_tables(source_cols, source_expanded, dest_cols, dest_expanded):
    """Create two aligned tables showing original and expanded column names with improved styling"""
    # Create DataFrames for both tables
    source_df = pd.DataFrame({
        'Original Column': source_cols,
        'Expanded Name': source_expanded
    })
    dest_df = pd.DataFrame({
        'Original Column': dest_cols,
        'Expanded Name': dest_expanded
    })
    
    # Apply better styling
    def style_df(df):
        return df.style.set_properties(**{
            'background-color': '#f0f2f6',
            'color': '#1e1e1e',
            'border': '1px solid #e6e9ef',
            'text-align': 'left',
            'padding': '0.5rem',
            'font-size': '0.9rem'
        }).set_properties(
            subset=['Original Column'],
            **{
                'font-weight': 'bold',
                'background-color': '#e6e9ef'
            }
        ).set_table_styles([
            {'selector': 'thead th', 'props': [
                ('background-color', '#4e7496'), 
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('padding', '0.5rem')
            ]},
            {'selector': 'tbody tr:nth-of-type(odd)', 'props': [
                ('background-color', '#f8f9fa')
            ]}
        ])
    
    # Ensure both DataFrames have the same number of rows
    max_rows = max(len(source_df), len(dest_df))
    source_df = source_df.reindex(range(max_rows), fill_value='')
    dest_df = dest_df.reindex(range(max_rows), fill_value='')
    
    return style_df(source_df), style_df(dest_df)

def get_dest_options(source_col, all_comparisons):
    """Get all destination options for a source column sorted by similarity"""
    dest_options = []
    for comp in all_comparisons:
        if comp['source_column'] == source_col:
            dest_options.append({
                'column': comp['dest_column'],
                'expanded': comp['dest_expanded'],
                'score': comp['similarity_score']
            })
    # Sort options by similarity score
    dest_options.sort(key=lambda x: x['score'], reverse=True)
    return dest_options

def display_matches(matches, all_comparisons):
    """Display matches with editing capability"""
    st.subheader("Column Matching Results")
    
    # Ensure we have edited_matches in session state
    if 'edited_matches' not in st.session_state:
        st.session_state.edited_matches = matches.copy()
    
    # Ensure we have editing_state in session state
    if 'editing_state' not in st.session_state:
        st.session_state.editing_state = {i: False for i in range(len(matches))}
    
    # Get destination columns from session state
    if 'dest_df' in st.session_state:
        dest_cols = st.session_state.dest_df.columns.tolist()
    else:
        # Fallback options
        dest_cols = list(set([comp["dest_column"] for comp in all_comparisons if comp["dest_column"] != "NO_MATCH"]))
        if not dest_cols:
            dest_cols = [match["dest_column"] for match in matches if match["dest_column"] != "NO_MATCH"]

    # Add memory matches to the top of dropdown if available
    memory_options = []
    if 'editing_source_col' in st.session_state:
        try:
            from column_memory import ColumnMatchMemory
            memory = ColumnMatchMemory(MEMORY_FILE_PATH)
            src_col = st.session_state.editing_source_col
            memory_matches = memory.get_match(src_col)
            
            if memory_matches is not None:  # Check explicitly for None
                for mem_match in memory_matches:
                    if mem_match in dest_cols:
                        memory_options.append(f"🔄 {mem_match} (from memory)")
        except Exception as e:
            pass

    # Create the dropdown options
    dropdown_options = memory_options + dest_cols + ["-- No match --"]
    
    # Now display the matches
    for i, match in enumerate(st.session_state.edited_matches):
        col1, col2, col3, col4, col5, col6 = st.columns([3, 3, 2, 1, 1, 1])  # Added an extra column for No Match button
        
        with col1:
            st.text(f"{match['source_column']} ({match['source_expanded']})")
        
        with col2:
            if st.session_state.editing_state.get(i, False):
                # Set the correct default selection
                if match['dest_column'] == "NO_MATCH":
                    selected_index = len(dropdown_options) - 1  # Index of "-- No match --"
                else:
                    try:
                        selected_index = dropdown_options.index(match['dest_column'])
                    except ValueError:
                        selected_index = 0  # Default to first option if not found
                
                # Create the dropdown
                selected_dest = st.selectbox(
                    "Select destination column",
                    options=dropdown_options,
                    index=selected_index,
                    key=f"edit_dropdown_{i}"
                )
                
                # Update the match immediately when the dropdown value changes
                if selected_dest == "-- No match --":
                    st.session_state.edited_matches[i]['dest_column'] = "NO_MATCH"
                    st.session_state.edited_matches[i]['dest_expanded'] = "NO_MATCH"
                    st.session_state.edited_matches[i]['similarity_score'] = 0.0
                else:
                    st.session_state.edited_matches[i]['dest_column'] = selected_dest
                    
                    # Try to get the expanded name
                    try:
                        if 'dest_expanded' in st.session_state and 'dest_df' in st.session_state:
                            dest_idx = st.session_state.dest_df.columns.get_loc(selected_dest)
                            if dest_idx < len(st.session_state.dest_expanded):
                                st.session_state.edited_matches[i]['dest_expanded'] = st.session_state.dest_expanded[dest_idx]
                            else:
                                st.session_state.edited_matches[i]['dest_expanded'] = selected_dest
                        else:
                            st.session_state.edited_matches[i]['dest_expanded'] = selected_dest
                    except Exception:
                        st.session_state.edited_matches[i]['dest_expanded'] = selected_dest
                    
                    # Set high similarity for manually selected
                    st.session_state.edited_matches[i]['similarity_score'] = 1.0
            else:
                # Display the current match
                if match['dest_column'] == "NO_MATCH":
                    st.text("-- No match --")
                else:
                    st.text(f"{match['dest_column']} ({match['dest_expanded']})")
        
        with col3:
            st.text(f"Similarity: {match['similarity_score']:.3f}")
        
        with col4:
            # Instead of trying to access the button's state, use a callback
            if st.button("Edit", key=f"edit_button_{i}"):
                st.session_state.editing_state[i] = True
                # Store the source column being edited for memory lookup
                st.session_state.editing_source_col = match['source_column']
                st.rerun()
        
        with col5:
            if st.session_state.editing_state.get(i, False):
                if st.button("Save", key=f"save_button_{i}"):
                    st.session_state.editing_state[i] = False
                    st.rerun()
        
        with col6:
            # Add a "No Match" button that directly sets this column to have no match
            # Only show if not already set to NO_MATCH and not in editing mode
            if not st.session_state.editing_state.get(i, False) and match['dest_column'] != "NO_MATCH":
                if st.button("No Match", key=f"no_match_button_{i}", 
                            help="Mark this column as having no match in the destination table"):
                    # Set to NO_MATCH
                    st.session_state.edited_matches[i]['dest_column'] = "NO_MATCH"
                    st.session_state.edited_matches[i]['dest_expanded'] = "NO_MATCH"
                    st.session_state.edited_matches[i]['similarity_score'] = 0.0
                    st.rerun()
            # If already NO_MATCH, show a disabled button or indicator
            elif not st.session_state.editing_state.get(i, False) and match['dest_column'] == "NO_MATCH":
                st.markdown("✓ No Match")

@st.cache_resource
def load_semantic_model():
    """Load and cache the semantic model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def load_models():
    """Initialize the required models"""
    # Set API key directly from environment variable
    deepseek_model = DeepSeekLLM()  # The model will get the API key from environment
    prompt_template = PromptTemplate()
    semantic_model = load_semantic_model()
    return deepseek_model, prompt_template, semantic_model

@st.cache_data
def cached_expansion(abbreviations: list, context: str, verbose: bool = False):
    """Cached wrapper for abbreviation expansion"""
    if 'models' not in st.session_state:
        st.session_state.models = load_models()
    
    deepseek_model, prompt_template, _ = st.session_state.models
    return expand_abbreviations(
        abbreviations, context, deepseek_model, prompt_template, verbose
    )

def process_files(source_df, dest_df, context, models, use_memory=True, save_new_matches=True):
    """Process the uploaded files and match columns"""
    deepseek_model, prompt_template, semantic_model = models
    verbose = False  # Hard-code to False instead of using a checkbox
    
    # Get column names
    source_cols = source_df.columns.tolist()
    dest_cols = dest_df.columns.tolist()
    
    # Initialize column memory if enabled
    memory_matches = {}  # Default to empty dict
    column_memory = None
    if use_memory:
        try:
            from column_memory import ColumnMatchMemory
            memory_file = MEMORY_FILE_PATH
            os.makedirs(os.path.dirname(os.path.abspath(memory_file)), exist_ok=True)
            column_memory = ColumnMatchMemory(memory_file)
            
            # Get matches from memory but ensure it's not None
            found_matches = column_memory.find_matches(source_cols, dest_cols)
            if found_matches is not None:
                memory_matches = found_matches
            
            if verbose and memory_matches:
                st.write("Using matches from memory:")
                for src, dst in memory_matches.items():
                    st.write(f"- '{src}' → '{dst}'")
        except Exception as e:
            if verbose:
                st.warning(f"Error accessing column memory: {str(e)}")
            memory_matches = {}  # Ensure it's an empty dict if there was an error
    
    # Determine which columns need expansion through the LLM
    columns_to_expand_source = [col for col in source_cols if col not in memory_matches]
    
    # Expand column names using cached function
    with st.spinner('Expanding source table columns...'):
        if columns_to_expand_source:
            source_expanded_new = cached_expansion(
                columns_to_expand_source, context, verbose
            )
            
            # Create full source_expanded list by combining memory and new expansions
            source_expanded = []
            expansion_index = 0
            for col in source_cols:
                if col in memory_matches:
                    # Use a placeholder - we'll use memory for matching directly
                    source_expanded.append(f"MEMORY_MATCH:{col}")
                else:
                    source_expanded.append(source_expanded_new[expansion_index])
                    expansion_index += 1
        else:
            # If all columns are in memory, we still need a placeholder list
            source_expanded = [f"MEMORY_MATCH:{col}" for col in source_cols]
    
    with st.spinner('Expanding destination table columns...'):
        dest_expanded = cached_expansion(
            dest_cols, context, verbose
        )
        
        # Debug output if needed
        if verbose:
            st.write("Destination Columns Expansion:")
            for orig, exp in zip(dest_cols, dest_expanded):
                st.write(f"{orig} -> {exp}")
    
    # Match columns
    with st.spinner('Matching columns...'):
        matches, all_comparisons, highest_similarity_pair = match_columns(
            source_cols, dest_cols, source_expanded, dest_expanded, 
            semantic_model, verbose, memory_matches, similarity_threshold=0.3
        )
    
    # Save new matches if requested - only save valid matches, not NO_MATCH
    if use_memory and save_new_matches and column_memory:
        saved_count = 0
        for match in matches:
            source_col = match["source_column"]
            dest_col = match["dest_column"]
            # Only save matches with high confidence and not NO_MATCH
            if dest_col != "NO_MATCH" and match["similarity_score"] > 0.85 and not match.get("from_memory", False):
                column_memory.add_match(source_col, dest_col)
                saved_count += 1
        
        if verbose and saved_count > 0:
            st.info(f"Saved {saved_count} new high-confidence matches to memory.")
    
    return source_expanded, dest_expanded, matches, all_comparisons, highest_similarity_pair

def create_matched_source_file(source_df, matches):
    """Create a new DataFrame with matched column names and handle missing/extra columns"""
    matched_df = source_df.copy()
    
    # Create a mapping from source column to destination column
    # Only include columns that have a valid match (not NO_MATCH)
    column_mapping = {
        match['source_column']: match['dest_column'] 
        for match in matches 
        if match['dest_column'] != "NO_MATCH"
    }
    
    # Step 1: Rename the matched columns
    matched_df = matched_df.rename(columns=column_mapping)
    
    # Step 2: Get list of columns we're keeping as-is (marked as NO_MATCH)
    unmatched_source_columns = [
        match['source_column'] 
        for match in matches 
        if match['dest_column'] == "NO_MATCH"
    ]
    
    # Step 3: Find ALL destination columns not present in the renamed dataframe
    # Get the complete list of destination columns from the original destination DataFrame
    if 'dest_df' in st.session_state:
        all_dest_columns = st.session_state.dest_df.columns.tolist()
        # Find which destination columns are missing in our matched dataframe
        missing_dest_columns = [col for col in all_dest_columns if col not in matched_df.columns]
    else:
        # Fallback to the old approach if dest_df is not available
        dest_columns = [match['dest_column'] for match in matches if match['dest_column'] != "NO_MATCH"]
        missing_dest_columns = [col for col in dest_columns if col not in matched_df.columns]
    
    # Step 4: Add missing destination columns with "MISSING" values (uppercase as requested)
    for col in missing_dest_columns:
        matched_df[col] = "MISSING"  # Changed from "Missing" to "MISSING" as requested
    
    # Log the column operations for debugging and user information
    col_operations = {
        "matched_columns": column_mapping,
        "kept_source_columns": unmatched_source_columns,
        "added_missing_columns": missing_dest_columns
    }
    
    # For verbose output, add to session state
    st.session_state.column_operations = col_operations
    
    return matched_df

def safe_edit_match(i, selected_dest):
    """Safely edit a match with comprehensive error handling"""
    try:
        if selected_dest == "-- No match --":
            st.session_state.edited_matches[i]['dest_column'] = "NO_MATCH"
            st.session_state.edited_matches[i]['dest_expanded'] = "NO_MATCH"
            st.session_state.edited_matches[i]['similarity_score'] = 0.0
        else:
            st.session_state.edited_matches[i]['dest_column'] = selected_dest
            
            # Try multiple approaches to get the expanded name
            try:
                if 'dest_expanded' in st.session_state and 'dest_df' in st.session_state:
                    dest_idx = st.session_state.dest_df.columns.get_loc(selected_dest)
                    if dest_idx < len(st.session_state.dest_expanded):
                        st.session_state.edited_matches[i]['dest_expanded'] = st.session_state.dest_expanded[dest_idx]
                    else:
                        st.session_state.edited_matches[i]['dest_expanded'] = selected_dest
                else:
                    st.session_state.edited_matches[i]['dest_expanded'] = selected_dest
            except Exception as e:
                # If any error occurs during expanded name lookup, just use the column name
                st.session_state.edited_matches[i]['dest_expanded'] = selected_dest
                
            # Set a high similarity score for manually selected matches
            st.session_state.edited_matches[i]['similarity_score'] = 1.0
    except Exception as e:
        st.error(f"Error editing match: {str(e)}")

def clear_column_memory():
    """Function to completely clear the column matching memory"""
    memory_file = MEMORY_FILE_PATH
    
    try:
        # Write an empty dictionary to the file
        import json
        with open(memory_file, 'w') as f:
            json.dump({}, f)
        return True, "Memory cleared successfully"
    except Exception as e:
        return False, f"Error clearing memory: {str(e)}"

def main():
    st.title("LLM Based CSV Table Matcher")
    
    # Add API key check
    try:
        if hasattr(st, 'secrets') and 'DEEPSEEK_API_KEY' in st.secrets:
            # API key is configured - no need to show anything
            pass
        else:
            # Key not configured - show admin-oriented error
            st.error("⚠️ DeepSeek API key not configured. Please add it to Streamlit Secrets.")
            st.stop()
    except:
        st.error("⚠️ Error checking API key configuration.")
        st.stop()
    
    # Add deployment info
    is_cloud = os.environ.get('STREAMLIT_DEPLOYMENT', '') == 'cloud'
    if is_cloud:
        st.sidebar.success("🌐 Running on Streamlit Cloud")
    else:
        st.sidebar.info("💻 Running Locally")
    
    # Add brief introduction with better styling
    st.markdown("""
    <div style="background-color: #f0f7fb; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #4e7496; margin-bottom: 1.5rem;">
        <p style="font-size: 1.1rem; line-height: 1.6; color: #1e3a5f; margin: 0;">
            This app matches your source data table to a destination table format by using Large Language Models (LLMs) 
            to interpret column names and find semantic matches between tables.
        </p>
        <p style="font-size: 1.1rem; line-height: 1.6; color: #1e3a5f; margin-top: 1rem;">
            The app includes a Medical Abbreviation RAG (Retrieval-Augmented Generation) system that enhances column name interpretation 
            using data from medical abbreviation databases (Zhang et al., 2021, DOI: 10.1038/s41597-021-00929-4).
        </p>
        <p style="font-size: 1.1rem; line-height: 1.6; color: #1e3a5f; margin-top: 1rem;">
            The matching memory feature learns from your confirmed column matches, saving them for future use to improve 
            accuracy and efficiency across similar datasets over time.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize models and RAG
    if 'models' not in st.session_state:
        with st.spinner('Loading models...'):
            st.session_state.models = load_models()
    
    # Check for RAG availability
    rag_available = init_rag()
    
    # Add RAG status indicator
    if rag_available:
        st.sidebar.success("📚 Medical Abbreviation RAG: Enabled")
    else:
        st.sidebar.warning("📚 Medical Abbreviation RAG: Not available")
        st.sidebar.info("To enable RAG support, run the create_abbreviation_rag.py script with the Metainventory CSV file.")
    
    # Initialize session state for matched file
    if 'matched_df' not in st.session_state:
        st.session_state.matched_df = None
    if 'matching_confirmed' not in st.session_state:
        st.session_state.matching_confirmed = False
    if 'source_filename' not in st.session_state:
        st.session_state.source_filename = ""
    
    # File uploaders section with guidance
    st.markdown("### Step 1: Upload Your Data")
    st.markdown("""
    Upload the source CSV file you want to transform and the destination CSV file 
    with the target column structure.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        source_file = st.file_uploader("Upload Source CSV", type=['csv'])
        if source_file is not None and st.session_state.source_filename != source_file.name:
            st.session_state.source_filename = source_file.name
            st.session_state.matching_confirmed = False
    with col2:
        dest_file = st.file_uploader("Upload Destination CSV", type=['csv'])
    
    # Context input with explanation
    st.markdown("### Step 2: Provide Context (Optional)")
    st.markdown("""
    Providing context about your data domain helps the AI better interpret column names. 
    For example, "financial data", "clinical study", "customer records", etc.
    """)
    
    context = st.text_input(
        "Context", 
        placeholder="e.g., financial data, customer data, etc."
    )
    
    # In the sidebar
    with st.sidebar:
        st.header("App Settings")
        use_memory = st.checkbox("Use column matching memory", value=True, 
                                help="If enabled, the app will remember previous successful matches and use them for future matching.")
        save_matches = st.checkbox("Save new matches to memory", value=True,
                                  help="If enabled, new high-confidence matches will be saved to memory.")
        
        if use_memory:
            st.info("Column matching memory is enabled. The app will use previously successful matches when possible.")
            
            # Option to view and clear memory
            if st.button("View Memory Contents"):
                st.subheader("Column Matching Memory Contents")
                
                try:
                    from column_memory import ColumnMatchMemory
                    memory_file = MEMORY_FILE_PATH
                    
                    if not os.path.exists(memory_file):
                        st.warning("No memory file found. The app hasn't stored any matches yet.")
                    else:
                        memory = ColumnMatchMemory(memory_file)
                        matches = memory.get_all_matches()
                        
                        if not matches:
                            st.info("Memory file exists but contains no matches.")
                        else:
                            # Convert the memory data to a more readable format for display
                            memory_data = []
                            
                            for source_col, dest_cols in matches.items():
                                if isinstance(dest_cols, list):
                                    for dest_col in dest_cols:
                                        memory_data.append({
                                            "Source Column": source_col,
                                            "Destination Column": dest_col
                                        })
                                else:
                                    # Handle old format (single string) for backward compatibility
                                    memory_data.append({
                                        "Source Column": source_col,
                                        "Destination Column": dest_cols
                                    })
                            
                            # Create a DataFrame for better display
                            memory_df = pd.DataFrame(memory_data)
                            
                            # Add some statistics
                            total_source_cols = len(memory_df['Source Column'].unique())
                            total_mappings = len(memory_df)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Unique Source Columns", total_source_cols)
                            with col2:
                                st.metric("Total Mappings", total_mappings)
                            
                            # Display the memory contents
                            st.dataframe(memory_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading memory contents: {str(e)}")
            
            st.markdown("""
            ### Memory Management
            To clear the column matching memory, please contact the owner of this App.
            """)
        
        st.info("This app uses the DeepSeek API for column name interpretation. Usage limits may apply.")
    
    if source_file and dest_file:
        # Match button with guidance
        st.markdown("### Step 3: Analyze Tables")
        st.markdown("""
        Click below to analyze both tables. The AI will interpret column names and suggest matches.
        """)
        
        try:
            # Load DataFrames
            source_df = pd.read_csv(source_file)
            dest_df = pd.read_csv(dest_file)
            
            # Store these in session state right away
            st.session_state.source_df = source_df
            st.session_state.dest_df = dest_df
            
            # Process button
            if st.button("Match Columns"):
                # Reset confirmed status when new matching is performed
                st.session_state.matching_confirmed = False
                
                # Process files
                results = process_files(
                    source_df, dest_df, context, 
                    st.session_state.models, use_memory=use_memory,
                    save_new_matches=save_matches
                )
                source_expanded, dest_expanded, matches, all_comparisons, highest_similarity_pair = results
                
                # Store results in session state
                st.session_state.matches = matches
                st.session_state.all_comparisons = all_comparisons
                st.session_state.source_expanded = source_expanded
                st.session_state.dest_expanded = dest_expanded
                st.session_state.source_df = source_df
                st.session_state.dest_df = dest_df
                
                # Reset edited matches
                st.session_state.edited_matches = matches.copy()
                st.session_state.editing_state = {i: False for i in range(len(matches))}
            
            # Display results if available in session state
            if 'matches' in st.session_state:
                # Column name interpreter section with explanation
                st.markdown("### Step 4: Review Column Interpretations")
                st.markdown("""
                Below shows how the AI has interpreted the column names in both tables. 
                This helps understand what each column represents before matching.
                """)
                
                st.header("Column Name Interpreter")
                
                # Create a container with a styled background for the tables
                with st.container():
                    st.markdown("""
                    <style>
                    .table-container {
                        background-color: white;
                        border-radius: 5px;
                        padding: 1rem;
                        margin-bottom: 1rem;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                source_table, dest_table = create_aligned_tables(
                    source_df.columns.tolist(),
                    st.session_state.source_expanded,
                    dest_df.columns.tolist(),
                    st.session_state.dest_expanded
                )
                
                    # Add table descriptions for better clarity
                col1, col2 = st.columns(2)
                with col1:
                        st.subheader("Source Table Columns")
                        st.markdown("""
                        <div class="table-container">
                        <p style="color:#4e7496;font-style:italic;margin-bottom:0.5rem;">
                        Original column names and their interpreted meanings
                        </p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.dataframe(source_table, hide_index=True, height=400)
                        
                with col2:
                        st.subheader("Destination Table Columns")
                        st.markdown("""
                        <div class="table-container">
                        <p style="color:#4e7496;font-style:italic;margin-bottom:0.5rem;">
                        Target column names and their interpreted meanings
                        </p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.dataframe(dest_table, hide_index=True, height=400)
                
                # Matches section with guidance
                st.markdown("### Step 5: Review and Edit Matches")
                st.markdown("""
                Below are the suggested matches between source and destination columns. 
                If you need to change a match, click the "Edit" button and select from the dropdown.
                """)
                
                st.header("Semantic Matching")
                display_matches(st.session_state.matches, st.session_state.all_comparisons)
                
                # Confirm section with guidance
                st.markdown("### Step 6: Confirm and Download")
                st.markdown("""
                When you're satisfied with the matches, click "Confirm Matching" to generate 
                your source file with the new column names.
                """)
                
                # We need to fix the preview disappearing issue
                # Move display of the matched file to a separate function that's always called if matching is confirmed
                def display_matched_file():
                    if st.session_state.matching_confirmed and st.session_state.matched_df is not None:
                        st.subheader("Preview of Matched File")
                        
                        # Display column operation summary
                        if hasattr(st.session_state, 'column_operations'):
                            with st.expander("Column Operations Summary", expanded=True):
                                # Show matched columns
                                if st.session_state.column_operations["matched_columns"]:
                                    st.markdown("#### Renamed Columns")
                                    for src, dest in st.session_state.column_operations["matched_columns"].items():
                                        st.write(f"- '{src}' → '{dest}'")
                                
                                # Show kept source columns
                                if st.session_state.column_operations["kept_source_columns"]:
                                    st.markdown("#### Kept Original Columns (No Match)")
                                    for col in st.session_state.column_operations["kept_source_columns"]:
                                        st.write(f"- '{col}'")
                                
                                # Show added missing columns
                                if st.session_state.column_operations["added_missing_columns"]:
                                    st.markdown("#### Added Missing Destination Columns")
                                    for col in st.session_state.column_operations["added_missing_columns"]:
                                        st.write(f"- '{col}' (filled with 'MISSING')")
                        
                        # Display the dataframe preview with highlighted columns
                        st.markdown("#### Preview of Matched Data")
                        
                        # Get lists of columns for highlighting
                        if hasattr(st.session_state, 'column_operations'):
                            kept_columns = st.session_state.column_operations["kept_source_columns"]
                            missing_columns = st.session_state.column_operations["added_missing_columns"]
                            
                            # Apply styling to show different column types
                            def highlight_columns(df):
                                styles = pd.DataFrame('', index=df.index, columns=df.columns)
                                
                                # Highlight kept original columns
                                for col in kept_columns:
                                    if col in df.columns:
                                        styles[col] = 'background-color: #ffffcc'  # Light yellow
                                
                                # Highlight added missing columns
                                for col in missing_columns:
                                    if col in df.columns:
                                        styles[col] = 'background-color: #ffcccc'  # Light red
                                
                                return styles
                            
                            # Display styled dataframe
                            st.dataframe(
                                st.session_state.matched_df.head(10).style.apply(highlight_columns, axis=None),
                                use_container_width=True
                            )
                        else:
                            # Basic display without styling
                            st.dataframe(st.session_state.matched_df.head(10), hide_index=True)
                        
                        st.markdown("""
                        You can now download your transformed data with the new column names.
                        Enter a filename or use the suggested default.
                        """)
                        
                        # Get default filename
                        default_filename = os.path.splitext(st.session_state.source_filename)[0]
                        default_filename = f"{default_filename}_Matched.csv" if default_filename else "matched_source.csv"
                        
                        # File name input
                        download_filename = st.text_input(
                            "Enter filename for download:", 
                            value=default_filename,
                            key="download_filename"
                        )
                        
                        # Ensure filename has .csv extension
                        if not download_filename.endswith('.csv'):
                            download_filename += '.csv'
                        
                        # Download button - Fix: Correctly indent this inside the function
                        st.download_button(
                            f"Download {download_filename}",
                            st.session_state.matched_df.to_csv(index=False).encode('utf-8'),
                            download_filename,
                            "text/csv",
                            key='download-matched-file'
                        )
                
                # Confirm matching button
                if st.button("Confirm Matching"):
                    st.session_state.matching_confirmed = True
                    # Create the matched source file
                    st.session_state.matched_df = create_matched_source_file(
                        st.session_state.source_df, 
                        st.session_state.edited_matches
                    )
                    
                    # Save edits to memory if enabled
                    if use_memory and save_matches:
                        try:
                            from column_memory import ColumnMatchMemory
                            memory_file = MEMORY_FILE_PATH
                            column_memory = ColumnMatchMemory(memory_file)
                            
                            # Save all confirmed matches
                            saved_count = 0
                            for match in st.session_state.edited_matches:
                                source_col = match["source_column"]
                                dest_col = match["dest_column"]
                                # Save user-confirmed matches with maximum confidence
                                if dest_col != "NO_MATCH":  # Only save actual matches, not NO_MATCH
                                    column_memory.add_match(source_col, dest_col)
                                    saved_count += 1
                            
                            st.success(f"Matching confirmed! Your source data has been transformed and {saved_count} matches saved to memory.")
                        except Exception as e:
                            st.warning(f"Could not save matches to memory: {str(e)}")
                            st.success("Matching confirmed! Your source data has been transformed.")
                    else:
                        st.success("Matching confirmed! Your source data has been transformed.")
                    
                # Always call the display function after the button
                # This ensures it stays visible regardless of other interactions
                if st.session_state.matching_confirmed:
                    display_matched_file()
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
    
    else:
        st.info("Please upload both source and destination CSV files to begin.")

if __name__ == "__main__":
    main()