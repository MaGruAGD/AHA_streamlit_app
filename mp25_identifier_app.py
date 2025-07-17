import streamlit as st
import pandas as pd
import re
from io import StringIO

def extract_mp25_codes(df):
    """
    Extract MP25 codes from the dataframe and return only the allowed code sections
    """
    DEFAULT_ALLOWED_CODES = [
        "ANACHL", "BLT412", "BLT3", "BLT8", "CHLSU", "CHLPS",
        "CLPERF", "DIVPR", "ECOLI", "ECORU", "ECF18Q", "EHDV", "HMELE",
        "HSOMN", "LEPTO", "MGMS", "MYCBO", "MTGPS", "MSUIS", "PASMHY", 
        "PCVT", "PCV2", "PRRS", "PTBC", "SALDI", "SRPR3", "SRPR", 
        "TETRA", "TOXO", "A2", "BHBP", "BVD", "CEM", "CYSU", 
        "INFA", "MHYO", "MSH", "PARVO", "PIA", "SALM", "BLT"
    ]
    
    found_codes = set()
    
    # Search through all columns for MP25 codes
    for column in df.columns:
        for value in df[column].astype(str):
            # Find all MP25 codes in the value using regex
            matches = re.findall(r'MP25[A-Z0-9]+', value)
            for match in matches:
                # Extract the code section after MP25
                code_section = match[4:]  # Remove "MP25" prefix
                
                # Check if this code section starts with any allowed code
                for allowed_code in DEFAULT_ALLOWED_CODES:
                    if code_section.startswith(allowed_code):
                        found_codes.add(allowed_code)
                        break
    
    return sorted(list(found_codes))

def main():
    # Page configuration
    st.set_page_config(
        page_title="MP25 Code Identifier",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for professional styling
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 2rem 0 1rem 0;
            color: #1f2937;
            font-weight: 600;
        }
        
        .upload-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 12px;
            margin: 2rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .upload-text {
            color: white;
            text-align: center;
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }
        
        .results-container {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            margin: 2rem 0;
        }
        
        .results-header {
            color: #1f2937;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        
        .code-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .code-item {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            font-family: 'Courier New', monospace;
            font-weight: 600;
            font-size: 0.95rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease;
        }
        
        .code-item:hover {
            transform: translateY(-2px);
        }
        
        .no-codes {
            text-align: center;
            color: #6b7280;
            font-size: 1.1rem;
            padding: 2rem;
            background: #f9fafb;
            border-radius: 8px;
            border: 2px dashed #d1d5db;
        }
        
        .footer {
            text-align: center;
            margin-top: 3rem;
            color: #6b7280;
            font-size: 0.9rem;
        }
        
        /* Hide Streamlit default elements */
        .css-1d391kg {
            padding-top: 1rem;
        }
        
        .stFileUploader > div > div > div > div {
            background: rgba(255, 255, 255, 0.1);
            border: 2px dashed rgba(255, 255, 255, 0.3);
            border-radius: 8px;
        }
        
        .stFileUploader > div > div > div > div > div {
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🔬 MP25 Code Identifier</h1>', unsafe_allow_html=True)
    
    # Upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<p class="upload-text">Upload your CSV file to identify MP25 codes</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload a CSV file containing MP25 codes to analyze"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Extract MP25 codes
            mp25_codes = extract_mp25_codes(df)
            
            # Results section
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            
            if mp25_codes:
                st.markdown('<h2 class="results-header">Identified Codes</h2>', unsafe_allow_html=True)
                
                # Create code grid
                cols = st.columns(4)
                for i, code in enumerate(mp25_codes):
                    with cols[i % 4]:
                        st.markdown(f'<div class="code-item">{code}</div>', unsafe_allow_html=True)
                
                # Add some spacing
                st.markdown('<br>', unsafe_allow_html=True)
                
                # Summary
                st.markdown(f'<p style="text-align: center; color: #6b7280; margin-top: 2rem;">Found {len(mp25_codes)} unique code{"s" if len(mp25_codes) != 1 else ""}</p>', unsafe_allow_html=True)
                
            else:
                st.markdown('<div class="no-codes">No MP25 codes found in the uploaded file</div>', unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error("⚠️ Error processing file. Please ensure it's a valid CSV format.")
            
    # Footer
    st.markdown('<div class="footer">Powered by Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
