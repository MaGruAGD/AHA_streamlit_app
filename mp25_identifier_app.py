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

def analyze_mp25_distribution(df):
    """
    Analyze the distribution of MP25 codes across columns
    """
    mp25_distribution = {}
    
    for column in df.columns:
        column_codes = set()
        for value in df[column].astype(str):
            matches = re.findall(r'MP25[A-Z0-9]+', value)
            column_codes.update(matches)
        
        if column_codes:
            mp25_distribution[column] = sorted(list(column_codes))
    
    return mp25_distribution

def main():
    st.title("MP25 Code Identifier")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Extract MP25 codes
            mp25_codes = extract_mp25_codes(df)
            
            if mp25_codes:
                # Display codes only
                for code in mp25_codes:
                    st.write(code)
                
            else:
                st.warning("No MP25 codes found in the uploaded file.")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please make sure the file is a valid CSV format.")

if __name__ == "__main__":
    main()
