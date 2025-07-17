import streamlit as st
import pandas as pd
import re
from io import StringIO

def extract_mp25_codes(df):
    """
    Extract all MP25 codes from the dataframe
    """
    mp25_codes = set()
    
    # Search through all columns for MP25 codes
    for column in df.columns:
        for value in df[column].astype(str):
            # Find all MP25 codes in the value using regex
            matches = re.findall(r'MP25[A-Z0-9]+', value)
            mp25_codes.update(matches)
    
    return sorted(list(mp25_codes))

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
    st.markdown("Upload a CSV file to identify and analyze MP25 codes")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Display basic info about the file
            st.subheader("File Information")
            st.write(f"**Rows:** {len(df)}")
            st.write(f"**Columns:** {len(df.columns)}")
            st.write(f"**Column Names:** {', '.join(df.columns)}")
            
            # Show first few rows
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Extract MP25 codes
            mp25_codes = extract_mp25_codes(df)
            
            if mp25_codes:
                st.subheader("MP25 Codes Found")
                st.write(f"**Total unique MP25 codes found:** {len(mp25_codes)}")
                
                # Display codes in a nice format
                cols = st.columns(3)
                for i, code in enumerate(mp25_codes):
                    with cols[i % 3]:
                        st.code(code)
                
                # Show distribution analysis
                st.subheader("MP25 Code Distribution by Column")
                distribution = analyze_mp25_distribution(df)
                
                for column, codes in distribution.items():
                    with st.expander(f"{column} ({len(codes)} codes)"):
                        for code in codes:
                            st.write(f"• {code}")
                
                # Create a summary table
                st.subheader("Summary Statistics")
                summary_data = []
                for column, codes in distribution.items():
                    summary_data.append({
                        'Column': column,
                        'Number of MP25 Codes': len(codes),
                        'Sample Codes': ', '.join(codes[:3]) + ('...' if len(codes) > 3 else '')
                    })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df)
                
                # Download option for extracted codes
                st.subheader("Download Results")
                codes_text = '\n'.join(mp25_codes)
                st.download_button(
                    label="Download MP25 codes as text file",
                    data=codes_text,
                    file_name="mp25_codes.txt",
                    mime="text/plain"
                )
                
            else:
                st.warning("No MP25 codes found in the uploaded file.")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please make sure the file is a valid CSV format.")

if __name__ == "__main__":
    main()