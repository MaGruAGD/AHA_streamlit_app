import streamlit as st
import pandas as pd
import re
from io import StringIO

def extract_mp25_codes(df):
    DEFAULT_ALLOWED_CODES = [
        "ANACHL", "BLT412", "BLT3", "BLT8", "CHLSU", "CHLPS",
        "CLPERF", "DIVPR", "ECOLI", "ECORU", "ECF18Q", "EHDV", "HMELE",
        "HSOMN", "LEPTO", "MGMS", "MYCBO", "MTGPS", "MSUIS", "PASMHY", 
        "PCVT", "PCV2", "PRRS", "PTBC", "SALDI", "SRPR3", "SRPR", 
        "TETRA", "TOXO", "A2", "BHBP", "BVD", "CEM", "CYSU", 
        "INFA", "MHYO", "MSH", "PARVO", "PIA", "SALM", "BLT"
    ]
    
    found_codes = set()
    for column in df.columns:
        for value in df[column].astype(str):
            matches = re.findall(r'MP25[A-Z0-9]+', value)
            for match in matches:
                code_section = match[4:]
                for allowed_code in DEFAULT_ALLOWED_CODES:
                    if code_section.startswith(allowed_code):
                        found_codes.add(allowed_code)
                        break
    return sorted(list(found_codes))

def filter_csv_by_codes(df, selected_codes):
    if not selected_codes:
        return pd.DataFrame()
    
    pattern = r'MP25(' + '|'.join(re.escape(code) for code in selected_codes) + r')[A-Z0-9]*'
    mask = df.astype(str).apply(lambda x: x.str.contains(pattern, regex=True, na=False)).any(axis=1)
    return df[mask]

def main():
    st.set_page_config(
        page_title="MP25 Code Filter",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

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
        .selection-section {
            background: #f8fafc;
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1.5rem 0;
            border: 1px solid #e2e8f0;
        }
        .code-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
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
        .download-section {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            padding: 2rem;
            border-radius: 12px;
            margin: 2rem 0;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .download-text {
            color: white;
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }
        .footer {
            text-align: center;
            margin-top: 3rem;
            color: #6b7280;
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🔬 MP25 Code Filter</h1>', unsafe_allow_html=True)

    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<p class="upload-text">Upload your CSV file to identify and filter MP25 codes</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload a CSV file containing MP25 codes to analyze")

    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            mp25_codes = extract_mp25_codes(df)

            st.markdown('<div class="results-container">', unsafe_allow_html=True)

            if mp25_codes:
                st.markdown('<h2 class="results-header">Select Codes to Keep</h2>', unsafe_allow_html=True)
                st.markdown(f"**Available codes:** {', '.join(mp25_codes)}")

                st.markdown('<div class="selection-section">', unsafe_allow_html=True)
                st.markdown("**Choose which codes to keep in the filtered CSV:**")

                # Select All / Clear All
                cols = st.columns([1, 1])
                if cols[0].button("✅ Select All"):
                    for code in mp25_codes:
                        st.session_state[f"code_{code}"] = True
                if cols[1].button("❌ Clear All"):
                    for code in mp25_codes:
                        st.session_state[f"code_{code}"] = False

                # Display checkboxes
                selected_codes = []
                st.markdown('<div class="code-grid">', unsafe_allow_html=True)
                for code in mp25_codes:
                    if st.checkbox(code, key=f"code_{code}"):
                        selected_codes.append(code)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                if selected_codes:
                    filtered_df = filter_csv_by_codes(df, selected_codes)
                    st.markdown('<div class="download-section">', unsafe_allow_html=True)
                    st.markdown(f'<p class="download-text">Filtered data ready! Found {len(filtered_df)} rows with selected codes.</p>', unsafe_allow_html=True)

                    csv_output = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Filtered CSV",
                        data=csv_output,
                        file_name=f"filtered_{uploaded_file.name}",
                        mime="text/csv",
                        help="Download the filtered CSV with only selected codes"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                    with st.expander("🔍 Preview Filtered Data"):
                        st.dataframe(filtered_df, use_container_width=True)
                        st.info(f"Showing {len(filtered_df)} of {len(df)} original rows")
                else:
                    st.info("👆 Use the checkboxes above to select codes for filtering.")
            else:
                st.markdown('<div class="no-codes">No MP25 codes found in the uploaded file</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error("⚠️ Error processing file. Please ensure it's a valid CSV format.")

    st.markdown('<div class="footer">Powered by Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
