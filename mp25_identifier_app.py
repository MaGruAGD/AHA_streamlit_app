import streamlit as st
import pandas as pd
import re

def extract_mp25_codes(df):
    DEFAULT_ALLOWED_CODES = [
        "ANACHL", "BLT412", "BLT3", "BLT8", "CHLSU", "CHLPS", "CLPERF",
        "DIVPR", "ECOLI", "ECORU", "ECF18Q", "EHDV", "HMELE", "HSOMN",
        "LEPTO", "MGMS", "MYCBO", "MTGPS", "MSUIS", "PASMHY", "PCVT",
        "PCV2", "PRRS", "PTBC", "SALDI", "SRPR3", "SRPR", "TETRA", "TOXO",
        "A2", "BHBP", "BVD", "CEM", "CYSU", "INFA", "MHYO", "MSH", "PARVO",
        "PIA", "SALM", "BLT"
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
    st.set_page_config(page_title="MP25 Filter", page_icon="🔬", layout="wide")

    # Minimal, modern CSS for compact layout
    st.markdown("""
        <style>
            html, body, [class*="css"] {
                font-family: system-ui, sans-serif;
                font-size: 14px;
            }
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 1rem;
            }
            .header {
                font-size: 1.8rem;
                font-weight: 600;
                margin-bottom: 1rem;
            }
            .section {
                padding: 1rem;
                border-radius: 6px;
                background-color: #f8f9fa;
                border: 1px solid #e0e0e0;
                margin-top: 1.5rem;
            }
            .code-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
                gap: 0.5rem;
                margin-top: 0.5rem;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="header">🔬 MP25 Code Filter</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV file with MP25 codes", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            codes = extract_mp25_codes(df)

            if codes:
                st.markdown('<div class="section">', unsafe_allow_html=True)

                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("✅ Select All"):
                        for c in codes:
                            st.session_state[f"code_{c}"] = True
                    if st.button("❌ Clear All"):
                        for c in codes:
                            st.session_state[f"code_{c}"] = False

                with col2:
                    st.markdown("**Select codes to keep:**", unsafe_allow_html=True)
                    selected = []
                    st.markdown('<div class="code-grid">', unsafe_allow_html=True)
                    for c in codes:
                        if st.checkbox(c, key=f"code_{c}"):
                            selected.append(c)
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                if selected:
                    filtered_df = filter_csv_by_codes(df, selected)
                    st.success(f"Filtered dataset contains {len(filtered_df)} row(s).")

                    st.download_button(
                        "📥 Download Filtered CSV",
                        filtered_df.to_csv(index=False),
                        file_name=f"filtered_{uploaded_file.name}",
                        mime="text/csv"
                    )

                    with st.expander("🔍 Preview filtered data"):
                        st.dataframe(filtered_df, use_container_width=True)
                else:
                    st.info("Select at least one code to enable filtering.")
            else:
                st.warning("No MP25 codes found in your file.")

        except Exception as e:
            st.error("Failed to process file. Please upload a valid CSV.")

    st.markdown("<hr style='margin-top:2rem;'>", unsafe_allow_html=True)
    st.caption("Built with ❤️ using Streamlit")

if __name__ == "__main__":
    main()
