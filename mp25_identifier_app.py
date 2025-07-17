import streamlit as st
import pandas as pd
import re

# --- Extract codes from CSV ---
def extract_mp25_codes(df):
    DEFAULT_ALLOWED_CODES = [
        "ANACHL", "BLT412", "BLT3", "BLT8", "CHLSU", "CHLPS", "CLPERF",
        "DIVPR", "ECOLI", "ECORU", "ECF18Q", "EHDV", "HMELE", "HSOMN",
        "LEPTO", "MGMS", "MYCBO", "MTGPS", "MSUIS", "PASMHY", "PCVT",
        "PCV2", "PRRS", "PTBC", "SALDI", "SRPR3", "SRPR", "TETRA", "TOXO",
        "A2", "BHBP", "BVD", "CEM", "CYSU", "INFA", "MHYO", "MSH", "PARVO",
        "PIA", "SALM", "BLT"
    ]
    pattern = r'MP25(' + '|'.join(DEFAULT_ALLOWED_CODES) + r')[A-Z0-9]*'
    all_text = df.astype(str).agg(' '.join, axis=1).str.cat(sep=' ')
    matches = re.findall(pattern, all_text)
    return sorted(set(matches))

# --- Filter based on selected codes ---
def filter_csv_by_codes(df, selected_codes):
    if not selected_codes:
        return pd.DataFrame()
    pattern = r'MP25(' + '|'.join(re.escape(code) for code in selected_codes) + r')[A-Z0-9]*'
    mask = df.astype(str).apply(lambda x: x.str.contains(pattern, regex=True, na=False)).any(axis=1)
    return df[mask]

# --- MAIN APP ---
def main():
    st.set_page_config(page_title="MP25 Filter", page_icon="🔬", layout="wide")

    # --- Modern UI CSS ---
    st.markdown("""
        <style>
            html, body, [class*="css"] {
                font-family: 'Segoe UI', Tahoma, sans-serif;
                background: linear-gradient(to right, #f0f4ff, #ffffff);
                font-size: 14px;
            }

            .header {
                font-size: 1.6rem;
                font-weight: 700;
                color: #003366;
                margin-bottom: 1rem;
            }

            .upload-box {
                background: rgba(255, 255, 255, 0.85);
                border-radius: 12px;
                padding: 1.2rem;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
                border: 1px solid #dce6f0;
            }

            .code-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
                gap: 0.4rem;
                margin-top: 0.5rem;
            }

            .stButton button {
                border-radius: 6px;
                padding: 0.4rem 0.75rem;
                font-size: 13px;
                background-color: #003366;
                color: white;
            }

            .stButton button:hover {
                background-color: #00509e;
                transform: scale(1.01);
                transition: 0.2s;
            }

            .footer {
                font-size: 0.8rem;
                text-align: center;
                margin-top: 2rem;
                color: #777;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="header">🔬 MP25 Code Filter</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📂 Upload CSV with MP25 codes", type="csv")
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.error("Uploaded CSV appears to be empty.")
                return

            codes = extract_mp25_codes(df)

            if not codes:
                st.warning("⚠️ No valid MP25 codes found in your file.")
                return

            # Layout: Filters on left, table on right
            col_filters, col_data = st.columns([1.2, 3])

            with col_filters:
                with st.expander("🎯 Filter by Codes", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Select All"):
                            for c in codes:
                                st.session_state[f"code_{c}"] = True
                    with col2:
                        if st.button("❌ Clear All"):
                            for c in codes:
                                st.session_state[f"code_{c}"] = False

                    selected = []
                    st.markdown('<div class="code-grid">', unsafe_allow_html=True)
                    for c in sorted(codes):
                        if f"code_{c}" not in st.session_state:
                            st.session_state[f"code_{c}"] = False
                        if st.checkbox(c, key=f"code_{c}"):
                            selected.append(c)
                    st.markdown('</div>', unsafe_allow_html=True)

            with col_data:
                if selected:
                    filtered_df = filter_csv_by_codes(df, selected)
                    st.success(f"✅ Filtered dataset: {len(filtered_df)} row(s)")

                    st.download_button(
                        "📥 Download Filtered CSV",
                        filtered_df.to_csv(index=False),
                        file_name=f"filtered_{uploaded_file.name}",
                        mime="text/csv"
                    )

                    st.dataframe(filtered_df, use_container_width=True, height=500)
                else:
                    st.info("👈 Select at least one code to view filtered data.")

        except Exception as e:
            st.error(f"💥 Error: {e}")

    st.markdown("<div class='footer'>🚀 Built with ❤️ using Streamlit</div>", unsafe_allow_html=True)


# --- Run App ---
if __name__ == "__main__":
    main()
