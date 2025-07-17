import streamlit as st
import pandas as pd
import re

# Optional: pip install streamlit-theme-toggle
from streamlit_theme_toggle import st_theme_toggle

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

    # --- Dark Mode Toggle ---
    dark_mode = st_theme_toggle(toggle_label="🌓 Dark Mode", default_theme="light")
    if dark_mode:
        st.markdown("""
            <style>
                body, html {
                    background-color: #1e1e1e !important;
                    color: #dddddd !important;
                }
                .stButton button {
                    background-color: #333 !important;
                    color: #fff !important;
                }
                .stCheckbox > div {
                    color: #fff !important;
                }
            </style>
        """, unsafe_allow_html=True)

    # --- Custom Styles ---
    st.markdown("""
        <style>
            html, body, [class*="css"] {
                font-family: 'Segoe UI', Tahoma, sans-serif;
                font-size: 14px;
            }
            .header {
                font-size: 1.6rem;
                font-weight: 700;
                color: #003366;
                margin-bottom: 1rem;
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
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="header">🔬 MP25 Code Filter</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload CSV with MP25 codes", type="csv")

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

            if selected:
                filtered_df = filter_csv_by_codes(df, selected)
                st.success(f"✅ Filtered dataset: {len(filtered_df)} row(s)")

                st.download_button(
                    "📥 Download Filtered CSV",
                    filtered_df.to_csv(index=False),
                    file_name=f"filtered_{uploaded_file.name}",
                    mime="text/csv"
                )
            else:
                st.info("☝️ Select at least one code to enable download.")

        except Exception as e:
            st.error(f"💥 Error: {e}")

    st.markdown("---")
    st.caption("Made with ❤️ and Streamlit")


if __name__ == "__main__":
    main()
