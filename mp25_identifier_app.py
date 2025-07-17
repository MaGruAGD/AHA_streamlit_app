def main():
    st.set_page_config(page_title="MP25 Filter", page_icon="🔬", layout="wide")

    # Add wow-factor CSS
    st.markdown("""
        <style>
            @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css');

            html, body, [class*="css"] {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 15px;
                background: radial-gradient(circle at top left, #f0f8ff, #ffffff);
            }

            .header {
                font-size: 2.4rem;
                font-weight: 800;
                color: #003366;
                margin-bottom: 1rem;
            }

            .upload-box {
                background: rgba(255, 255, 255, 0.85);
                border-radius: 16px;
                padding: 2rem;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
                border: 1px solid #dce6f0;
                backdrop-filter: blur(8px);
            }

            .section {
                background: rgba(240, 248, 255, 0.65);
                padding: 1.5rem;
                margin-top: 2rem;
                border-radius: 14px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.08);
                transition: all 0.3s ease-in-out;
            }

            .code-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
                gap: 0.5rem;
                margin-top: 1rem;
                transition: all 0.4s ease;
            }

            .stButton button {
                border-radius: 8px;
                background-color: #003366;
                color: white;
                font-weight: bold;
                transition: 0.2s ease;
            }

            .stButton button:hover {
                background-color: #00509e;
                transform: scale(1.02);
            }

            .stCheckbox > label {
                font-weight: 600;
                color: #003366;
            }

            .footer {
                font-size: 0.9rem;
                text-align: center;
                margin-top: 3rem;
                color: #666;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="header">🔬 MP25 Code Filter</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📂 Upload your CSV file with MP25 codes", type="csv")
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.error("Uploaded CSV appears to be empty.")
                return

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
                    st.markdown("**🎯 Select codes to filter:**", unsafe_allow_html=True)
                    selected = []
                    st.markdown('<div class="code-grid">', unsafe_allow_html=True)
                    for c in sorted(codes):
                        if f"code_{c}" not in st.session_state:
                            st.session_state[f"code_{c}"] = False
                        if st.checkbox(c, key=f"code_{c}"):
                            selected.append(c)
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                if selected:
                    filtered_df = filter_csv_by_codes(df, selected)
                    st.success(f"🎉 Filtered dataset contains {len(filtered_df)} row(s).")

                    st.download_button(
                        "📥 Download Filtered CSV",
                        filtered_df.to_csv(index=False),
                        file_name=f"filtered_{uploaded_file.name}",
                        mime="text/csv"
                    )

                    with st.expander("🔍 Preview filtered data"):
                        st.dataframe(filtered_df, use_container_width=True)
                else:
                    st.info("👈 Select at least one code to enable filtering.")
            else:
                st.warning("⚠️ No valid MP25 codes found in the file.")

        except Exception as e:
            st.error(f"💥 Failed to process file. Error: {e}")

    st.markdown("<div class='footer'>🚀 Built with ❤️ using Streamlit</div>", unsafe_allow_html=True)
