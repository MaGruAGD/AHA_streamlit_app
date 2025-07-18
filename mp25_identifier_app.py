import streamlit as st
import pandas as pd
import re
import io
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="AHA! - Andrew Helper App",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database from the attached file
MERGED_DATABASE = {
    "ANACHL": {
        "allowed": True,
        "control_samples": {
            "pcs_ana": {"position": "G6", "name": "PCS ANA"},
            "ncs_ana": {"position": "H6", "name": "NCS ANA"},
            "pcs_chlam": {"position": "G12", "name": "PCS CHLAM"},
            "ncs_chlam": {"position": "H12", "name": "NCS CHLAM"}
        }
    },
    "A2": {
        "allowed": True,
        "control_samples": {
            "pcs": {"position": "G12", "name": "PCS"},
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "BHBP": {
        "allowed": True,
        "control_samples": {
            "ncs_hyo": {"position": "H9", "name": "NCS HYO"},
            "ncs_pilo": {"position": "H12", "name": "NCS PILO"}
        }
    },
    "BLT": {
        "allowed": True,
        "control_samples": {
            "ref": {"position": "F12", "name": "REF"},
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "BLT3": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "BLT8": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "BLT412": {
        "allowed": True,
        "control_samples": {
            "ncs_4": {"position": "H6", "name": "NCS 4"},
            "ncs_12": {"position": "H12", "name": "NCS 12"}
        }
    },
    "BVD": {
        "allowed": True,
        "control_samples": {
            "ref": {"position": "F12", "name": "REF"},
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "CEM": {
        "allowed": True,
        "control_samples": {
            "ref": {"position": "F12", "name": "REF"},
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "CYSU": {
        "allowed": True,
        "control_samples": {
            "pcs": {"position": "G12", "name": "PCS"},
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "CHLSU": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "CHLPS": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "CLPERF": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "DIVPR": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "ECOLI": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "A12", "name": "NCS"}
        }
    },
    "ECORU": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "A12", "name": "NCS"}
        }
    },
    "EHDV": {
        "allowed": True,
        "control_samples": {
            "ref": {"position": "F12", "name": "REF"},
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "HMELE": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "HSOMN": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "INFA": {
        "allowed": True,
        "control_samples": {
            "ref": {"position": "F12", "name": "REF"},
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "LEPTO": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "MGMS": {
        "allowed": True,
        "control_samples": {
            "ref": {"position": "E12", "name": "REF"},
            "pcs_mg": {"position": "F12", "name": "PCS MG"},
            "pcs_ms": {"position": "G12", "name": "PCS MS"},
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "MYCBO": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "MHYO": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "MSH": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "MTGPS": {
        "allowed": True,
        "control_samples": {
            "ncs_mt": {"position": "H8", "name": "NCS MT"},
            "ncs_gps": {"position": "H12", "name": "NCS GPS"}
        }
    },
    "MSUIS": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "PARVO": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "PASMHY": {
        "allowed": True,
        "control_samples": {
            "ncs_past": {"position": "H6", "name": "NCS PAST"},
            "ncs_mhyo": {"position": "H12", "name": "NCS MHYO"}
        }
    },
    "PCV2": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "PCVT": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "PIA": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "PRRS": {
        "allowed": True,
        "control_samples": {
            "ref": {"position": "E12", "name": "REF"},
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "PTBC": {
        "allowed": True,
        "control_samples": {
            "ref": {"position": "F12", "name": "REF"},
            "pcs": {"position": "G12", "name": "PCS"},
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "SALDI": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "SALM": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "SRPR": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "B12", "name": "NCS"}
        }
    },
    "SRPR3": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "B12", "name": "NCS"}
        }
    },
    "TETRA": {
        "allowed": True,
        "control_samples": {
            "ncs_crypto": {"position": "H4", "name": "NCS crypto"},
            "ncs_rota": {"position": "H8", "name": "NCS rota"},
            "ncs_corona": {"position": "H12", "name": "NCS corona"}
        }
    },
    "TOXO": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "H12", "name": "NCS"}
        }
    },
    "ECF18Q": {
        "allowed": True,
        "control_samples": {
            "ncs": {"position": "H12", "name": "NCS"}
        }
    }
}

# Extract allowed codes from the database
DEFAULT_ALLOWED_CODES = sorted([code for code, data in MERGED_DATABASE.items() if data["allowed"]])

# Convert control samples to the format expected by the original code
CONTROL_SAMPLES = {}
for code, data in MERGED_DATABASE.items():
    if data["allowed"] and "control_samples" in data:
        control_samples = data["control_samples"]
        positions = []
        names = []
        
        for control_id, control_info in control_samples.items():
            positions.append(control_info["position"])
            names.append(control_info["name"])
        
        CONTROL_SAMPLES[code] = {
            'positions': positions,
            'names': names
        }

# Custom default volumes
CUSTOM_DEFAULTS = {
    'SRPR': 50,
    'SRPR3': 40,
    'ECORU': 50,
    'BLT3': 10,
    'BLT8': 10,
    'BLT412': 10
}

class CSVProcessor:
    def __init__(self, df):
        self.original_df = df.copy()
        self.df = df.copy()
        self.codes = self.extract_codes()
        
    def extract_codes(self):
        """Extract MP25 and PP25 codes from the CSV data"""
        codes = set()
        
        # Sort codes by length (longest first) to handle overlapping codes correctly
        sorted_codes = sorted(DEFAULT_ALLOWED_CODES, key=len, reverse=True)
        
        # Create pattern that matches the longest codes first
        for code in sorted_codes:
            pattern = r'(?:MP25|PP25)' + re.escape(code) + r'(?:\d+)?'
            
            for col in self.df.columns:
                for value in self.df[col].astype(str):
                    if re.search(pattern, value):
                        codes.add(code)
        
        return sorted(list(codes))
    
    def reset_data(self):
        """Reset data to original state"""
        self.df = self.original_df.copy()
        
    def add_row(self, row_data):
        """Add a new row to the dataframe"""
        self.df = pd.concat([self.df, pd.DataFrame([row_data])], ignore_index=True)
    
    def filter_data(self, selected_codes, run_number):
        """Filter data based on selected codes and run number"""
        filtered_df = self.df.copy()
        
        # Create regex pattern for selected codes
        if selected_codes:
            # Sort selected codes by length (longest first) to handle overlapping codes
            sorted_selected = sorted(selected_codes, key=len, reverse=True)
            patterns = []
            
            for code in sorted_selected:
                patterns.append(r'(?:MP25|PP25)' + re.escape(code) + r'(?:\d+)?')
            
            # Combine all patterns
            combined_pattern = '|'.join(patterns)
            
            # Filter rows that contain any of the selected codes
            mask = filtered_df.astype(str).apply(
                lambda x: x.str.contains(combined_pattern, regex=True, na=False)
            ).any(axis=1)
            
            filtered_df = filtered_df[mask]
        
        return filtered_df
    
    def apply_volumes(self, df, volumes):
        """Apply custom volumes to the dataframe"""
        df_copy = df.copy()
        
        # Sort codes by length (longest first) to handle overlapping codes
        sorted_codes = sorted(volumes.keys(), key=len, reverse=True)
        
        for code in sorted_codes:
            volume = volumes[code]
            pattern = r'(?:MP25|PP25)' + re.escape(code) + r'(?:\d+)?'
            
            mask = df_copy.astype(str).apply(
                lambda x: x.str.contains(pattern, regex=True, na=False)
            ).any(axis=1)
            
            if mask.any():
                df_copy.loc[mask, df_copy.columns[8]] = volume
        
        return df_copy

def position_to_sample_number(position):
    """Convert well position (A1, B2, etc.) to sample number (1-96)"""
    if len(position) < 2:
        return 1
    
    row = position[0].upper()
    try:
        col = int(position[1:])
    except ValueError:
        return 1
    
    row_num = ord(row) - ord('A') + 1
    return (row_num - 1) * 12 + col

def create_96_well_plate_selector(key_prefix, selected_position="A1"):
    """Create a 96-well plate selector widget"""
    st.write("96-Well Plate Selector:")
    
    # Initialize session state for plate selector
    if f"{key_prefix}_selected" not in st.session_state:
        st.session_state[f"{key_prefix}_selected"] = selected_position
    
    # Create grid layout
    cols = st.columns(13)  # 12 columns + 1 for row labels
    
    # Column headers
    with cols[0]:
        st.write("")
    for i in range(1, 13):
        with cols[i]:
            st.write(f"**{i}**")
    
    # Create wells
    for row_idx, row in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']):
        cols = st.columns(13)
        
        # Row label
        with cols[0]:
            st.write(f"**{row}**")
        
        # Wells
        for col_idx in range(1, 13):
            with cols[col_idx]:
                well_position = f"{row}{col_idx}"
                is_selected = st.session_state[f"{key_prefix}_selected"] == well_position
                
                if st.button(
                    well_position,
                    key=f"{key_prefix}_well_{well_position}",
                    type="primary" if is_selected else "secondary",
                    use_container_width=True
                ):
                    st.session_state[f"{key_prefix}_selected"] = well_position
                    st.rerun()
    
    return st.session_state[f"{key_prefix}_selected"]

def initialize_session_state():
    """Initialize session state variables"""
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'num_runs' not in st.session_state:
        st.session_state.num_runs = 1
    if 'selected_codes' not in st.session_state:
        st.session_state.selected_codes = {}
    if 'volumes' not in st.session_state:
        st.session_state.volumes = {}
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'filtered_data' not in st.session_state:
        st.session_state.filtered_data = {}

def main():
    st.title("🧪 AHA! - Andrew Helper App")
    st.markdown("*CSV Processing Tool for Laboratory Data Analysis*")
    
    initialize_session_state()
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        step = st.radio(
            "Select Step:",
            ["1. Upload CSV", "2. Select Runs", "3. Select Codes", "4. Add Rows", "5. Process Data", "6. Download Results"]
        )
        
        # Create container that fills remaining space
        with st.container():
            st.markdown("<div style='height: 400px;'></div>", unsafe_allow_html=True)
            
            # Reset button at bottom
            if st.session_state.get('processor') is not None:
                if st.button("🔄 Reset to Original Data", key="sidebar_reset"):
                    st.session_state.processor.reset_data()
                    st.session_state.data_processed = False
                    st.session_state.filtered_data = {}
                    st.success("Data reset to original state!")
                    st.rerun()
    
    # Main content area
    if step == "1. Upload CSV":
        st.header("Step 1: Upload CSV File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload your laboratory data CSV file"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV with proper quote handling
                df = pd.read_csv(uploaded_file, quoting=1)
                st.session_state.processor = CSVProcessor(df)
                st.success("✅ CSV file uploaded successfully!")
                              
                # Show extracted codes
                st.subheader("Extracted Codes")
                if st.session_state.processor.codes:
                    st.write(f"Found {len(st.session_state.processor.codes)} codes:")
                    st.write(", ".join(st.session_state.processor.codes))
                    

                else:
                    st.warning("No valid codes found in the CSV file.")                       
                    st.write("**Looking for patterns like:** MP25[CODE] or PP25[CODE]")
                    st.write("**Available codes:** " + ", ".join(DEFAULT_ALLOWED_CODES[:10]) + "...")
                
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    elif step == "2. Select Runs":
        st.header("Step 2: Select Number of Runs")
        
        if st.session_state.processor is None:
            st.warning("Please upload a CSV file first.")
            return
        
        st.session_state.num_runs = st.radio(
            "Number of runs:",
            options=[1, 2, 3],
            index=st.session_state.num_runs - 1,
            horizontal=True
        )
        
        st.success(f"Selected {st.session_state.num_runs} run(s)")
    
    elif step == "3. Select Codes":
        st.header("Step 3: Select Codes for Each Run")
        
        if st.session_state.processor is None:
            st.warning("Please upload a CSV file first.")
            return
        
        if not st.session_state.processor.codes:
            st.warning("No codes found in the CSV file.")
            return
        
        # Create checkboxes for each code and run combination
        st.subheader("Select which codes to include in each run:")
        
        # Track selected codes across runs
        selected_in_previous_runs = set()

        for run in range(1, st.session_state.num_runs + 1):
            st.write(f"**Run {run}:**")
            cols = st.columns(4)  # 4 columns for better layout
        
            for idx, code in enumerate(st.session_state.processor.codes):
                col_idx = idx % 4
                with cols[col_idx]:
                    key = f"code_{code}_run_{run}"
                    # Disable if selected in any earlier run
                    disabled = code in selected_in_previous_runs
                    current_value = st.session_state.selected_codes.get(key, False)
        
                    # Render checkbox with disable logic
                    st.session_state.selected_codes[key] = st.checkbox(
                        f"{code}",
                        value=current_value and not disabled,
                        key=key,
                        disabled=disabled
                    )
        
                    # Track selected code for future runs
                    if st.session_state.selected_codes[key]:
                        selected_in_previous_runs.add(code)
        
            st.write("---")
   
    elif step == "4. Add Rows":
        st.header("Step 4: Add New Rows (Optional)")
        
        if st.session_state.processor is None:
            st.warning("Please upload a CSV file first.")
            return
        
        with st.expander("Add New Row", expanded=False):
            # Sample type selection
            sample_type = st.radio(
                "Sample Type:",
                ["Regular Sample", "Control Sample"],
                horizontal=True
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Poolplaat (PP25)")
                pp25_codes = [code for code in st.session_state.processor.codes if code != ""]
                if pp25_codes:
                    pp25_code = st.selectbox("PP25 Code:", pp25_codes)
                    
                    # Position input with plate selector
                    pp25_position = st.text_input("Position:", value="A1", key="pp25_pos")
                    
                    if st.checkbox("Show Plate Selector", key="show_pp25_plate"):
                        selected_pos = create_96_well_plate_selector("pp25", pp25_position)
                        if selected_pos != pp25_position:
                            st.session_state.pp25_pos = selected_pos
                            st.rerun()
            
            with col2:
                st.subheader("Analyseplaat (MP25)")
                mp25_code = st.selectbox("MP25 Code:", st.session_state.processor.codes)
                mp25_num = st.number_input("Number:", min_value=1, max_value=999, value=1)
                mp25_position = st.text_input("Position:", value="A1", key="mp25_pos")
                
                if st.checkbox("Show Plate Selector", key="show_mp25_plate"):
                    selected_pos = create_96_well_plate_selector("mp25", mp25_position)
                    if selected_pos != mp25_position:
                        st.session_state.mp25_pos = selected_pos
                        st.rerun()
            
            # Control sample specific options
            if sample_type == "Control Sample":
                if mp25_code in CONTROL_SAMPLES:
                    control_info = CONTROL_SAMPLES[mp25_code]
                    control_idx = st.selectbox(
                        "Control Sample:",
                        range(len(control_info['names'])),
                        format_func=lambda x: control_info['names'][x]
                    )
                    pp25_position = control_info['positions'][control_idx]
                    st.info(f"Auto-selected position: {pp25_position}")
            
            # Add row button
            if st.button("Add Row"):
                if 'pp25_code' in locals() and 'mp25_code' in locals():
                    sample_num = position_to_sample_number(pp25_position)
                    
                    # Create new row data
                    col0 = f'"{pp25_code}":{pp25_position}'
                    col7 = col0
                    col9 = f'"MP25{mp25_code}{mp25_num}":{mp25_position}'
                    
                    new_row = [
                        col0, '100', f'Sample {sample_num}', 'Sample', '1 M', '', '', 
                        col7, '20', col9
                    ]
                    
                    # Pad row to match original dataframe columns
                    while len(new_row) < len(st.session_state.processor.df.columns):
                        new_row.append('')
                    
                    st.session_state.processor.add_row(new_row)
                    st.success("Row added successfully!")
                    st.rerun()
    
    elif step == "5. Process Data":
        st.header("Step 5: Process Data and Set Volumes")
        
        if st.session_state.processor is None:
            st.warning("Please upload a CSV file first.")
            return
        
        # Process data button
        if st.button("Process Selected Data"):
            # Get selected codes for each run
            selected_by_run = {}
            for run in range(1, st.session_state.num_runs + 1):
                selected_codes = []
                for code in st.session_state.processor.codes:
                    key = f"code_{code}_run_{run}"
                    if st.session_state.selected_codes.get(key, False):
                        selected_codes.append(code)
                selected_by_run[run] = selected_codes
            
            # Filter data for each run
            st.session_state.filtered_data = {}
            for run, codes in selected_by_run.items():
                if codes:
                    filtered_df = st.session_state.processor.filter_data(codes, run)
                    st.session_state.filtered_data[run] = filtered_df
            
            st.session_state.data_processed = True
            st.success("Data processed successfully!")
        
        # Show volume editors if data is processed
        if st.session_state.data_processed and st.session_state.filtered_data:
            st.subheader("Volume Settings")
            
            # Get all unique codes from filtered data
            all_codes = set()
            for run_data in st.session_state.filtered_data.values():
                for code in st.session_state.processor.codes:
                    pattern = r'(?:MP25|PP25)' + re.escape(code) + r'(?:\d+)?'
                    if run_data.astype(str).apply(
                        lambda x: x.str.contains(pattern, regex=True, na=False)
                    ).any(axis=1).any():
                        all_codes.add(code)
            
            # Create volume inputs
            cols = st.columns(3)
            for idx, code in enumerate(sorted(all_codes)):
                col_idx = idx % 3
                with cols[col_idx]:
                    default_vol = CUSTOM_DEFAULTS.get(code, 20)
                    st.session_state.volumes[code] = st.number_input(
                        f"{code} Volume:",
                        min_value=1,
                        max_value=1000,
                        value=st.session_state.volumes.get(code, default_vol),
                        key=f"vol_{code}"
                    )
            
            # Show preview of processed data
            st.subheader("Processed Data Preview")
            for run, df in st.session_state.filtered_data.items():
                with st.expander(f"Run {run} ({len(df)} rows)"):
                    # Apply volumes
                    df_with_volumes = st.session_state.processor.apply_volumes(df, st.session_state.volumes)
                    st.dataframe(df_with_volumes)
                    
                    # Show MP25 codes found
                    mp25_codes = set()
                    for col in df_with_volumes.columns:
                        for value in df_with_volumes[col].astype(str):
                            matches = re.findall(r'MP25([A-Z0-9]+)', value)
                            mp25_codes.update(matches)
                    
                    if mp25_codes:
                        st.write(f"**MP25 codes found:** {', '.join(sorted(mp25_codes))}")
    
    elif step == "6. Download Results":
        st.header("Step 6: Download Results")
        
        if st.session_state.processor is None:
            st.warning("Please upload a CSV file first.")
            return
        
        if not st.session_state.data_processed or not st.session_state.filtered_data:
            st.warning("Please process data first.")
            return
        
        # Generate download buttons for each run
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        for run, df in st.session_state.filtered_data.items():
            # Apply volumes
            df_with_volumes = st.session_state.processor.apply_volumes(df, st.session_state.volumes)
            
            # Generate filename
            filename = f"Werklijst - Andrew - Run {run} - {current_date}.csv"
            
            # Convert to CSV
            csv_buffer = io.StringIO()
            df_with_volumes.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            # Create download button
            st.download_button(
                label=f"📥 Download Run {run}",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                key=f"download_run_{run}"
            )
        
if __name__ == "__main__":
    main()
