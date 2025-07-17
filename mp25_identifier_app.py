import streamlit as st
import pandas as pd
import re
import datetime
import json
import io
from typing import Dict, List, Optional, Tuple
import numpy as np

# Configure page
st.set_page_config(
    page_title="Laboratory Sample Processing App",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Predefined analytical codes and their control configurations
ANALYTICAL_CODES = {
    'ANACHL': {'controls': [('PCS ANA', 'G6'), ('NCS ANA', 'H6'), ('PCS CHLAM', 'G12'), ('NCS CHLAM', 'H12')], 'default_volume': 20},
    'BVD': {'controls': [('REF', 'F12'), ('NCS', 'H12')], 'default_volume': 20},
    'MGMS': {'controls': [('REF', 'E12'), ('PCS MG', 'F12'), ('PCS MS', 'G12'), ('NCS', 'H12')], 'default_volume': 20},
    'SRPR': {'controls': [('NCS', 'H12')], 'default_volume': 50},
    'BLT3': {'controls': [('NCS', 'H12')], 'default_volume': 10},
    'TETRA': {'controls': [('NCS TET', 'F12'), ('NCS CHL', 'G12'), ('NCS SUL', 'H12')], 'default_volume': 20},
    'ECOLI': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'PRRS': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'PARVO': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'SALM': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'LEPTO': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'MYCBO': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'A2': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'BHBP': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'TOXO': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'BLT1': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'BLT2': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'BLT4': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'BLT5': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'CAM': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'CAMPY': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'CHLA': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'CHLAM': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'CLAM': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'FLUO': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'GENTA': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'LINCO': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'NEOMY': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'SPIRA': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'SULFA': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'TYLO': {'controls': [('NCS', 'H12')], 'default_volume': 20},
    'VANCO': {'controls': [('NCS', 'H12')], 'default_volume': 20}
}

class CSVProcessor:
    """Handle CSV file processing and code extraction"""
    
    def __init__(self):
        self.original_data = None
        self.processed_data = None
        self.mp25_codes = []
        self.pp25_codes = []
        self.analytical_codes = []
    
    def process_csv(self, uploaded_file) -> pd.DataFrame:
        """Process uploaded CSV file and extract codes"""
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            self.original_data = df.copy()
            self.processed_data = df.copy()
            
            # Extract codes from all columns
            self._extract_codes(df)
            
            return df
        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")
            return None
    
    def _extract_codes(self, df: pd.DataFrame):
        """Extract MP25, PP25, and analytical codes from dataframe"""
        # Convert dataframe to string for regex search
        df_str = df.astype(str)
        all_text = ' '.join(df_str.values.flatten())
        
        # Extract MP25 codes
        mp25_pattern = r'MP25[A-Z0-9]+'
        self.mp25_codes = list(set(re.findall(mp25_pattern, all_text)))
        
        # Extract PP25 codes
        pp25_pattern = r'PP25[A-Z0-9]+'
        self.pp25_codes = list(set(re.findall(pp25_pattern, all_text)))
        
        # Extract analytical codes
        for code in ANALYTICAL_CODES.keys():
            if code in all_text:
                self.analytical_codes.append(code)
        
        # Sort codes
        self.mp25_codes.sort()
        self.pp25_codes.sort()
        self.analytical_codes.sort()
    
    def filter_by_codes(self, selected_codes: List[str]) -> pd.DataFrame:
        """Filter dataframe by selected analytical codes"""
        if not selected_codes:
            return self.processed_data
        
        # Create filter based on selected codes
        mask = pd.Series([False] * len(self.processed_data))
        
        for code in selected_codes:
            # Search for code in all columns
            code_mask = self.processed_data.astype(str).apply(
                lambda x: x.str.contains(code, na=False)
            ).any(axis=1)
            mask = mask | code_mask
        
        return self.processed_data[mask]
    
    def add_sample_row(self, sample_data: Dict) -> None:
        """Add a new sample row to the processed data"""
        new_row = pd.DataFrame([sample_data])
        self.processed_data = pd.concat([self.processed_data, new_row], ignore_index=True)
    
    def reset_data(self):
        """Reset processed data to original"""
        if self.original_data is not None:
            self.processed_data = self.original_data.copy()

class PlateSelector:
    """Handle 96-well plate selection interface"""
    
    def __init__(self):
        self.rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.cols = list(range(1, 13))
    
    def position_to_number(self, position: str) -> int:
        """Convert well position (e.g., 'A1') to sample number"""
        if len(position) < 2:
            return 1
        
        row = position[0].upper()
        col = int(position[1:])
        
        if row in self.rows and col in self.cols:
            row_idx = self.rows.index(row)
            return row_idx * 12 + col
        return 1
    
    def number_to_position(self, number: int) -> str:
        """Convert sample number to well position"""
        if number < 1 or number > 96:
            return 'A1'
        
        number -= 1  # Convert to 0-based
        row_idx = number // 12
        col_idx = number % 12
        
        return f"{self.rows[row_idx]}{col_idx + 1}"
    
    def render_plate_selector(self, key: str, selected_position: str = "A1") -> str:
        """Render interactive plate selector"""
        st.write(f"**96-Well Plate Selector ({key})**")
        
        # Create plate grid
        cols = st.columns(13)
        
        # Header row
        cols[0].write("")
        for i, col_num in enumerate(self.cols):
            cols[i + 1].write(f"**{col_num}**")
        
        selected_pos = selected_position
        
        # Plate rows
        for row in self.rows:
            row_cols = st.columns(13)
            row_cols[0].write(f"**{row}**")
            
            for i, col_num in enumerate(self.cols):
                well_pos = f"{row}{col_num}"
                is_selected = well_pos == selected_position
                
                if row_cols[i + 1].button(
                    well_pos,
                    key=f"{key}_{well_pos}",
                    type="primary" if is_selected else "secondary"
                ):
                    selected_pos = well_pos
        
        return selected_pos

def main():
    st.title("🧪 Laboratory Sample Processing App")
    st.markdown("---")
    
    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = CSVProcessor()
    if 'plate_selector' not in st.session_state:
        st.session_state.plate_selector = PlateSelector()
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'selected_codes' not in st.session_state:
        st.session_state.selected_codes = {1: [], 2: [], 3: []}
    if 'volume_settings' not in st.session_state:
        st.session_state.volume_settings = {}
    if 'num_runs' not in st.session_state:
        st.session_state.num_runs = 1
    
    # Sidebar for main controls
    with st.sidebar:
        st.header("Main Controls")
        
        # CSV File Upload
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        
        if uploaded_file is not None:
            # Process CSV
            df = st.session_state.processor.process_csv(uploaded_file)
            if df is not None:
                st.session_state.current_data = df
                st.success(f"✅ CSV processed successfully! ({len(df)} rows)")
                
                # Display extracted codes
                st.subheader("Extracted Codes")
                st.write(f"**MP25 Codes:** {len(st.session_state.processor.mp25_codes)}")
                st.write(f"**PP25 Codes:** {len(st.session_state.processor.pp25_codes)}")
                st.write(f"**Analytical Codes:** {len(st.session_state.processor.analytical_codes)}")
        
        # Number of runs selection
        st.subheader("Number of Runs")
        num_runs = st.radio("Select number of runs:", [1, 2, 3], key="num_runs_radio")
        st.session_state.num_runs = num_runs
        
        # Reset button
        if st.button("🔄 Reset Data"):
            st.session_state.processor.reset_data()
            st.session_state.selected_codes = {1: [], 2: [], 3: []}
            st.rerun()
    
    # Main content area
    if st.session_state.current_data is not None:
        # Code selection for each run
        st.header("Code Selection for Runs")
        
        # Create separate containers for each run instead of tabs to avoid key conflicts
        for run_num in range(1, st.session_state.num_runs + 1):
            with st.container():
                st.subheader(f"Run {run_num} - Select Analytical Codes")
                
                if st.session_state.processor.analytical_codes:
                    # Create columns for checkboxes
                    cols = st.columns(4)
                    
                    # Track checkbox changes
                    for j, code in enumerate(st.session_state.processor.analytical_codes):
                        col_idx = j % 4
                        
                        is_selected = code in st.session_state.selected_codes[run_num]
                        
                        # Create unique key for each checkbox
                        checkbox_key = f"run_{run_num}_code_{code}_{j}"
                        
                        checkbox_value = cols[col_idx].checkbox(
                            f"{code}",
                            value=is_selected,
                            key=checkbox_key
                        )
                        
                        # Update selected codes based on checkbox state
                        if checkbox_value and code not in st.session_state.selected_codes[run_num]:
                            st.session_state.selected_codes[run_num].append(code)
                        elif not checkbox_value and code in st.session_state.selected_codes[run_num]:
                            st.session_state.selected_codes[run_num].remove(code)
                
                # Display selected codes
                if st.session_state.selected_codes[run_num]:
                    st.write(f"**Selected codes for Run {run_num}:**")
                    st.write(", ".join(st.session_state.selected_codes[run_num]))
                
                st.markdown("---")  # Add separator between runs
        
        # Volume Management
        st.header("Volume Management")
        
        with st.expander("Volume Settings", expanded=False):
            st.subheader("Set Transfer Volumes")
            
            # Volume settings for each analytical code
            for code in st.session_state.processor.analytical_codes:
                default_vol = ANALYTICAL_CODES.get(code, {}).get('default_volume', 20)
                current_vol = st.session_state.volume_settings.get(code, default_vol)
                
                new_vol = st.slider(
                    f"{code} Volume (μL)",
                    min_value=0,
                    max_value=1000,
                    value=current_vol,
                    step=5,
                    key=f"vol_{code}"
                )
                st.session_state.volume_settings[code] = new_vol
        
        # Sample Addition System
        st.header("Sample Addition System")
        
        with st.expander("Add New Samples", expanded=False):
            sample_type = st.radio("Sample Type", ["Regular Samples", "Control Samples"])
            
            if sample_type == "Regular Samples":
                st.subheader("Add Regular Sample")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Poolplaat selection
                    if st.session_state.processor.pp25_codes:
                        selected_poolplaat = st.selectbox(
                            "Select Poolplaat:",
                            st.session_state.processor.pp25_codes
                        )
                    else:
                        selected_poolplaat = st.text_input("Poolplaat Code:")
                    
                    # Position selection
                    position = st.text_input("Starting Position (e.g., D2):", value="A1")
                    
                    # Target analysis
                    target_analysis = st.selectbox(
                        "Target Analysis:",
                        st.session_state.processor.analytical_codes
                    )
                    
                    plate_number = st.number_input("Plate Number:", min_value=1, value=1)
                    
                    # Volume
                    volume = st.number_input(
                        "Volume (μL):",
                        min_value=0,
                        max_value=1000,
                        value=st.session_state.volume_settings.get(target_analysis, 20),
                        step=5
                    )
                
                with col2:
                    # Plate selector
                    if st.checkbox("Show Plate Selector"):
                        selected_position = st.session_state.plate_selector.render_plate_selector(
                            "regular_sample",
                            position
                        )
                        position = selected_position
                
                # Add sample button
                if st.button("Add Regular Sample"):
                    sample_number = st.session_state.plate_selector.position_to_number(position)
                    
                    sample_data = {
                        'Sample_Type': 'Regular',
                        'Poolplaat': selected_poolplaat,
                        'Position': position,
                        'Sample_Number': sample_number,
                        'Target_Analysis': target_analysis,
                        'Plate_Number': plate_number,
                        'Volume': volume,
                        'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    st.session_state.processor.add_sample_row(sample_data)
                    st.success(f"Added regular sample at position {position}")
                    st.rerun()
            
            else:  # Control Samples
                st.subheader("Add Control Sample")
                
                # Analytical code selection
                analytical_code = st.selectbox(
                    "Select Analytical Code:",
                    st.session_state.processor.analytical_codes,
                    key="control_analytical_code"
                )
                
                # Show available controls for selected code
                if analytical_code in ANALYTICAL_CODES:
                    controls = ANALYTICAL_CODES[analytical_code]['controls']
                    
                    st.write(f"**Available controls for {analytical_code}:**")
                    for control_name, control_pos in controls:
                        st.write(f"- {control_name}: {control_pos}")
                    
                    # Control selection
                    control_names = [control[0] for control in controls]
                    selected_control = st.selectbox("Select Control:", control_names)
                    
                    # Find position for selected control
                    control_position = None
                    for control_name, control_pos in controls:
                        if control_name == selected_control:
                            control_position = control_pos
                            break
                    
                    # Volume
                    volume = st.number_input(
                        "Control Volume (μL):",
                        min_value=0,
                        max_value=1000,
                        value=st.session_state.volume_settings.get(analytical_code, 20),
                        step=5,
                        key="control_volume"
                    )
                    
                    # Add control button
                    if st.button("Add Control Sample"):
                        sample_number = st.session_state.plate_selector.position_to_number(control_position)
                        
                        control_data = {
                            'Sample_Type': 'Control',
                            'Control_Name': selected_control,
                            'Position': control_position,
                            'Sample_Number': sample_number,
                            'Target_Analysis': analytical_code,
                            'Volume': volume,
                            'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        st.session_state.processor.add_sample_row(control_data)
                        st.success(f"Added control sample {selected_control} at position {control_position}")
                        st.rerun()
        
        # Data Preview
        st.header("Data Preview")
        
        with st.expander("Current Data", expanded=True):
            if st.session_state.processor.processed_data is not None:
                st.dataframe(st.session_state.processor.processed_data)
                st.write(f"Total rows: {len(st.session_state.processor.processed_data)}")
        
        # Export System
        st.header("Export Worklists")
        
        if st.button("Generate Worklists"):
            export_files = {}
            
            for run_num in range(1, st.session_state.num_runs + 1):
                if st.session_state.selected_codes[run_num]:
                    # Filter data for this run
                    filtered_data = st.session_state.processor.filter_by_codes(
                        st.session_state.selected_codes[run_num]
                    )
                    
                    # Apply volume settings
                    if len(filtered_data) > 0:
                        # Create a copy to avoid modifying original
                        export_data = filtered_data.copy()
                        
                        # Apply volumes (assuming volume column is at index 8)
                        if len(export_data.columns) > 8:
                            for code in st.session_state.selected_codes[run_num]:
                                volume = st.session_state.volume_settings.get(code, 20)
                                # Apply volume to rows containing this code
                                code_mask = export_data.astype(str).apply(
                                    lambda x: x.str.contains(code, na=False)
                                ).any(axis=1)
                                export_data.loc[code_mask, export_data.columns[8]] = volume
                        
                        # Generate filename
                        today = datetime.datetime.now().strftime("%Y-%m-%d")
                        filename = f"Werklijst - Andrew - Run {run_num} - {today}.csv"
                        
                        # Convert to CSV
                        csv_buffer = io.StringIO()
                        export_data.to_csv(csv_buffer, index=False)
                        export_files[filename] = csv_buffer.getvalue()
            
            # Display download buttons
            if export_files:
                st.success(f"Generated {len(export_files)} worklist files!")
                
                for filename, csv_content in export_files.items():
                    st.download_button(
                        label=f"📥 Download {filename}",
                        data=csv_content,
                        file_name=filename,
                        mime="text/csv"
                    )
                
                # Show summary
                st.subheader("Export Summary")
                for run_num in range(1, st.session_state.num_runs + 1):
                    if st.session_state.selected_codes[run_num]:
                        st.write(f"**Run {run_num}:** {', '.join(st.session_state.selected_codes[run_num])}")
            else:
                st.warning("No runs selected for export. Please select analytical codes for at least one run.")
    
    else:
        st.info("👆 Please upload a CSV file to begin processing.")
        
        # Show example of expected CSV format
        st.subheader("Expected CSV Format")
        st.write("Your CSV file should contain sample data with columns that may include:")
        st.write("- Sample identifiers")
        st.write("- MP25 codes (e.g., MP25ANACHL)")
        st.write("- PP25 codes (e.g., PP25PLSTA)")
        st.write("- Analytical codes (e.g., ANACHL, BVD, MGMS)")
        st.write("- Position information")
        st.write("- Volume settings")

if __name__ == "__main__":
    main()
