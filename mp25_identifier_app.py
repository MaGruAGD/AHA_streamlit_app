import streamlit as st
import pandas as pd
import re
import io
import json
import requests
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="AHA! - Andrew Helper App",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_database_from_github(repo_url, branch="main", filename="database.json"):
    """Load database from GitHub repository"""
    try:
        # Construct raw GitHub URL
        raw_url = f"https://raw.githubusercontent.com/{repo_url}/{branch}/{filename}"
        response = requests.get(raw_url)
        response.raise_for_status()
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Error loading database from GitHub: {str(e)}")
        return None

@st.cache_data
def load_local_database(uploaded_file):
    """Load database from uploaded file"""
    try:
        if uploaded_file.name.endswith('.json'):
            return json.loads(uploaded_file.read().decode())
        else:
            st.error("Please upload a JSON file")
            return None
    except Exception as e:
        st.error(f"Error loading database file: {str(e)}")
        return None

def initialize_database():
    """Initialize database from various sources"""
    st.subheader("Database Configuration")
    
    # Database source selection
    source = st.radio(
        "Database Source:",
        ["GitHub Repository", "Upload File", "Use Default"],
        help="Choose how to load the database configuration"
    )
    
    database = None
    
    if source == "GitHub Repository":
        col1, col2 = st.columns(2)
        with col1:
            repo_url = st.text_input(
                "Repository (owner/repo):", 
                value="yourusername/your-repo",
                help="Format: username/repository-name"
            )
        with col2:
            filename = st.text_input(
                "Database filename:", 
                value="database.json",
                help="JSON file containing the database"
            )
        
        if st.button("Load from GitHub"):
            database = load_database_from_github(repo_url, filename=filename)
            if database:
                st.success("✅ Database loaded from GitHub!")
                st.session_state.database = database
    
    elif source == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload Database File",
            type=['json'],
            help="Upload a JSON file containing the database configuration"
        )
        
        if uploaded_file is not None:
            database = load_local_database(uploaded_file)
            if database:
                st.success("✅ Database loaded from file!")
                st.session_state.database = database
    
    elif source == "Use Default":
        # Use the original embedded database as fallback
        database = get_default_database()
        st.session_state.database = database
        st.info("Using default embedded database")
    
    return database

def get_default_database():
    """Return the default database (your original one)"""
    return {
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
        # ... (rest of your original database)
        # I'll truncate this for brevity, but you'd include all your original entries
    }

def process_database(database):
    """Process database into the format expected by the app"""
    if not database:
        return {}, {}
    
    # Extract allowed codes from the database
    allowed_codes = sorted([code for code, data in database.items() if data.get("allowed", False)])
    
    # Convert control samples to the format expected by the original code
    control_samples = {}
    for code, data in database.items():
        if data.get("allowed", False) and "control_samples" in data:
            control_samples_data = data["control_samples"]
            positions = []
            names = []
            
            for control_id, control_info in control_samples_data.items():
                positions.append(control_info["position"])
                names.append(control_info["name"])
            
            control_samples[code] = {
                'positions': positions,
                'names': names
            }
    
    return allowed_codes, control_samples

# Custom default volumes (you might want to move this to the database file too)
CUSTOM_DEFAULTS = {
    'SRPR': 50,
    'SRPR3': 40,
    'ECORU': 50,
    'BLT3': 10,
    'BLT8': 10,
    'BLT412': 10
}

class CSVProcessor:
    def __init__(self, df, allowed_codes):
        self.original_df = df.copy()
        self.df = df.copy()
        self.allowed_codes = allowed_codes
        self.codes = self.extract_codes()
        
    def extract_codes(self):
        """Extract MP25 and PP25 codes from the CSV data"""
        codes = set()
        
        # Sort codes by length (longest first) to handle overlapping codes correctly
        sorted_codes = sorted(self.allowed_codes, key=len, reverse=True)
        
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

def create_plate_selector(key_prefix, selected_position="A1"):
    """Create a visual plate selector widget"""
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
    if 'database' not in st.session_state:
        st.session_state.database = None
    if 'database_loaded' not in st.session_state:
        st.session_state.database_loaded = False

def main():
    st.title("🧪 AHA! - Andrew Helper App")
    st.markdown("*CSV Processing Tool for Laboratory Data Analysis*")
    
    initialize_session_state()
    
    # Check if database is loaded
    if not st.session_state.database_loaded:
        st.header("Database Setup")
        database = initialize_database()
        if database:
            st.session_state.database_loaded = True
            st.rerun()
        else:
            st.stop()
    
    # Process database
    allowed_codes, control_samples = process_database(st.session_state.database)
    
    if not allowed_codes:
        st.error("No valid codes found in database")
        st.stop()

    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        
        # Initialize step in session state if not exists
        if 'current_step' not in st.session_state:
            st.session_state.current_step = "1. Upload CSV"
            
        # Create navigation buttons
        steps = ["1. Upload CSV", "2. Select Runs", "3. Select Codes", "4. Add Rows", "5. Process Data", "6. Download Results"]
        
        for step_name in steps:
            is_current = st.session_state.current_step == step_name
            if st.button(
                step_name, 
                key=f"nav_{step_name}",
                type="primary" if is_current else "secondary",
                use_container_width=True
            ):
                st.session_state.current_step = step_name
                st.rerun()
        
        # Add reset button
        st.markdown("---")
        if st.button(
            "🔄 Reset Application",
            type="secondary",
            use_container_width=True,
            help="Clear all data and start over"
        ):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Database info
        st.markdown("---")
        st.subheader("Database Info")
        st.write(f"**Loaded codes:** {len(allowed_codes)}")
        if st.button("🔄 Reload Database"):
            st.session_state.database_loaded = False
            st.rerun()

    # Main content area
    step = st.session_state.current_step
    
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
                st.session_state.processor = CSVProcessor(df, allowed_codes)
                st.success("✅ CSV file uploaded successfully!")
                              
                # Show extracted codes
                st.subheader("Extracted Codes")
                if st.session_state.processor.codes:
                    st.write(f"Found {len(st.session_state.processor.codes)} codes:")
                    st.write(", ".join(st.session_state.processor.codes))
                else:
                    st.warning("No valid codes found in the CSV file.")                       
                    st.write("**Looking for patterns like:** MP25[CODE] or PP25[CODE]")
                    st.write("**Available codes:** " + ", ".join(allowed_codes[:10]) + "...")
                
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    # ... (rest of your steps remain the same, but use the loaded allowed_codes and control_samples)
    
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
    
    # ... (continue with remaining steps, using control_samples variable instead of CONTROL_SAMPLES)

if __name__ == "__main__":
    main()
