import streamlit as st
import pandas as pd
import re
import json
import requests
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AHA! - Andrew Helper App",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
CUSTOM_DEFAULTS = {
    'SRPR': 50,
    'SRPR3': 40,
    'ECORU': 50,
    'BLT3': 10,
    'BLT8': 10,
    'BLT412': 10
}

# Expected CSV columns - this is the key fix
EXPECTED_COLUMNS = [
    'LabwareName', 'Volume', 'SolutionName', 'SolutionType', 'Concentration',
    'SampleIdentifier', 'LabwareIdentifier', 'Step1Source', 'Step1Volume', 'Step1Destination'
]

# Database Functions
@st.cache_data
def load_database_from_github(repo_url, branch="main", filename="database.json"):
    """Load database from GitHub repository"""
    try:
        raw_url = f"https://raw.githubusercontent.com/{repo_url}/{branch}/{filename}"
        response = requests.get(raw_url)
        response.raise_for_status()
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Error loading database from GitHub: {str(e)}")
        return None

def initialize_database():
    """Initialize database from GitHub"""
    st.subheader("Database Configuration")
    
    # Load from GitHub
    repo_url = "MaGruAGD/AHA_streamlit_app"
    filename = "database.json"
    
    with st.spinner("Loading database from GitHub..."):
        database = load_database_from_github(repo_url, filename=filename)
    
    if database:
        st.success(f"✅ Database loaded from GitHub repository: {repo_url}")
        st.session_state.database = database
        return database
    else:
        st.error("❌ Failed to load database from GitHub. Please check your internet connection and try again.")
        return None

def process_database(database):
    """Process database into the format expected by the app"""
    if not database:
        return {}, {}
    
    # Extract allowed codes from the database
    allowed_codes = sorted([code for code, data in database.items() if data.get("allowed", False)])
    
    # Convert control samples to the format expected by the app
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

# CSV Processing Class
class CSVProcessor:
    def __init__(self, df, allowed_codes):
        self.original_df = df.copy()
        self.df = df.copy()
        self.allowed_codes = allowed_codes
        self.codes = self.extract_codes()
        
        # Ensure the dataframe has the correct columns
        self._normalize_columns()
        
    def _normalize_columns(self):
        """Ensure the dataframe has exactly the expected columns"""
        # If the uploaded CSV has different columns, try to map them or create the standard ones
        if len(self.df.columns) != len(EXPECTED_COLUMNS):
            # Create a new dataframe with the expected columns
            new_df = pd.DataFrame(columns=EXPECTED_COLUMNS)
            
            # Try to copy data from existing columns if they match
            for i, col in enumerate(EXPECTED_COLUMNS):
                if i < len(self.df.columns):
                    new_df[col] = self.df.iloc[:, i] if len(self.df) > 0 else None
            
            self.df = new_df
        else:
            # Rename columns to match expected format
            self.df.columns = EXPECTED_COLUMNS
    
    def extract_codes(self):
        """Extract MP25 and PP25 codes from the CSV data"""
        codes = set()
        
        # Sort codes by length (longest first) to handle overlapping codes correctly
        sorted_codes = sorted(self.allowed_codes, key=len, reverse=True)
        
        for code in sorted_codes:
            pattern = r'(?:MP25|PP25)' + re.escape(code) + r'(?:\d+)?'
            
            for col in self.df.columns:
                for value in self.df[col].astype(str):
                    if re.search(pattern, value):
                        codes.add(code)
        
        return sorted(list(codes))
    
    def get_pp25_ids(self, code):
        """Get all PP25 IDs for a specific code from the CSV"""
        pattern = r'PP25' + re.escape(code) + r'\d+'
        pp25_ids = set()
        
        for col in self.df.columns:
            for value in self.df[col].astype(str):
                matches = re.findall(pattern, value)
                pp25_ids.update(matches)
        
        return sorted(list(pp25_ids))
    
    def get_plsta_ids(self):
        """Get all PP25PLSTA IDs from the CSV"""
        pattern = r'PP25PLSTA\d+'
        plsta_ids = set()
        
        for col in self.df.columns:
            for value in self.df[col].astype(str):
                matches = re.findall(pattern, value)
                plsta_ids.update(matches)
        
        return sorted(list(plsta_ids))
    
    def get_mp25_ids(self, code):
        """Get all MP25 IDs for a specific code from the CSV"""
        pattern = r'MP25' + re.escape(code) + r'\d+'
        mp25_ids = set()
        
        for col in self.df.columns:
            for value in self.df[col].astype(str):
                matches = re.findall(pattern, value)
                mp25_ids.update(matches)
        
        return sorted(list(mp25_ids))
    
    def add_row(self, row_data):
        """Add a new row to the dataframe with proper column mapping"""
        # Ensure row_data has exactly the right number of elements
        if len(row_data) != len(EXPECTED_COLUMNS):
            # Pad or truncate to match expected columns
            if len(row_data) < len(EXPECTED_COLUMNS):
                row_data.extend([''] * (len(EXPECTED_COLUMNS) - len(row_data)))
            else:
                row_data = row_data[:len(EXPECTED_COLUMNS)]
        
        # Create a dictionary mapping column names to values
        row_dict = {col: row_data[i] for i, col in enumerate(EXPECTED_COLUMNS)}
        
        # Add the row to the dataframe
        new_row = pd.DataFrame([row_dict])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
    
    def filter_data(self, selected_codes, run_number):
        """Filter data based on selected codes and run number"""
        filtered_df = self.df.copy()
        
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
                # Update the 'Step1Volume' column instead of using index
                df_copy.loc[mask, 'Step1Volume'] = volume
        
        return df_copy

# Utility Functions
def position_to_sample_number(position):
    """Convert well position (A1, B2, etc.) to sample number (1-96)"""
    if len(position) < 2:
        return 1
    
    row = position[0].upper()
    try:
        col = int(position[1:])
    except ValueError:
        return 1
    
    # Validate row (A-H) and column (1-12)
    if row < 'A' or row > 'H' or col < 1 or col > 12:
        return 1
    
    row_num = ord(row) - ord('A')  # A=0, B=1, C=2, ..., H=7
    return (col - 1) * 8 + row_num + 1

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'processor': None,
        'num_runs': 1,
        'selected_codes': {},
        'volumes': {},
        'data_processed': False,
        'filtered_data': {},
        'database': None,
        'database_loaded': False,
        'current_step': "1. Upload CSV"
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# UI Components
def create_sidebar():
    """Create the sidebar navigation"""
    with st.sidebar:
        st.header("Navigation")
        
        steps = [
            "1. Upload CSV", 
            "2. Select Runs", 
            "3. Select Codes", 
            "4. Add Rows", 
            "5. Volume Manager",  
            "6. Process Data", 
            "7. Download Results"  
        ]
        
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
        
        # Reset button
        st.markdown("---")
        if st.button(
            "🔄 Reset Application",
            type="secondary",
            use_container_width=True,
            help="Clear all data and start over"
        ):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Database info
        st.markdown("---")
        st.subheader("Database Info")
        st.write("**Source:** MaGruAGD/AHA_streamlit_app")
        if st.session_state.database:
            allowed_codes, _ = process_database(st.session_state.database)
            st.write(f"**Loaded codes:** {len(allowed_codes)}")
        if st.button("🔄 Reload Database"):
            st.session_state.database_loaded = False
            st.rerun()

def add_row_interface(processor, allowed_codes, control_samples):
    """Enhanced add row interface with regular and control samples"""
    st.header("Step 4: Add Rows")
    
    if processor is None:
        st.warning("Please upload a CSV file first.")
        return
    
    # Sample type selection
    sample_type = st.radio(
        "Sample Type:",
        ["Regular Samples", "Control Samples"],
        key="sample_type_radio"
    )
    
    # Code selection
    selected_code = st.selectbox(
        "Select Code:",
        options=allowed_codes,
        key="code_selector"
    )
    
    if not selected_code:
        st.warning("Please select a code.")
        return
    
    # Create two columns for Poolplaat and Analyseplaat
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Poolplaat")
        
        # Get available PP25PLSTA IDs from the CSV
        plsta_ids = processor.get_plsta_ids()
        
        if plsta_ids:
            poolplaat_id = st.selectbox(
                "Poolplaat ID:",
                options=plsta_ids,
                key="poolplaat_id_selector",
                help="Select PP25PLSTA ID from the uploaded CSV"
            )
        else:
            poolplaat_id = st.text_input(
                "Poolplaat ID:",
                value="PP25PLSTA0001",
                key="poolplaat_id_input",
                help="Enter PP25PLSTA ID (format: PP25PLSTAXXXX)"
            )
        
        # Position input
        poolplaat_position = st.text_input(
            "Positie op poolplaat:",
            value="A1",
            key="poolplaat_position",
            help="Enter position like A1, B2, etc."
        )
        
        # Calculate sample number automatically
        sample_number = position_to_sample_number(poolplaat_position)
        st.text_input(
            "Sample number:",
            value=str(sample_number),
            disabled=True,
            key="sample_number_display",
            help="Automatically calculated from position"
        )
    
    with col2:
        st.subheader("🔬 Analyseplaat")
        
        # Get available MP25 IDs for the selected code
        mp25_ids = processor.get_mp25_ids(selected_code)
        
        if mp25_ids:
            analyseplaat_id = st.selectbox(
                "Analyseplaat ID:",
                options=mp25_ids,
                key="analyseplaat_id_selector"
            )
        else:
            analyseplaat_id = st.text_input(
                "Analyseplaat ID:",
                value=f"MP25{selected_code}",
                key="analyseplaat_id_input"
            )
        
        # Control sample selection (only for control samples)
        if sample_type == "Control Samples":
            if selected_code in control_samples:
                control_options = control_samples[selected_code]['names']
                control_positions = control_samples[selected_code]['positions']
                
                selected_control_idx = st.selectbox(
                    "Control Sample:",
                    options=range(len(control_options)),
                    format_func=lambda x: control_options[x],
                    key="control_sample_selector"
                )
                
                # Automatically set position based on control sample
                analyseplaat_position = control_positions[selected_control_idx]
                st.text_input(
                    "Positie op analyseplaat:",
                    value=analyseplaat_position,
                    disabled=True,
                    key="analyseplaat_position_control",
                    help="Position automatically set based on control sample"
                )
            else:
                st.warning(f"No control samples defined for code {selected_code}")
                analyseplaat_position = "A1"
        else:
            # Regular samples - allow manual position selection
            analyseplaat_position = st.text_input(
                "Positie op analyseplaat:",
                value="A1",
                key="analyseplaat_position_regular",
                help="Enter position like A1, B2, etc."
            )
    
    # Get volume for the selected code
    volume = CUSTOM_DEFAULTS.get(selected_code, 20)
    

    # Preview section
    if poolplaat_id and poolplaat_position and analyseplaat_id and analyseplaat_position:
        poolplaat_entry = f'"{poolplaat_id}":{poolplaat_position}'
        analyseplaat_entry = f'"{analyseplaat_id}":{analyseplaat_position}'
        
        # Preview row in CSV format
        preview_row = [
            poolplaat_entry,           # LabwareName
            100,                       # Volume
            f"Sample {sample_number}", # SolutionName
            "Sample",                  # SolutionType
            "1 M",                     # Concentration
            "",                        # SampleIdentifier
            "",                        # LabwareIdentifier
            poolplaat_entry,           # Step1Source
            volume,                    # Step1Volume
            analyseplaat_entry         # Step1Destination
        ]
        
        # Format as CSV string
        preview_csv = ",".join([f'"{str(item)}"' if item else '""' for item in preview_row])
        
        # Display preview
        st.code(preview_csv, language="csv")
    
    # Replace the existing "Add Sample" button with this:
    if st.button("➕ Add Sample", type="primary", use_container_width=True):
        # Validate inputs
        if not poolplaat_id or not poolplaat_position or not analyseplaat_id or not analyseplaat_position:
            st.error("Please fill in all required fields.")
            return
        
        # Create the row data in the correct format
        poolplaat_entry = f'"{poolplaat_id}":{poolplaat_position}'
        analyseplaat_entry = f'"{analyseplaat_id}":{analyseplaat_position}'
        
        # Map to the exact expected columns
        row_data = [
            poolplaat_entry,           # LabwareName
            100,                       # Volume
            f"Sample {sample_number}", # SolutionName
            "Sample",                  # SolutionType
            "1 M",                     # Concentration
            "",                        # SampleIdentifier
            "",                        # LabwareIdentifier
            poolplaat_entry,           # Step1Source
            volume,                    # Step1Volume
            analyseplaat_entry         # Step1Destination
        ]
        
        processor.add_row(row_data)
        
        # Determine which run this sample belongs to
        run_for_sample = None
        for run_num, codes in st.session_state.selected_codes.items():
            # Convert codes to list of values if it's a dict-like structure
            if isinstance(codes, dict):
                code_list = list(codes.values())
            else:
                code_list = codes
            
            if selected_code in code_list:
                run_for_sample = run_num
                break
        
        # Show run notification with toast for 3 seconds
        if run_for_sample:
            st.toast(f"✅ Sample added to Run {run_for_sample}", icon="✅")
        else:
            st.toast(f"✅ Sample added successfully", icon="✅")
        
        # Use JavaScript to delay the rerun
        st.html("""
        <script>
        setTimeout(function() {
            window.parent.location.reload();
        }, 3000);
        </script>
        """)

             
        
def volume_manager_interface(processor, allowed_codes):
    """Volume Manager interface to edit volumes for all MP25 codes"""
    st.header("Step 5: Volume Manager")
    
    if processor is None:
        st.warning("Please upload a CSV file first.")
        return
    
    # Get all unique MP25 codes from the current dataframe
    mp25_codes = set()
    for col in processor.df.columns:
        for value in processor.df[col].astype(str):
            for code in allowed_codes:
                pattern = r'MP25' + re.escape(code) + r'(?:\d+)?'
                if re.search(pattern, value):
                    mp25_codes.add(code)
    
    mp25_codes = sorted(list(mp25_codes))
    
    if not mp25_codes:
        st.warning("No MP25 codes found in the current data.")
        return
    
    st.subheader("Edit Volumes for MP25 Codes")
    st.write("Adjust the volumes for each MP25 code found in your data:")
    
    # Create volume inputs for each found MP25 code
    volume_changes = {}
    
    for code in mp25_codes:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**MP25{code}**")
        
        with col2:
            # Get current volume from the dataframe or use default
            current_volume = CUSTOM_DEFAULTS.get(code, 20)
            
            # Check if there's already a volume set for this code
            pattern = r'MP25' + re.escape(code) + r'(?:\d+)?'
            mask = processor.df.astype(str).apply(
                lambda x: x.str.contains(pattern, regex=True, na=False)
            ).any(axis=1)
            
            if mask.any() and 'Step1Volume' in processor.df.columns:
                # Get the first matching volume from the dataframe
                existing_volumes = processor.df.loc[mask, 'Step1Volume'].dropna()
                if not existing_volumes.empty:
                    try:
                        current_volume = int(existing_volumes.iloc[0])
                    except (ValueError, TypeError):
                        current_volume = CUSTOM_DEFAULTS.get(code, 20)
            
            new_volume = st.number_input(
                "Volume (μL)",
                min_value=1,
                max_value=1000,
                value=current_volume,
                key=f"volume_manager_{code}"
            )
            
            volume_changes[code] = new_volume
        
        with col3:
            st.write(f"Default: {CUSTOM_DEFAULTS.get(code, 20)} μL")
    
    # Apply changes button
    if st.button("🔄 Apply Volume Changes", type="primary", use_container_width=True):
        # Apply the volume changes to the dataframe
        updated_df = processor.apply_volumes(processor.df, volume_changes)
        processor.df = updated_df
        
        st.success("✅ Volume changes applied successfully!")
        
        # Show summary of changes
        st.subheader("Applied Changes:")
        for code, volume in volume_changes.items():
            st.write(f"• **MP25{code}**: {volume} μL")

# Main Application Steps
def step_upload_csv(allowed_codes):
    """Step 1: Upload CSV File"""
    st.header("Step 1: Upload CSV File")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your laboratory data CSV file"
    )
    
    if uploaded_file is not None:
        try:
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

def step_select_runs():
    """Step 2: Select Number of Runs"""
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

def step_select_codes():
    """Step 3: Select Codes and Volumes"""
    st.header("Step 3: Select Codes and Volumes")
    
    if st.session_state.processor is None:
        st.warning("Please upload a CSV file first.")
        return
    
    available_codes = st.session_state.processor.codes
    
    if not available_codes:
        st.warning("No codes found in the uploaded CSV file.")
        return
    
    # Code selection for each run
    for run_num in range(1, st.session_state.num_runs + 1):
        st.subheader(f"Run {run_num}")
        
        # Initialize selected codes for this run
        if run_num not in st.session_state.selected_codes:
            st.session_state.selected_codes[run_num] = []
        
        selected = []
        
        # Get all codes that are already selected in OTHER runs
        other_runs_codes = set()
        for other_run in range(1, st.session_state.num_runs + 1):
            if other_run != run_num:
                other_runs_codes.update(st.session_state.selected_codes.get(other_run, []))
        
        # Create columns for better layout (adjust number of columns as needed)
        num_cols = 3
        cols = st.columns(num_cols)
        
        for i, code in enumerate(available_codes):
            col_idx = i % num_cols
            with cols[col_idx]:
                # Check if this code is already selected for this run
                is_selected = code in st.session_state.selected_codes[run_num]
                
                # Check if this code is already used in another run
                is_used_elsewhere = code in other_runs_codes
                
                # Create checkbox - disabled if used in another run
                checkbox_value = st.checkbox(
                    code,
                    value=is_selected,
                    key=f"checkbox_{run_num}_{code}",
                    disabled=is_used_elsewhere,
                    help=f"Already selected in another run" if is_used_elsewhere else None
                )
                
                # Add to selected list if checked
                if checkbox_value:
                    selected.append(code)
        
        # Update session state for this specific run
        st.session_state.selected_codes[run_num] = selected

def step_process_data():
    """Step 6: Process Data"""
    st.header("Step 6: Process Data")
    
    if st.session_state.processor is None:
        st.warning("Please upload a CSV file first.")
        return
    
    if not any(st.session_state.selected_codes.values()):
        st.warning("Please select codes for processing.")
        return
    
    if st.button("🔄 Process Data", type="primary"):
        st.session_state.filtered_data = {}
        
        for run_num in range(1, st.session_state.num_runs + 1):
            selected_codes = st.session_state.selected_codes.get(run_num, [])
            
            if selected_codes:
                # Filter data
                filtered_df = st.session_state.processor.filter_data(selected_codes, run_num)
                
                # Apply volumes
                volumes = {}
                for code in selected_codes:
                    volume_key = f"{run_num}_{code}"
                    if volume_key in st.session_state.volumes:
                        volumes[code] = st.session_state.volumes[volume_key]
                
                if volumes:
                    filtered_df = st.session_state.processor.apply_volumes(filtered_df, volumes)
                
                st.session_state.filtered_data[run_num] = filtered_df
        
        st.session_state.data_processed = True
        st.success("✅ Data processed successfully!")
    
    # Display processed data
    if st.session_state.data_processed and st.session_state.filtered_data:
        for run_num, df in st.session_state.filtered_data.items():
            st.subheader(f"Run {run_num} - Processed Data")
            st.write(f"**Rows:** {len(df)}")
            st.dataframe(df, use_container_width=True)

def step_download_results():
    """Step 7: Download Results"""
    st.header("Step 7: Download Results")
    
    if not st.session_state.data_processed or not st.session_state.filtered_data:
        st.warning("Please process data first.")
        return
    
    # Download options
    download_format = st.radio(
        "Download Format:",
        ["Separate files for each run", "Combined file with all runs"],
        horizontal=True
    )
    
    if download_format == "Separate files for each run":
        for run_num, df in st.session_state.filtered_data.items():
            # Use QUOTE_MINIMAL to only quote fields that contain special characters
            csv_data = df.to_csv(index=False, quoting=0)  # 0 = QUOTE_MINIMAL
            st.download_button(
                label=f"📥 Download Run {run_num}",
                data=csv_data,
                file_name=f"processed_run_{run_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
# Main Application
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

    # Create sidebar
    create_sidebar()

    # Main content area based on current step
    step = st.session_state.current_step
    
    if step == "1. Upload CSV":
        step_upload_csv(allowed_codes)
    elif step == "2. Select Runs":
        step_select_runs()
    elif step == "3. Select Codes":
        step_select_codes()
    elif step == "4. Add Rows":
        add_row_interface(st.session_state.processor, allowed_codes, control_samples)
    elif step == "5. Volume Manager":  # Add this new step
        volume_manager_interface(st.session_state.processor, allowed_codes)
    elif step == "6. Process Data":  # Update step number
        step_process_data()
    elif step == "7. Download Results":  # Update step number
        step_download_results()

if __name__ == "__main__":
    main()
