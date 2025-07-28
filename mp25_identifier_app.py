import streamlit as st
import pandas as pd
import re
import json
import requests
from datetime import datetime
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="AHA!",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
CUSTOM_DEFAULTS = {
    'SRPR': 50,
    'SRPR3': 50,
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

class CSVProcessor:
    def __init__(self, df, allowed_codes):
        self.original_df = df.copy()
        self.df = df.copy()
        self.allowed_codes = allowed_codes
        self._normalize_columns()
        
        # Current codes (updated as data changes)
        self.codes = self.extract_codes()

    def get_code_statistics(self):
        """Get basic statistics about codes for UI display"""
        current_codes = set(self.extract_codes())
        
        return {
            'total_codes': len(current_codes),
            'all_codes_list': sorted(list(current_codes))
        }

    def _normalize_columns(self):
        """Ensure the dataframe has exactly the expected columns"""
        if len(self.df.columns) != len(EXPECTED_COLUMNS):
            new_df = pd.DataFrame(columns=EXPECTED_COLUMNS)
            for i, col in enumerate(EXPECTED_COLUMNS):
                if i < len(self.df.columns):
                    new_df[col] = self.df.iloc[:, i] if len(self.df) > 0 else None
            self.df = new_df
        else:
            self.df.columns = EXPECTED_COLUMNS

    def _get_exact_code_matches(self, text, target_code):
        """
        Get exact MP25/PP25 code matches using strict matching.
        Codes are always in format: (MP25|PP25) + CODE + exactly 4 digits
        For example: MP25BLT0001, MP25BLT30001, MP25BLT80001, MP25BLT4120001
        
        This ensures that when searching for "BLT", we only match MP25BLT + 4 digits,
        not MP25BLT3 + 4 digits or MP25BLT8 + 4 digits.
        """
        if not isinstance(text, str):
            text = str(text)

        # Create pattern that matches (MP25|PP25) + exact target_code + exactly 4 digits
        # Use word boundaries to ensure we don't match partial codes
        pattern = r'\b(MP25|PP25)' + re.escape(target_code) + r'(\d{4})\b'
        
        matches = re.findall(pattern, text)
        return [f"{prefix}{target_code}{digits}" for prefix, digits in matches]

    def _find_code_in_text(self, text, target_code):
        """
        Check if a specific code exists in the text.
        This method ensures exact code matching by checking that the code
        is followed by exactly 4 digits and not part of a longer code name.
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Pattern: (MP25|PP25) + target_code + exactly 4 digits + word boundary
        pattern = r'\b(MP25|PP25)' + re.escape(target_code) + r'\d{4}\b'
        return bool(re.search(pattern, text))

    def extract_codes(self):
        """Extract MP25 and PP25 codes from the CSV data with exact matching"""
        found_codes = set()

        # Search through all cells in the dataframe
        for col in self.df.columns:
            for value in self.df[col].fillna(''):  # Handle NaN values
                value_str = str(value)
                
                # Check each allowed code to see if it appears in this cell
                for code in self.allowed_codes:
                    if self._find_code_in_text(value_str, code):
                        found_codes.add(code)

        return sorted(list(found_codes))

    def get_pp25_ids(self, code):
        """Get all PP25 IDs for a specific code from the CSV with exact matching"""
        pp25_ids = set()

        for col in self.df.columns:
            for value in self.df[col].fillna(''):
                exact_matches = self._get_exact_code_matches(str(value), code)
                for match in exact_matches:
                    if match.startswith('PP25'):
                        pp25_ids.add(match)

        return sorted(list(pp25_ids))

    def get_mp25_ids(self, code):
        """Get all MP25 IDs for a specific code from the CSV with exact matching"""
        mp25_ids = set()

        for col in self.df.columns:
            for value in self.df[col].fillna(''):
                exact_matches = self._get_exact_code_matches(str(value), code)
                for match in exact_matches:
                    if match.startswith('MP25'):
                        mp25_ids.add(match)

        return sorted(list(mp25_ids))

    def get_plsta_ids(self):
        """Get all PP25PLSTA IDs from the CSV"""
        plsta_ids = set()
        pattern = r'\bPP25PLSTA\d{4}\b'  # Use word boundaries and exactly 4 digits

        for col in self.df.columns:
            for value in self.df[col].fillna(''):
                matches = re.findall(pattern, str(value))
                plsta_ids.update(matches)

        return sorted(list(plsta_ids))

    def add_row(self, row_data):
        """Add a new row to the dataframe"""
        # Create new row dataframe
        new_row_df = pd.DataFrame([row_data], columns=EXPECTED_COLUMNS)
        
        # Add the row
        self.df = pd.concat([self.df, new_row_df], ignore_index=True)
        
        # Update the current codes list
        self.codes = self.extract_codes()

    def remove_rows(self, row_indices):
        """Remove rows from the dataframe"""
        # Remove the rows
        self.df = self.df.drop(row_indices).reset_index(drop=True)
        
        # Update codes list
        self.codes = self.extract_codes()

    def filter_data(self, selected_codes, run_number):
        """Filter data based on selected codes and run number with exact matching"""
        filtered_df = self.df.copy()

        if not selected_codes:
            return filtered_df

        matching_rows = set()

        for code in selected_codes:
            for idx in filtered_df.index:
                row_matches = False
                for col in filtered_df.columns:
                    cell_value = str(filtered_df.at[idx, col])
                    if self._find_code_in_text(cell_value, code):
                        row_matches = True
                        break
                if row_matches:
                    matching_rows.add(idx)

        if matching_rows:
            filtered_df = filtered_df.loc[list(matching_rows)]
        else:
            filtered_df = filtered_df.iloc[0:0]

        return filtered_df

    def apply_volumes(self, df, volumes):
        """Apply custom volumes to the dataframe with exact matching"""
        df_copy = df.copy()

        for code, volume in volumes.items():
            rows_to_update = []

            for idx in df_copy.index:
                row_matches = False
                for col in df_copy.columns:
                    cell_value = str(df_copy.at[idx, col])
                    if self._find_code_in_text(cell_value, code):
                        row_matches = True
                        break

                if row_matches:
                    rows_to_update.append(idx)

            if rows_to_update and 'Step1Volume' in df_copy.columns:
                df_copy.loc[rows_to_update, 'Step1Volume'] = volume

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

def well_plate_selector_visual(key, title="Select Position", default_position="A1"):
    """
    Visual well plate selector using Streamlit columns and buttons
    Returns the selected position (e.g., "A1", "B2", etc.)
    """
    st.write(f"**{title}**")
    
    # Initialize session state for this specific selector
    state_key = f"well_plate_state_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = default_position
    
    # Create the visual well plate using columns
    rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    cols = list(range(1, 13))
    
    # Column headers - centered
    header_cols = st.columns([1] + [1] * 12)
    with header_cols[0]:
        st.write("")  # Empty space for row labels
    for i, col_num in enumerate(cols):
        with header_cols[i + 1]:
            st.markdown(f"<div style='text-align: center; font-weight: bold;'>{col_num}</div>", unsafe_allow_html=True)
    
    # Well grid
    for row in rows:
        row_cols = st.columns([1] + [1] * 12)
        
        # Row label - centered
        with row_cols[0]:
            st.markdown(f"<div style='text-align: center; font-weight: bold; padding-top: 8px;'>{row}</div>", unsafe_allow_html=True)
        
        # Wells in this row
        for i, col_num in enumerate(cols):
            position = f"{row}{col_num}"
            with row_cols[i + 1]:
                # Check if this is the currently selected position
                is_selected = st.session_state[state_key] == position
                
                # Create button with different styling for selected position
                if st.button(
                    "●" if is_selected else "○",
                    key=f"{key}_well_{position}",
                    type="primary" if is_selected else "secondary",
                    help=f"Select {position}",
                    use_container_width=True
                ):
                    st.session_state[state_key] = position
                    st.rerun()
    
    # Display selected position
    st.info(f"Selected: **{st.session_state[state_key]}**")
    
    return st.session_state[state_key]


def well_plate_selector_compact(key, title="Select Position", default_position="A1"):
    """
    Compact well plate selector using selectboxes
    Returns the selected position (e.g., "A1", "B2", etc.)
    """
    st.write(f"**{title}**")
    
    # Create two columns for row and column selection
    col1, col2 = st.columns(2)
    
    # Extract row and column from default position
    default_row = default_position[0] if default_position else 'A'
    try:
        default_col = int(default_position[1:]) if len(default_position) > 1 else 1
    except ValueError:
        default_col = 1
    
    with col1:
        selected_row = st.selectbox(
            "Row:",
            options=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'].index(default_row) if default_row in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] else 0,
            key=f"{key}_row"
        )
    
    with col2:
        selected_col = st.selectbox(
            "Column:",
            options=list(range(1, 13)),
            index=default_col - 1 if 1 <= default_col <= 12 else 0,
            key=f"{key}_col"
        )
    
    # Combine row and column
    position = f"{selected_row}{selected_col}"
    
    # Show selected position with visual indicator
    st.info(f"Selected position: **{position}**")
    
    return position


def well_plate_selector(key: str, title: str = "Select Position", default_position: str = "A1"):
    rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    cols = list(range(1, 13))

    row_key = f"{key}_row"
    col_key = f"{key}_col"
    popup_key = f"{key}_show_popup"
    visual_state_key = f"{key}_visual_state"

    # Initialize session state for row and col if not present
    if row_key not in st.session_state:
        st.session_state[row_key] = default_position[0]

    if col_key not in st.session_state:
        try:
            st.session_state[col_key] = int(default_position[1:])
        except:
            st.session_state[col_key] = 1

    # Initialize visual selection state
    if visual_state_key not in st.session_state:
        st.session_state[visual_state_key] = f"{st.session_state[row_key]}{st.session_state[col_key]}"

    st.markdown(f"### {title}")

    # Create three columns for better layout
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        selected_row = st.selectbox(
            "Row:",
            options=rows,
            index=rows.index(st.session_state[row_key]),
            key=f"{key}_row_dropdown"
        )

    with col2:
        selected_col = st.selectbox(
            "Column:",
            options=cols,
            index=st.session_state[col_key] - 1,
            key=f"{key}_col_dropdown"
        )

    with col3:
        # Add some top padding to align with the selectboxes
        st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
        if st.button("🧬 Well Selector", key=f"{key}_visual_btn", help="Open visual well plate selector", use_container_width=True):
            st.session_state[popup_key] = True
            # Sync visual selector to current dropdown selection when opening
            st.session_state[visual_state_key] = f"{st.session_state[row_key]}{st.session_state[col_key]}"
            st.rerun()

    # Update session_state with dropdown selections
    st.session_state[row_key] = selected_row
    st.session_state[col_key] = selected_col

    # Show popup if active
    if st.session_state.get(popup_key, False):
        with st.expander("🧬 Visual Well Plate Selector", expanded=True):
            # Column headers
            header_cols = st.columns([1] + [1] * 12)
            with header_cols[0]:
                st.write("")  # Empty corner for row labels
            for i, col_num in enumerate(cols):
                with header_cols[i + 1]:
                    st.markdown(f"<div style='text-align: center; font-weight: bold;'>{col_num}</div>", unsafe_allow_html=True)

            # Well grid
            for row in rows:
                row_cols = st.columns([1] + [1] * 12)
                with row_cols[0]:
                    st.markdown(f"<div style='text-align: center; font-weight: bold; padding-top: 8px;'>{row}</div>", unsafe_allow_html=True)
                for i, col_num in enumerate(cols):
                    well_pos = f"{row}{col_num}"
                    with row_cols[i + 1]:
                        is_selected = st.session_state[visual_state_key] == well_pos
                        btn_label = "●" if is_selected else "○"
                        # Use unique key for each button
                        if st.button(
                            btn_label,
                            key=f"{key}_visual_well_{well_pos}",
                            type="primary" if is_selected else "secondary",
                            help=f"Select {well_pos}",
                            use_container_width=True
                        ):
                            # Update visual selection
                            st.session_state[visual_state_key] = well_pos
                            # Also update main dropdown selections immediately
                            st.session_state[row_key] = well_pos[0]
                            st.session_state[col_key] = int(well_pos[1:])
                            # Close popup immediately
                            st.session_state[popup_key] = False
                            st.rerun()

            st.success(f"Visual selection: **{st.session_state[visual_state_key]}**")

            # Add a close button at the bottom
            if st.button("❌ Close", key=f"{key}_close_visual", type="secondary", use_container_width=True):
                st.session_state[popup_key] = False
                st.rerun()

    # Return combined selected position string
    return f"{st.session_state[row_key]}{st.session_state[col_key]}"

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'processor': None,
        'num_runs': 1,
        'selected_codes': {},
        'volumes': {},
        'data_processed': False,
        'volume_changes_applied': False,
        'filtered_data': {},
        'database': None,
        'database_loaded': False,
        'current_step': "1. Upload CSV",
        'added_samples_metadata': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# UI Components
def create_sidebar():
    """Create the sidebar navigation"""
    with st.sidebar:
        st.header("Navigatie")
        
        steps = [
            "1. Upload CSV", 
            "2. Aantal runs", 
            "3. Monster toevoegen", 
            "4. Analyses", 
            "5. Volumebeheer",  
            "6. Data verwerken", 
            "7. Download CSV"  
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
            "🔄 Reset",
            type="secondary",
            use_container_width=True,
            help="Clear all data and start over"
        ):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
def add_row_interface(processor, allowed_codes, control_samples):
    """Enhanced add row interface with regular and control samples + sample management - DUTCH VERSION"""
    st.header("Stap 3: Monsters en/of Controle toevoegen")
    
    if processor is None:
        st.warning("Upload eerst een CSV-bestand.")
        return
    
    # Track original row count if not already tracked
    if 'original_row_count' not in st.session_state:
        st.session_state.original_row_count = len(processor.original_df)
    
    # Sample management section - moved inside a container that refreshes
    def display_sample_status():
        current_row_count = len(processor.df)
        added_rows_count = current_row_count - st.session_state.original_row_count

        if added_rows_count > 0:
            st.info(f"📝 {added_rows_count} monster(s) zijn toegevoegd")
        else:
            st.info("📝 Er zijn nog geen monsters toegevoegd")
        
        return added_rows_count

    # Display current status
    added_rows_count = display_sample_status()

    # Manage Added Samples button - now always visible
    if st.button("🗂️ Toegevoegde monsters beheren", type="secondary", use_container_width=True):
        st.session_state.show_sample_manager = not st.session_state.get('show_sample_manager', False)
        st.rerun()

    # Sample manager interface
    if st.session_state.get('show_sample_manager', False):
        with st.expander("🗂️ Toegevoegde monsters:", expanded=True):
            
            # Get added rows (rows beyond the original count)
            original_count = st.session_state.get('original_row_count', 0)
            added_rows = processor.df.iloc[original_count:].copy()
            
            if len(added_rows) > 0:
                # Create a list to track which samples to delete
                samples_to_delete = []
                metadata_indices_to_delete = []
                
                # Display each added sample with delete option
                for idx, (df_idx, row) in enumerate(added_rows.iterrows()):
                    # Get the metadata for this sample
                    metadata = None
                    if 'added_samples_metadata' in st.session_state and idx < len(st.session_state.added_samples_metadata):
                        metadata = st.session_state.added_samples_metadata[idx]
                    
                    with st.container():
                        # Create columns for sample info and delete button
                        col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 2, 1])
                        
                        # Extract sample information from the row
                        solution_name = row.get('SolutionName', 'Onbekend')
                        step1_source = row.get('Step1Source', '')
                        step1_destination = row.get('Step1Destination', '')
                        step1_volume = row.get('Step1Volume', '')
                                                
                        # Parse source and destination for display
                        source_match = re.search(r'"([^"]+)"', str(step1_source))
                        dest_match = re.search(r'"([^"]+)"', str(step1_destination))
                        
                        source_id = source_match.group(1) if source_match else str(step1_source)
                        dest_id = dest_match.group(1) if dest_match else str(step1_destination)
                        
                        # Try multiple patterns for extracting positions
                        # Pattern 1: :A1, :B2, etc. at the end
                        source_pos_match = re.search(r':([A-H]\d{1,2})$', source_id)
                        dest_pos_match = re.search(r':([A-H]\d{1,2})$', dest_id)
                        
                        # Pattern 2: Just A1, B2, etc. at the end (if Pattern 1 fails)
                        if not source_pos_match:
                            source_pos_match = re.search(r'([A-H]\d{1,2})$', source_id)
                        if not dest_pos_match:
                            dest_pos_match = re.search(r'([A-H]\d{1,2})$', dest_id)
                        
                        # Pattern 3: Any A1, B2 pattern anywhere in the string (if Pattern 2 fails)
                        if not source_pos_match:
                            source_pos_match = re.search(r'([A-H]\d{1,2})', source_id)
                        if not dest_pos_match:
                            dest_pos_match = re.search(r'([A-H]\d{1,2})', dest_id)
                        
                        source_position = metadata.get('poolplaat_position', 'Onbekend') if metadata else 'Onbekend'
                        dest_position = metadata.get('analyseplaat_position', 'Onbekend') if metadata else 'Onbekend'
                        
                        # Get MP25 code (keep existing logic)
                        mp25_match = re.search(r'MP25([A-Z0-9]+)\d{4}', dest_id)
                        mp25_code = mp25_match.group(1) if mp25_match else "Onbekend"
                        
                        # Use metadata to determine sample type and details
                        if metadata:
                            is_control = metadata['sample_type'] == "Control Samples"
                            control_type = metadata.get('control_name', 'Controle')
                            sample_type_from_metadata = metadata['sample_type']
                        else:
                            # Fallback if no metadata (shouldn't happen with new system)
                            is_control = False
                            control_type = None
                            sample_type_from_metadata = "Regular Samples"
                        
                        # Display sample information
                        with col1:
                            if is_control:
                                st.write(f"🧪 **Controlemonster**")
                                st.caption(f"Type: {control_type}")
                            else:
                                st.write(f"🔬 **Routine Monster**")
                                st.caption(f"Naam: {solution_name}")
                        
                        with col2:
                            st.write(f"**MP25 Code:**")
                            st.write(f"`{mp25_code}`")
                        
                        with col3:
                            st.write(f"**Volume:**")
                            st.write(f"{step1_volume} μL")
                        
                        with col4:
                            st.write(f"**Transfer:**")
                            st.write(f"{source_position} → {dest_position}")
                            st.caption(f"Poolplaat naar Analyseplaat")
                        
                        with col5:
                            # Delete checkbox
                            delete_key = f"delete_sample_{idx}_{df_idx}"
                            st.write("**Verwijderen**")
                            if st.checkbox("🗑️", key=delete_key, help="Markeer voor verwijdering"):
                                samples_to_delete.append(df_idx)
                                metadata_indices_to_delete.append(idx)
                    
                    st.markdown("---")
                
                # Delete selected samples
                if samples_to_delete:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("🗑️ Geselecteerde Monsters Verwijderen", type="secondary", use_container_width=True):
                            # Remove selected rows from the dataframe
                            processor.df = processor.df.drop(samples_to_delete).reset_index(drop=True)
                            
                            # Remove corresponding metadata entries (in reverse order to maintain indices)
                            for idx in sorted(metadata_indices_to_delete, reverse=True):
                                if 'added_samples_metadata' in st.session_state and idx < len(st.session_state.added_samples_metadata):
                                    st.session_state.added_samples_metadata.pop(idx)
                            
                            st.success(f"✅ {len(samples_to_delete)} monster(s) verwijderd")
                            
                            # Clear the sample manager display and force refresh
                            st.session_state.show_sample_manager = False
                            st.rerun()
                    
                    with col2:
                        st.write(f"**{len(samples_to_delete)} monster(s) geselecteerd voor verwijdering**")
                
                # Close manager button
                if st.button("❌ Sluiten", use_container_width=True):
                    st.session_state.show_sample_manager = False
                    st.rerun()
            else:
                st.info("Geen toegevoegde monsters gevonden. Voeg monsters toe via de interface hieronder en ze verschijnen hier voor beheer.")

    st.markdown("---")
                           
    # Sample type selection
    sample_type = st.radio(
        "Monstertype:",
        ["Routine", "Controle"],
        key="sample_type_radio",
        index=st.session_state.get('sample_type_index', 0)
    )
    
    # Add this right after the radio button to store the state properly
    if sample_type == "Controle":
        st.session_state.sample_type = "Control Samples"  # Keep internal English for compatibility
        st.session_state.sample_type_index = 1
    else:
        st.session_state.sample_type = "Regular Samples"  # Keep internal English for compatibility
        st.session_state.sample_type_index = 0
   
    # Code selection - filter to only show codes with MP25 entries in CSV for control samples
    if sample_type == "Controle":
        # For control samples, only show codes that have MP25 entries in the CSV
        codes_with_mp25 = []
        for code in allowed_codes:
            mp25_ids = processor.get_mp25_ids(code)
            if mp25_ids:  # Only include codes that have MP25 entries
                codes_with_mp25.append(code)
        
        if codes_with_mp25:
            selected_code = st.selectbox(
                "Selecteer Code:",
                options=codes_with_mp25,
                key="code_selector",
                help="Toont alleen codes met MP25 vermeldingen gevonden in het geüploade CSV"
            )
        else:
            st.warning("Geen codes met MP25 vermeldingen gevonden in het geüploade CSV.")
            selected_code = None
    else:
        # For regular samples, show all allowed codes
        selected_code = st.selectbox(
            "Selecteer Code:",
            options=allowed_codes,
            key="code_selector"
        )
    
    if not selected_code:
        st.warning("Selecteer eerst een code.")
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
                help="Selecteer PP25PLSTA ID uit het geüploade CSV"
            )
        else:
            poolplaat_id = st.text_input(
                "Poolplaat ID:",
                value="PP25PLSTA0001",
                key="poolplaat_id_input",
                help="Voer PP25PLSTA ID in (formaat: PP25PLSTAXXXX)"
            )
        
        # Use the improved well plate selector
        poolplaat_position = well_plate_selector(
            key="poolplaat_position_selector",
            title="Selecteer positie op poolplaat",
            default_position="A1"
        )
        
        # Calculate sample number automatically
        sample_number = position_to_sample_number(poolplaat_position)
        st.text_input(
            "Monsternummer:",
            value=str(sample_number),
            disabled=True,
            key="sample_number_display",
            help="Automatisch berekend op basis van positie"
        )
    
    with col2:
        st.subheader("🔬 Analyseplaat")
        
        # Function to validate Analyseplaat ID format
        def validate_analyseplaat_id(plate_id, code):
            """Validate that the plate ID follows MP25 + code + 4 digits format"""
            import re
            expected_pattern = f"^MP25{code}\\d{{4}}$"
            return re.match(expected_pattern, plate_id) is not None
        
        # Function to get suggested Analyseplaat ID from CSV data
        def get_suggested_analyseplaat_id(processor, code):
            """Get suggested Analyseplaat ID from CSV data or generate default"""
            mp25_ids = processor.get_mp25_ids(code)
            if mp25_ids:
                # Return the first found MP25 ID from the CSV
                return mp25_ids[0]
            else:
                # Fallback to default format if none found in CSV
                return f"MP25{code}XXXX"
        
        # Analyseplaat ID input with validation
        default_analyseplaat_id = get_suggested_analyseplaat_id(processor, selected_code)
        
        # Initialize session state for analyseplaat ID if not exists
        analyseplaat_key = f"analyseplaat_id_{selected_code}"
        if analyseplaat_key not in st.session_state:
            st.session_state[analyseplaat_key] = default_analyseplaat_id
        
        analyseplaat_id = st.text_input(
            "Analyseplaat ID:",
            value=st.session_state.get(analyseplaat_key, default_analyseplaat_id),
            key=f"analyseplaat_id_input_{selected_code}",
            help=f"Vereist formaat: MP25{selected_code}XXXX (bijv., MP25{selected_code}0081)",
            placeholder=f"MP25{selected_code}0001"
        )
        
        # Update session state
        st.session_state[analyseplaat_key] = analyseplaat_id
        
        # Validate the Analyseplaat ID format
        is_valid_analyseplaat_id = validate_analyseplaat_id(analyseplaat_id, selected_code)
        
        if analyseplaat_id and not is_valid_analyseplaat_id:
            st.error(f"⚠️ Ongeldig formaat! Moet zijn: MP25{selected_code}XXXX (waarbij XXXX 4 cijfers zijn)")
            st.info(f"Voorbeeld: MP25{selected_code}0081")
        elif analyseplaat_id and is_valid_analyseplaat_id:
            st.success("✅ Geldig Analyseplaat ID formaat")
        
        # Control sample selection (only for control samples)
        control_sample_name = None
        if sample_type == "Controle":
            if selected_code in control_samples:
                control_options = control_samples[selected_code]['names']
                control_positions = control_samples[selected_code]['positions']
                
                selected_control_idx = st.selectbox(
                    "Controlemonster:",
                    options=range(len(control_options)),
                    format_func=lambda x: control_options[x],
                    key="control_sample_selector"
                )
                
                # Store control sample name for later use (for UI display only)
                control_sample_name = control_options[selected_control_idx]
                
                # Automatically set position based on control sample
                analyseplaat_position = control_positions[selected_control_idx]
                st.text_input(
                    "Positie op analyseplaat:",
                    value=analyseplaat_position,
                    disabled=True,
                    key="analyseplaat_position_control",
                    help="Positie automatisch ingesteld op basis van controlemonster"
                )
            else:
                st.warning(f"Geen Controle gedefinieerd voor code {selected_code}")
                analyseplaat_position = "A1"
        else:
            # Regular samples - use improved well plate selector
            analyseplaat_position = well_plate_selector(
                key="analyseplaat_position_selector",
                title="Selecteer positie op analyseplaat",
                default_position="A1"
            )
    
    # Get volume for the selected code
    volume = CUSTOM_DEFAULTS.get(selected_code, 20)
    
    # Preview section
    if poolplaat_id and poolplaat_position and analyseplaat_id and analyseplaat_position and is_valid_analyseplaat_id:
        poolplaat_entry = f'"{poolplaat_id}":{poolplaat_position}'
        analyseplaat_entry = f'"{analyseplaat_id}":{analyseplaat_position}'
        
        # Always use the sample number format for consistent output
        solution_name = f"Sample {sample_number}"
        
        # Preview row in CSV format
        preview_row = [
            poolplaat_entry,           # LabwareName
            100,                       # Volume
            solution_name,             # SolutionName (always "Sample X" format)
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
           
    # Add Sample button - only enabled if Analyseplaat ID is valid
    add_button_disabled = not (poolplaat_id and poolplaat_position and analyseplaat_id and analyseplaat_position and is_valid_analyseplaat_id)
    
    if st.button("➕ Monster Toevoegen", type="primary", use_container_width=True, disabled=add_button_disabled):
        # Validate inputs
        if not poolplaat_id or not poolplaat_position or not analyseplaat_id or not analyseplaat_position:
            st.error("Vul alle vereiste velden in.")
            return
        
        if not is_valid_analyseplaat_id:
            st.error(f"Ongeldig Analyseplaat ID formaat. Moet zijn: MP25{selected_code}XXXX")
            return
        
        # Create the row data in the correct format
        poolplaat_entry = f'"{poolplaat_id}":{poolplaat_position}'
        analyseplaat_entry = f'"{analyseplaat_id}":{analyseplaat_position}'
        
        # Always use the sample number format for consistent output
        solution_name = f"Sample {sample_number}"
        
        # Map to the exact expected columns
        row_data = [
            poolplaat_entry,           # LabwareName
            100,                       # Volume
            solution_name,             # SolutionName (always "Sample X" format)
            "Sample",                  # SolutionType
            "1 M",                     # Concentration
            "",                        # SampleIdentifier
            "",                        # LabwareIdentifier
            poolplaat_entry,           # Step1Source
            volume,                    # Step1Volume
            analyseplaat_entry         # Step1Destination
        ]
        
        # Add the row to the processor
        processor.add_row(row_data)
        
        # NEW: Store metadata about this added sample
        sample_metadata = {
            'row_index': len(processor.df) - 1,  # Index of the newly added row
            'sample_type': st.session_state.sample_type,  # Use the internal English value from session state
            'mp25_code': selected_code,
            'control_name': control_sample_name if st.session_state.sample_type == "Control Samples" else None,
            'poolplaat_position': poolplaat_position,
            'analyseplaat_position': analyseplaat_position,
            'volume': volume,
            'sample_number': sample_number,
            'solution_name': solution_name
        }
        
        # Initialize the list if it doesn't exist
        if 'added_samples_metadata' not in st.session_state:
            st.session_state.added_samples_metadata = []
        
        st.session_state.added_samples_metadata.append(sample_metadata)
        
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
            if sample_type == "Controle":
                st.toast(f"✅ Controlemonster '{control_sample_name}' toegevoegd aan Run {run_for_sample}", icon="✅")
            else:
                st.toast(f"✅ Monster toegevoegd aan Run {run_for_sample}", icon="✅")
        else:
            if sample_type == "Controle":
                st.toast(f"✅ Controlemonster '{control_sample_name}' succesvol toegevoegd", icon="✅")
            else:
                st.toast(f"✅ Monster succesvol toegevoegd", icon="✅")
        
        # Force a rerun to refresh the display immediately
        st.rerun()
        
def volume_manager_interface(processor, allowed_codes):
    """Volume Manager interface to edit volumes for selected MP25 codes only"""
    st.header("Stap 5: Volumebeheer")
    
    if processor is None:
        st.warning("Upload eerst een CSV-bestand.")
        return
    
    # Get codes that have been selected in the "Select Codes" step
    selected_codes_from_runs = set()
    for run_num, codes in st.session_state.selected_codes.items():
        if codes:  # Only add if codes exist for this run
            selected_codes_from_runs.update(codes)
    
    # Only show selected codes
    relevant_codes = sorted(list(selected_codes_from_runs))
    
    if not relevant_codes:
        st.warning("Nog geen analyses geselecteerd. Ga eerst naar 'Analyses'.")
        return
    
    st.subheader("Bewerk de volumes voor de geselecteerde analyses")
           
    # Create volume inputs for each relevant MP25 code
    volume_changes = {}
    
    for code in relevant_codes:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**MP25{code}**")
        
        with col2:
            # Always use custom default for this code (20µL for all except CUSTOM_DEFAULTS)
            # Ignore the imported data volume and use our custom defaults instead
            current_volume = CUSTOM_DEFAULTS.get(code, 20)
            
            new_volume = st.number_input(
                "Volume (μL)",
                min_value=1,
                max_value=100,
                value=current_volume,
                key=f"volume_manager_{code}"
            )
            
            volume_changes[code] = new_volume
        
        with col3:
            default_text = f"Default: {CUSTOM_DEFAULTS.get(code, 20)} μL"
            if code in CUSTOM_DEFAULTS:
                st.write(f"🔧 {default_text}")
            else:
                st.write(default_text)
    
    # Apply changes button
    if st.button("🔄 Toepassen", type="primary", use_container_width=True):
        # Apply the volume changes to the dataframe
        updated_df = processor.apply_volumes(processor.df, volume_changes)
        processor.df = updated_df
        
        st.session_state.volume_changes_applied = True  # Add this line
        st.success("✅ Volumewijzigingen succesvol toegepast!")
        
# Main Application Steps
def step_upload_csv(allowed_codes):
    """Step 1: Upload CSV File"""
    st.header("Stap 1: Upload Werklijst - Andrew.csv")
    
    uploaded_file = st.file_uploader(
        "Kies een CSV file uit uw bestanden",
        type="csv",
        help="Upload je Werklijst - Andrew.csv verkregen uit Labo++"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, quoting=1)
            st.session_state.processor = CSVProcessor(df, allowed_codes)
            st.success("✅ Het CSV-bestand is succesvol geüpload.")
                          
            # Show extracted codes
            st.subheader("Analyses")
            if st.session_state.processor.codes:
                st.write(f"{len(st.session_state.processor.codes)} analyses geïdentificeerd:")
                st.write(", ".join(st.session_state.processor.codes))
            else:
                st.warning("No valid codes found in the CSV file.")
                st.write("**Looking for patterns like:** MP25[CODE] or PP25[CODE]")
                st.write("**Available codes:** " + ", ".join(allowed_codes[:10]) + "...")
        
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

def step_select_runs():
    """Step 2: Select Number of Runs"""
    st.header("Step 2: Selecteer het aantal runs")
    
    if st.session_state.processor is None:
        st.warning("Upload eerst een CSV-bestand.")
        return
    
    st.session_state.num_runs = st.radio(
        "Aantal runs:",
        options=[1, 2, 3],
        index=st.session_state.num_runs - 1,
        horizontal=True
    )
    
    st.success(f"{st.session_state.num_runs} run(s) geselecteerd")

def step_select_codes():
    """Step 4: Select Codes and Volumes"""
    st.header("Stap 4: Selecteer analyses")
    
    if st.session_state.processor is None:
        st.warning("Upload eerst een CSV-bestand.")
        return
    
    # Get codes from the original CSV upload
    original_codes = set(st.session_state.processor.codes)
    
    # Update the processor's codes by re-extracting from current dataframe
    st.session_state.processor.codes = st.session_state.processor.extract_codes()
    current_codes = set(st.session_state.processor.codes)
    
    # Find newly added codes
    added_codes = current_codes - original_codes
    
    # Combine all available codes
    all_available_codes = sorted(list(current_codes))
    
    if not all_available_codes:
        st.warning("No codes found in the uploaded CSV file.")
        return
    
    # Initialize selected_codes structure if it doesn't exist
    if 'selected_codes' not in st.session_state:
        st.session_state.selected_codes = {}
    
    # Initialize all runs
    for run_num in range(1, st.session_state.num_runs + 1):
        if run_num not in st.session_state.selected_codes:
            st.session_state.selected_codes[run_num] = []
    
    # ==========================================
    # STEP 1: READ ALL CURRENT WIDGET STATES FIRST
    # ==========================================
    all_selections = {}  # run_num -> list of selected codes
    
    for run_num in range(1, st.session_state.num_runs + 1):
        all_selections[run_num] = []
        for code in all_available_codes:
            checkbox_key = f"checkbox_{run_num}_{code}"
            if checkbox_key in st.session_state and st.session_state[checkbox_key]:
                all_selections[run_num].append(code)
    
    # ==========================================
    # STEP 2: RESOLVE CONFLICTS GLOBALLY
    # ==========================================
    # If a code is selected in multiple runs, keep it only in the lowest numbered run
    used_codes = set()
    for run_num in range(1, st.session_state.num_runs + 1):
        cleaned_selection = []
        for code in all_selections[run_num]:
            if code not in used_codes:
                cleaned_selection.append(code)
                used_codes.add(code)
            else:
                # Force uncheck the checkbox for this conflicting code
                checkbox_key = f"checkbox_{run_num}_{code}"
                st.session_state[checkbox_key] = False
        st.session_state.selected_codes[run_num] = cleaned_selection
    
    # ==========================================
    # STEP 3: NOW DISPLAY THE UI
    # ==========================================
    for run_num in range(1, st.session_state.num_runs + 1):
        st.subheader(f"Run {run_num}")
        
        # Get all codes that are already selected in OTHER runs
        used_in_other_runs = set()
        for other_run in range(1, st.session_state.num_runs + 1):
            if other_run != run_num:
                used_in_other_runs.update(st.session_state.selected_codes[other_run])
        
        # Create columns for better layout
        num_cols = 3
        cols = st.columns(num_cols)
        
        for i, code in enumerate(all_available_codes):
            col_idx = i % num_cols
            with cols[col_idx]:
                # Check if this code is already used in another run
                is_disabled = code in used_in_other_runs
                
                # Add indicator for newly added codes
                code_label = code
                if code in added_codes:
                    code_label = f"{code} ✨"
                
                # Create checkbox - the state is already managed above
                st.checkbox(
                    code_label,
                    key=f"checkbox_{run_num}_{code}",
                    disabled=is_disabled,
                    help="Already selected in another run" if is_disabled else None
                )
        
        # Display selected codes for this run
        if st.session_state.selected_codes[run_num]:
            st.write(f"**Geselecteerde analyses:** {', '.join(sorted(st.session_state.selected_codes[run_num]))}")
        else:
            st.write("**Geselecteerde analyses:** Geen")
        
        st.write("---")  # Visual separator between runs
        
def step_process_data():
    """Step 6: Process Data with compact, efficient display for large datasets"""
    st.header("Stap 6: Data verwerken")
    
    if st.session_state.processor is None:
        st.warning("Upload eerst een CSV-bestand.")
        return
        
    if not st.session_state.volume_changes_applied:
        st.warning("Pas de volumewijzigingen toe in stap 5 (Volumebeheer) voordat u de gegevens verwerkt.")
        return
    
    if not any(st.session_state.selected_codes.values()):
        st.warning("Please select codes for processing.")
        return
    
    if st.button("🔄 Verwerken", type="primary"):
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
        st.success("✅ Gegevens succesvol verwerkt!")
    
    # Display processed data with compact, efficient interface
    if st.session_state.data_processed and st.session_state.filtered_data:
        for run_num, df in st.session_state.filtered_data.items():
            with st.expander(f"🗂️ Run {run_num} - Verwerkte data ({len(df)} rijen)", expanded=True):
                
                if len(df) == 0:
                    st.info("Geen data gevonden voor deze run.")
                    continue
                
                # Compact statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Totaal rijen", len(df))
                with col2:
                    unique_codes = len(set([code for col in df.columns for value in df[col].astype(str) 
                                          for code in st.session_state.processor.allowed_codes 
                                          if st.session_state.processor._find_code_in_text(value, code)]))
                    st.metric("Unieke codes", unique_codes)
                with col3:
                    total_volume = df['Step1Volume'].sum() if 'Step1Volume' in df.columns else 0
                    st.metric("Totaal volume", f"{total_volume} μL")
                with col4:
                    # Count MP25 vs PP25
                    mp25_count = sum(1 for col in df.columns for value in df[col].astype(str) if 'MP25' in str(value))
                    st.metric("MP25 entries", mp25_count)
                
                st.markdown("---")
                
                # Compact data view with pagination and filtering
                st.subheader("📊 Data overzicht")
                
                # Filter options
                filter_col1, filter_col2, filter_col3 = st.columns(3)
                
                with filter_col1:
                    # Search filter
                    search_term = st.text_input(
                        "🔍 Zoeken in data:",
                        key=f"search_run_{run_num}",
                        placeholder="Zoek naar MP25, PP25, positie..."
                    )
                
                with filter_col2:
                    # Column filter
                    show_columns = st.multiselect(
                        "📋 Toon kolommen:",
                        options=df.columns.tolist(),
                        default=['Step1Source', 'Step1Volume', 'Step1Destination'],
                        key=f"columns_run_{run_num}"
                    )
                
                with filter_col3:
                    # Rows per page
                    rows_per_page = st.selectbox(
                        "📄 Rijen per pagina:",
                        options=[10, 25, 50, 100, 500],
                        index=2,  # Default to 50
                        key=f"rows_per_page_run_{run_num}"
                    )
                
                # Apply search filter
                display_df = df.copy()
                if search_term:
                    mask = display_df.astype(str).apply(
                        lambda x: x.str.contains(search_term, case=False, na=False)
                    ).any(axis=1)
                    display_df = display_df[mask]
                
                # Apply column filter
                if show_columns:
                    display_df = display_df[show_columns]
                
                # Pagination
                total_rows = len(display_df)
                total_pages = (total_rows - 1) // rows_per_page + 1 if total_rows > 0 else 1
                
                page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
                
                with page_col2:
                    current_page = st.number_input(
                        f"Pagina (1-{total_pages}):",
                        min_value=1,
                        max_value=total_pages,
                        value=1,
                        key=f"page_run_{run_num}"
                    )
                
                # Calculate start and end indices
                start_idx = (current_page - 1) * rows_per_page
                end_idx = min(start_idx + rows_per_page, total_rows)
                
                # Display paginated data
                if total_rows > 0:
                    page_df = display_df.iloc[start_idx:end_idx]
                    
                    st.info(f"Toont rijen {start_idx + 1}-{end_idx} van {total_rows} gefilterde rijen")
                    
                    # Display the dataframe with better formatting
                    st.dataframe(
                        page_df,
                        use_container_width=True,
                        height=min(400, len(page_df) * 35 + 50),  # Dynamic height
                        hide_index=True
                    )
                else:
                    st.info("Geen data gevonden met huidige filters.")
                
                st.markdown("---")
                
                # Advanced deletion options
                st.subheader("🗑️ Geavanceerde verwijderopties")
                
                delete_col1, delete_col2 = st.columns(2)
                
                with delete_col1:
                    st.write("**Verwijderen op basis van index:**")
                    
                    # Range deletion
                    range_col1, range_col2 = st.columns(2)
                    with range_col1:
                        start_range = st.number_input(
                            "Van rij:",
                            min_value=0,
                            max_value=len(df)-1,
                            value=0,
                            key=f"start_range_run_{run_num}"
                        )
                    with range_col2:
                        end_range = st.number_input(
                            "Tot rij:",
                            min_value=start_range,
                            max_value=len(df)-1,
                            value=min(start_range + 9, len(df)-1),
                            key=f"end_range_run_{run_num}"
                        )
                    
                    if st.button(
                        f"🗑️ Verwijder rijen {start_range}-{end_range}",
                        key=f"delete_range_run_{run_num}",
                        type="secondary"
                    ):
                        # Confirm deletion
                        if f"confirm_range_delete_run_{run_num}" not in st.session_state:
                            st.session_state[f"confirm_range_delete_run_{run_num}"] = True
                            st.rerun()
                    
                    # Individual row deletion
                    st.write("**Individuele rij verwijderen:**")
                    specific_row = st.number_input(
                        "Rij nummer:",
                        min_value=0,
                        max_value=len(df)-1,
                        value=0,
                        key=f"specific_row_run_{run_num}"
                    )
                    
                    if st.button(
                        f"🗑️ Verwijder rij {specific_row}",
                        key=f"delete_specific_run_{run_num}",
                        type="secondary"
                    ):
                        # Delete immediately for single row
                        df_updated = df.drop(df.index[specific_row]).reset_index(drop=True)
                        st.session_state.filtered_data[run_num] = df_updated
                        st.success(f"✅ Rij {specific_row} verwijderd")
                        st.rerun()
                
                with delete_col2:
                    st.write("**Verwijderen op basis van inhoud:**")
                    
                    # Content-based deletion
                    delete_pattern = st.text_input(
                        "Verwijder rijen die bevatten:",
                        key=f"delete_pattern_run_{run_num}",
                        placeholder="bijv. MP25BLT0001"
                    )
                    
                    if delete_pattern:
                        # Show preview of rows that would be deleted
                        pattern_mask = df.astype(str).apply(
                            lambda x: x.str.contains(delete_pattern, case=False, na=False)
                        ).any(axis=1)
                        matching_rows = df[pattern_mask]
                        
                        if len(matching_rows) > 0:
                            st.info(f"🎯 {len(matching_rows)} rijen gevonden met '{delete_pattern}'")
                            
                            if st.button(
                                f"🗑️ Verwijder {len(matching_rows)} rijen",
                                key=f"delete_pattern_confirm_run_{run_num}",
                                type="secondary"
                            ):
                                if f"confirm_pattern_delete_run_{run_num}" not in st.session_state:
                                    st.session_state[f"confirm_pattern_delete_run_{run_num}"] = True
                                    st.rerun()
                        else:
                            st.info(f"Geen rijen gevonden met '{delete_pattern}'")
                    
                    # Clear all data option
                    st.write("**Gevaarlijke opties:**")
                    if st.button(
                        "🚨 Verwijder ALLE data",
                        key=f"delete_all_run_{run_num}",
                        type="secondary"
                    ):
                        if f"confirm_delete_all_run_{run_num}" not in st.session_state:
                            st.session_state[f"confirm_delete_all_run_{run_num}"] = True
                            st.rerun()
                
                # Handle confirmations
                # Range deletion confirmation
                if st.session_state.get(f"confirm_range_delete_run_{run_num}", False):
                    st.error(f"⚠️ **Weet je zeker dat je rijen {start_range}-{end_range} wilt verwijderen?**")
                    conf_col1, conf_col2 = st.columns(2)
                    
                    with conf_col1:
                        if st.button(f"✅ Ja, verwijder", key=f"confirm_range_yes_run_{run_num}", type="primary"):
                            indices_to_delete = list(range(start_range, end_range + 1))
                            df_updated = df.drop(df.index[indices_to_delete]).reset_index(drop=True)
                            st.session_state.filtered_data[run_num] = df_updated
                            st.session_state[f"confirm_range_delete_run_{run_num}"] = False
                            deleted_count = end_range - start_range + 1
                            st.success(f"✅ {deleted_count} rijen verwijderd")
                            st.rerun()
                    
                    with conf_col2:
                        if st.button(f"❌ Annuleren", key=f"confirm_range_no_run_{run_num}", type="secondary"):
                            st.session_state[f"confirm_range_delete_run_{run_num}"] = False
                            st.rerun()
                
                # Pattern deletion confirmation
                if st.session_state.get(f"confirm_pattern_delete_run_{run_num}", False):
                    pattern_mask = df.astype(str).apply(
                        lambda x: x.str.contains(delete_pattern, case=False, na=False)
                    ).any(axis=1)
                    matching_count = pattern_mask.sum()
                    
                    st.error(f"⚠️ **Weet je zeker dat je {matching_count} rijen met '{delete_pattern}' wilt verwijderen?**")
                    conf_col1, conf_col2 = st.columns(2)
                    
                    with conf_col1:
                        if st.button(f"✅ Ja, verwijder {matching_count} rijen", key=f"confirm_pattern_yes_run_{run_num}", type="primary"):
                            df_updated = df[~pattern_mask].reset_index(drop=True)
                            st.session_state.filtered_data[run_num] = df_updated
                            st.session_state[f"confirm_pattern_delete_run_{run_num}"] = False
                            st.success(f"✅ {matching_count} rijen verwijderd")
                            st.rerun()
                    
                    with conf_col2:
                        if st.button(f"❌ Annuleren", key=f"confirm_pattern_no_run_{run_num}", type="secondary"):
                            st.session_state[f"confirm_pattern_delete_run_{run_num}"] = False
                            st.rerun()
                
                # Delete all confirmation
                if st.session_state.get(f"confirm_delete_all_run_{run_num}", False):
                    st.error(f"⚠️ **GEVAARLIJK: Weet je zeker dat je ALLE {len(df)} rijen wilt verwijderen?**")
                    st.write("Deze actie kan niet ongedaan worden gemaakt!")
                    
                    conf_col1, conf_col2 = st.columns(2)
                    
                    with conf_col1:
                        if st.button(f"🚨 JA, VERWIJDER ALLES", key=f"confirm_all_yes_run_{run_num}", type="primary"):
                            empty_df = df.iloc[0:0].copy()  # Empty dataframe with same structure
                            st.session_state.filtered_data[run_num] = empty_df
                            st.session_state[f"confirm_delete_all_run_{run_num}"] = False
                            st.success(f"✅ Alle data verwijderd uit Run {run_num}")
                            st.rerun()
                    
                    with conf_col2:
                        if st.button(f"❌ Annuleren", key=f"confirm_all_no_run_{run_num}", type="secondary"):
                            st.session_state[f"confirm_delete_all_run_{run_num}"] = False
                            st.rerun()
            
def step_download_results():
    """Step 7: Download Results"""
    st.header("Stap 7: Download CSV")
    
    if st.session_state.processor is None:
        st.warning("Upload eerst een CSV-bestand.")
        return
    
    if not st.session_state.data_processed or not st.session_state.filtered_data:
        st.warning("Verwerk eerst de data in Stap 6 (Data verwerken) voordat u kunt downloaden.")
        return
    
    # Simple download buttons for each run
    for run_num, df in st.session_state.filtered_data.items():
        # Use QUOTE_MINIMAL to only quote fields that contain special characters
        csv_data = df.to_csv(index=False, quoting=0)  # 0 = QUOTE_MINIMAL
        
        # Create filename in format: Werklijst - Andrew - DD-MM-YYYY - Run X.csv
        current_date = datetime.now().strftime('%d-%m-%Y')
        filename = f"Werklijst - Andrew - {current_date} - Run {run_num}.csv"
        
        st.download_button(
            label=f"📥 Download Run {run_num}",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            use_container_width=True
        )

    # Display MP25 codes for each run with copy functionality
    st.subheader("📋 Analyseplaat ID's")
    
    for run_num, df in st.session_state.filtered_data.items():
        with st.expander(f"Run {run_num}", expanded=True):
            # Get the selected codes for this specific run
            selected_codes_for_run = st.session_state.selected_codes.get(run_num, [])
            
            if not selected_codes_for_run:
                st.info("No codes selected for this run.")
                continue
            
            # Extract MP25 codes from the dataframe using the exact matching logic
            mp25_codes = set()
            
            # Only look for MP25 codes that match the selected codes for this run
            for selected_code in selected_codes_for_run:
                # Look through all columns for this specific code using exact matching
                for col in df.columns:
                    for value in df[col].astype(str):
                        # Use the same exact matching logic as CSVProcessor
                        exact_matches = st.session_state.processor._get_exact_code_matches(value, selected_code)
                        for match in exact_matches:
                            if match.startswith('MP25'):
                                mp25_codes.add(match)
            
            # Sort codes for consistent display
            sorted_codes = sorted(list(mp25_codes))
            
            if sorted_codes: 
                # Display codes with copy buttons
                for code in sorted_codes:
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.code(code, language=None)
                    
                    with col2:
                        # Create a unique key for each copy button
                        copy_key = f"copy_{run_num}_{code}"
                        
                        # JavaScript for copying to clipboard
                        copy_script = f"""
                        <script>
                        function copyToClipboard_{copy_key.replace('-', '_')}() {{
                            navigator.clipboard.writeText('{code}').then(function() {{
                                // You could add a toast notification here if needed
                            }});
                        }}
                        </script>
                        <button onclick="copyToClipboard_{copy_key.replace('-', '_')}()" 
                                style="padding: 4px 8px; font-size: 12px; cursor: pointer; 
                                       border: 1px solid #ccc; background: #f8f9fa; border-radius: 3px;"
                                title="Copy {code} to clipboard">
                            📋
                        </button>
                        """
                        
                        # Use Streamlit's HTML component
                        components.html(copy_script, height=30)
            else:
                st.info(f"No MP25 codes found for the selected codes in this run.")
        
        st.markdown("---")
            
import streamlit as st
import requests
import time
from typing import Optional

def load_theme_from_github(
    username: str, 
    repository: str, 
    file_path: str, 
    branch: str = "main",
    cache_duration: int = 3600  # Cache for 1 hour
) -> Optional[str]:
    """
    Load CSS theme from a GitHub repository
    
    Args:
        username: GitHub username
        repository: Repository name
        file_path: Path to CSS file (e.g., 'theme.css')
        branch: Branch name (default: 'main')
        cache_duration: Cache duration in seconds
    
    Returns:
        CSS content as string or None if failed
    """
    
    # Create cache key
    cache_key = f"github_theme_{username}_{repository}_{file_path}_{branch}"
    
    # Check if theme is cached and still valid
    if cache_key in st.session_state:
        cached_data = st.session_state[cache_key]
        if time.time() - cached_data['timestamp'] < cache_duration:
            return cached_data['css']
    
    try:
        # Construct raw GitHub URL
        url = f"https://raw.githubusercontent.com/{username}/{repository}/{branch}/{file_path}"
        
        # Show loading indicator
        with st.spinner("Loading theme..."):
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        
        css_content = response.text
        
        # Cache the CSS content
        st.session_state[cache_key] = {
            'css': css_content,
            'timestamp': time.time()
        }
        
        return css_content
        
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not load theme from GitHub: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error loading theme: {e}")
        return None

def apply_github_theme(
    username: str,
    repository: str, 
    file_path: str,
    branch: str = "main",
    fallback_theme: Optional[str] = None,
    show_status: bool = False
):
    """
    Apply CSS theme from GitHub repository to Streamlit app
    
    Args:
        username: GitHub username
        repository: Repository name
        file_path: Path to CSS file
        branch: Branch name
        fallback_theme: Fallback CSS if GitHub load fails
        show_status: Whether to show theme loading status
    """
    
    # Load theme from GitHub
    css_content = load_theme_from_github(username, repository, file_path, branch)
    
    # Use fallback if GitHub load failed
    if css_content is None:
        if fallback_theme is not None:
            css_content = fallback_theme
            if show_status:
                st.info("💡 Using fallback theme - GitHub theme could not be loaded")
        else:
            if show_status:
                st.error("❌ No theme could be loaded")
            return
    else:
        if show_status:
            st.success(f"✅ Theme loaded from GitHub: {username}/{repository}")
    
    # Apply the theme
    if css_content:
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

def main():
    # Your GitHub repository details
    GITHUB_USERNAME = "MaGruAGD"
    GITHUB_REPO = "AHA_streamlit_app"
    THEME_FILE = "theme.css"
    
    # Initialize theme selection early
    if 'selected_theme' not in st.session_state:
        st.session_state.selected_theme = "☀️ Lichtmodus"
    
    # Theme options mapping
    theme_options = {
        "☀️ Lichtmodus": "theme.css",
        "🌙 Donkermodus": "dark_theme_css.css",
        
    }
    
    # Get current theme file
    current_theme_file = theme_options.get(st.session_state.selected_theme, THEME_FILE)
    
    # Minimal fallback theme in case GitHub is unreachable
    FALLBACK_THEME = """
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .stApp { 
            font-family: 'Inter', sans-serif; 
            background: linear-gradient(135deg, #fafafa 0%, #fff 100%);
        }
        
        .main .block-container {
            padding: 2rem;
            background: rgba(255,255,255,0.95);
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            margin: 1.5rem;
            max-width: 1200px;
        }
        
        .header-container {
            background: linear-gradient(135deg, #0066cc, #00a896);
            padding: 2rem;
            margin: -2rem -2rem 2rem -2rem;
            border-radius: 12px 12px 0 0;
            color: white;
        }
        
        .header-title {
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .header-subtitle {
            font-size: 0.875rem;
            margin: 0.75rem 0 0 0;
            opacity: 0.85;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .status-card {
            background: rgba(255,255,255,0.95);
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #e5e5e5;
            margin: 1rem 0;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }
        
        .breadcrumb-container {
            background: rgba(255,255,255,0.9);
            border: 1px solid #e5e5e5;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 1rem;
            font-family: 'Courier New', monospace;
            font-size: 0.8rem;
            color: #666;
        }
    """
    
    # Apply theme from GitHub with fallback
    apply_github_theme(
        username=GITHUB_USERNAME,
        repository=GITHUB_REPO,
        file_path=current_theme_file,
        branch="main",
        fallback_theme=FALLBACK_THEME,
        show_status=False  # Set to True if you want to see loading status
    )
    
    # Add theme selector to sidebar (call this early)
    add_theme_selector()
    
   
    # Modern professional laboratory header
    st.markdown(
        """
        <div class="header-container">
            <h1 class="header-title">
                <span class="lab-icon">🧪</span>
                Andrew Helper App
            </h1>
            <p class="header-subtitle">CSV's verwerken en exporteren</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    initialize_session_state()
    
    # Modern database setup section
    if not st.session_state.database_loaded:
        st.markdown("""
            <div class="status-card status-warning">
                <h3><span class="lab-icon">🗄️</span>Database Configuration Required</h3>
            </div>
        """, unsafe_allow_html=True)
        
        database = initialize_database()
        if database:
            st.session_state.database_loaded = True
            st.markdown("""
                <div class="status-card status-success">
                    <h4><span class="lab-icon">✅</span>Database Connection Established</h4>
                </div>
            """, unsafe_allow_html=True)
            st.rerun()
        else:
            st.stop()
    
    # Process database
    allowed_codes, control_samples = process_database(st.session_state.database)
    
    if not allowed_codes:
        st.markdown("""
            <div class="status-card status-error">
                <h4><span class="lab-icon">❌</span>Database Validation Failed</h4>
            </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Enhanced sidebar
    create_sidebar()
    
    # Modern breadcrumb navigation
    step = st.session_state.current_step
    
    # Step routing
    if step == "1. Upload CSV":
        step_upload_csv(allowed_codes)
        
    elif step == "2. Aantal runs":
        step_select_runs()
        
    elif step == "3. Monster toevoegen":
        add_row_interface(st.session_state.processor, allowed_codes, control_samples)
        
    elif step == "4. Analyses":
        step_select_codes()
        
    elif step == "5. Volumebeheer":
        volume_manager_interface(st.session_state.processor, allowed_codes)
        
    elif step == "6. Data verwerken":
        step_process_data()
        
    elif step == "7. Download CSV":
        step_download_results()

# Add this function if you want to provide theme switching capability
def add_theme_selector():
    """Add theme selection in sidebar with real-time switching"""
    with st.sidebar:
        
        # Available theme files in your repo
        theme_options = {
            "☀️ Lichtmodus": "theme.css",
            "🌙 Donkermodus": "dark_theme_css.css",
        }
        
        # Initialize theme selection in session state
        if 'selected_theme' not in st.session_state:
            st.session_state.selected_theme = "☀️ Lichtmodus"
        
        # Theme selector with callback
        selected_theme = st.selectbox(
            "Thema kiezen",
            options=list(theme_options.keys()),
            index=list(theme_options.keys()).index(st.session_state.selected_theme),
            key="theme_selector"
        )
        
        # Auto-apply theme when selection changes
        if selected_theme != st.session_state.selected_theme:
            st.session_state.selected_theme = selected_theme
            
            # Clear theme cache to force reload
            cache_keys = [k for k in st.session_state.keys() if k.startswith("github_theme_")]
            for key in cache_keys:
                del st.session_state[key]
            
            # Apply new theme
            apply_github_theme(
                username="MaGruAGD",
                repository="AHA_streamlit_app",
                file_path=theme_options[selected_theme],
                show_status=False
            )
            st.rerun()


if __name__ == "__main__":
    main()
