import streamlit as st
import pandas as pd
import json
import requests
import hashlib
import base64
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="AHA! Database Admin",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Admin credentials - TEMPORARY FOR DEVELOPMENT
# TODO: Move to environment variables for production security!
ADMIN_CREDENTIALS = {
    "admin": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",  # 'password' 
    "superuser": "ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f",  # 'secret123'
    "andrew": "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"   # 'secret'
}

def get_admin_credentials():
    """Get admin credentials - checks environment variables first, then fallback"""
    credentials = {}
    
    # Try to load from environment variables first
    admin_users = os.getenv('ADMIN_USERS', '').split(',')
    
    for user in admin_users:
        if user.strip():
            username = user.strip()
            password_hash = os.getenv(f'ADMIN_PASSWORD_{username.upper()}')
            if password_hash:
                credentials[username] = password_hash
    
    # If no environment variables set, use hardcoded ones (DEVELOPMENT ONLY)
    if not credentials:
        if not os.getenv('PRODUCTION'):  # Only show warning if not in production
            st.warning("⚠️ Using hardcoded credentials - set environment variables for production!")
        credentials = ADMIN_CREDENTIALS
    
    return credentials

# GitHub configuration
GITHUB_CONFIG = {
    "repo_owner": "MaGruAGD",
    "repo_name": "AHA_streamlit_app",
    "file_path": "database.json",
    "branch": "main"
}

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def check_login():
    """Check if user is logged in"""
    return st.session_state.get('logged_in', False)

def get_github_token():
    """Get GitHub token from session state or environment"""
    # First check session state
    if 'github_token' in st.session_state and st.session_state.github_token:
        return st.session_state.github_token
    
    # Then check environment variable
    token = os.getenv('GITHUB_TOKEN')
    if token:
        st.session_state.github_token = token
        return token
    
    return None

def github_setup():
    """Setup GitHub token if not configured"""
    st.subheader("🔑 GitHub Configuration")
    
    current_token = get_github_token()
    
    if current_token:
        st.success("✅ GitHub token is configured")
        
        # Show masked token
        masked_token = current_token[:4] + "*" * (len(current_token) - 8) + current_token[-4:]
        st.info(f"Current token: {masked_token}")
        
        # Option to change token
        if st.button("Change GitHub Token"):
            st.session_state.github_token = None
            st.rerun()
    else:
        st.warning("⚠️ GitHub token not configured. You won't be able to push changes to GitHub.")
        
        with st.form("github_token_form"):
            st.markdown("""
            **To enable GitHub integration:**
            1. Go to GitHub → Settings → Developer settings → Personal access tokens
            2. Generate a new token with 'repo' permissions
            3. Enter the token below
            """)
            
            token_input = st.text_input("GitHub Personal Access Token", type="password")
            
            if st.form_submit_button("Save Token"):
                if token_input:
                    # Test the token
                    if test_github_token(token_input):
                        st.session_state.github_token = token_input
                        st.success("✅ GitHub token saved and verified!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid token or insufficient permissions")
                else:
                    st.error("Please enter a token")

def test_github_token(token):
    """Test if GitHub token is valid and has required permissions"""
    try:
        url = f"https://api.github.com/repos/{GITHUB_CONFIG['repo_owner']}/{GITHUB_CONFIG['repo_name']}"
        headers = {"Authorization": f"token {token}"}
        
        response = requests.get(url, headers=headers)
        return response.status_code == 200
    except:
        return False

def get_github_file_sha(token):
    """Get the SHA of the current file on GitHub"""
    try:
        url = f"https://api.github.com/repos/{GITHUB_CONFIG['repo_owner']}/{GITHUB_CONFIG['repo_name']}/contents/{GITHUB_CONFIG['file_path']}"
        headers = {"Authorization": f"token {token}"}
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get("sha")
        return None
    except:
        return None

def update_github_file(database, token, commit_message=None):
    """Update the database file on GitHub"""
    try:
        # Get current file SHA
        sha = get_github_file_sha(token)
        if not sha:
            st.error("❌ Could not get current file SHA from GitHub")
            return False
        
        # Prepare content
        content = json.dumps(database, indent=2)
        encoded_content = base64.b64encode(content.encode()).decode()
        
        # Default commit message
        if not commit_message:
            commit_message = f"Update database from admin interface - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Update file
        url = f"https://api.github.com/repos/{GITHUB_CONFIG['repo_owner']}/{GITHUB_CONFIG['repo_name']}/contents/{GITHUB_CONFIG['file_path']}"
        headers = {"Authorization": f"token {token}"}
        
        data = {
            "message": commit_message,
            "content": encoded_content,
            "sha": sha,
            "branch": GITHUB_CONFIG['branch']
        }
        
        response = requests.put(url, json=data, headers=headers)
        
        if response.status_code == 200:
            return True
        else:
            st.error(f"❌ GitHub API error: {response.status_code} - {response.json().get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        st.error(f"❌ Error updating GitHub: {str(e)}")
        return False

def login_page():
    """Display login page"""
    st.markdown(
        """
        <div style="text-align: center; margin-top: 50px;">
            <h1>🔐 AHA! Database Admin</h1>
            <p style="font-style: italic; color: #666;">
                Superuser Access Required
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Login")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login", use_container_width=True)
            
            if login_button:
                admin_credentials = get_admin_credentials()
                if username in admin_credentials:
                    hashed_password = hash_password(password)
                    if hashed_password == admin_credentials[username]:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid password")
                else:
                    st.error("Invalid username")

def logout():
    """Logout function"""
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.github_token = None
    st.rerun()

@st.cache_data
def load_database_from_github(repo_url, branch="main", filename="database.json"):
    """Load database from GitHub repository"""
    try:
        raw_url = f"https://raw.githubusercontent.com/{repo_url}/{branch}/{filename}"
        response = requests.get(raw_url)
        response.raise_for_status()
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Error loading database: {str(e)}")
        return {}

def save_database_locally(database, filename="database.json"):
    """Save database to local file"""
    try:
        with open(filename, 'w') as f:
            json.dump(database, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving database: {str(e)}")
        return False

def admin_sidebar():
    """Create admin sidebar"""
    with st.sidebar:
        st.header(f"👋 Welcome, {st.session_state.username}")
        
        if st.button("🚪 Logout", use_container_width=True):
            logout()
        
        st.markdown("---")
        
        # GitHub status
        github_token = get_github_token()
        if github_token:
            st.success("🔗 GitHub: Connected")
        else:
            st.warning("⚠️ GitHub: Not connected")
        
        st.markdown("---")
        
        st.header("Admin Actions")
        
        admin_actions = [
            "📊 View Database",
            "➕ Add New Code",
            "✏️ Edit Code",
            "🗑️ Delete Code",
            "🔄 Refresh Database",
            "💾 Export Database",
            "🔑 GitHub Setup",
            "🚀 Push to GitHub"
        ]
        
        selected_action = st.selectbox("Select Action:", admin_actions)
        
        return selected_action

def push_to_github(database):
    """Push database changes to GitHub"""
    st.header("🚀 Push to GitHub")
    
    github_token = get_github_token()
    
    if not github_token:
        st.error("❌ GitHub token not configured. Please set it up first.")
        return
    
    # Show current status
    st.subheader("Repository Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Repository:** {GITHUB_CONFIG['repo_owner']}/{GITHUB_CONFIG['repo_name']}")
        st.info(f"**File:** {GITHUB_CONFIG['file_path']}")
    
    with col2:
        st.info(f"**Branch:** {GITHUB_CONFIG['branch']}")
        st.info(f"**Total Codes:** {len(database)}")
    
    # Commit message
    st.subheader("Commit Details")
    
    default_message = f"Update database from admin interface - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    commit_message = st.text_area("Commit Message", value=default_message, height=100)
    
    # Show what will be pushed
    with st.expander("Preview Changes"):
        st.code(json.dumps(database, indent=2)[:1000] + "..." if len(json.dumps(database, indent=2)) > 1000 else json.dumps(database, indent=2), language="json")
    
    # Push button
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Push to GitHub", type="primary", use_container_width=True):
            with st.spinner("Pushing to GitHub..."):
                if update_github_file(database, github_token, commit_message):
                    st.success("✅ Successfully pushed to GitHub!")
                    st.session_state.database_modified = False
                    st.balloons()
                else:
                    st.error("❌ Failed to push to GitHub. Check the error messages above.")
    
    with col2:
        if st.button("💾 Save Locally Only", use_container_width=True):
            if save_database_locally(database):
                st.success("✅ Database saved locally!")
                st.session_state.database_modified = False

def view_database(database):
    """View current database"""
    st.header("📊 Database Overview")
    
    if not database:
        st.warning("Database is empty")
        return
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Codes", len(database))
    
    with col2:
        allowed_codes = sum(1 for data in database.values() if data.get("allowed", False))
        st.metric("Allowed Codes", allowed_codes)
    
    with col3:
        codes_with_controls = sum(1 for data in database.values() if data.get("control_samples"))
        st.metric("Codes with Controls", codes_with_controls)
    
    # Database table
    st.subheader("Database Contents")
    
    # Convert to DataFrame for better display
    rows = []
    for code, data in database.items():
        control_count = len(data.get("control_samples", {}))
        rows.append({
            "Code": code,
            "Allowed": "✅" if data.get("allowed", False) else "❌",
            "Description": data.get("description", "No description"),
            "Control Samples": control_count,
            "Default Volume": data.get("default_volume", "Not set")
        })
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

def add_new_code(database):
    """Add new code to database"""
    st.header("➕ Add New Code")
    
    with st.form("add_code_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            code = st.text_input("Code (e.g., SRPR)", help="Enter the code without MP25/PP25 prefix")
            description = st.text_area("Description", help="Brief description of what this code represents")
            allowed = st.checkbox("Allowed", value=True, help="Whether this code is allowed in the main app")
            default_volume = st.number_input("Default Volume (μL)", min_value=1, max_value=1000, value=20)
        
        with col2:
            st.subheader("Control Samples")
            num_controls = st.number_input("Number of Control Samples", min_value=0, max_value=10, value=0)
            
            control_samples = {}
            for i in range(num_controls):
                st.write(f"**Control Sample {i+1}**")
                control_name = st.text_input(f"Name", key=f"control_name_{i}")
                control_position = st.text_input(f"Position", value="A1", key=f"control_pos_{i}")
                
                if control_name and control_position:
                    control_samples[f"control_{i+1}"] = {
                        "name": control_name,
                        "position": control_position
                    }
        
        submitted = st.form_submit_button("Add Code", use_container_width=True)
        
        if submitted:
            if code and code not in database:
                database[code] = {
                    "allowed": allowed,
                    "description": description,
                    "default_volume": default_volume,
                    "control_samples": control_samples,
                    "created_by": st.session_state.username,
                    "created_at": datetime.now().isoformat()
                }
                st.success(f"Code '{code}' added successfully!")
                st.session_state.database_modified = True
            elif code in database:
                st.error(f"Code '{code}' already exists!")
            else:
                st.error("Please enter a code name!")

def edit_code(database):
    """Edit existing code"""
    st.header("✏️ Edit Code")
    
    if not database:
        st.warning("No codes to edit")
        return
    
    selected_code = st.selectbox("Select Code to Edit:", list(database.keys()))
    
    if selected_code:
        code_data = database[selected_code]
        
        with st.form("edit_code_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_description = st.text_area("Description", value=code_data.get("description", ""))
                new_allowed = st.checkbox("Allowed", value=code_data.get("allowed", False))
                new_default_volume = st.number_input("Default Volume (μL)", 
                                                   min_value=1, max_value=1000, 
                                                   value=code_data.get("default_volume", 20))
            
            with col2:
                st.subheader("Control Samples")
                current_controls = code_data.get("control_samples", {})
                
                # Display current controls
                if current_controls:
                    st.write("**Current Control Samples:**")
                    for control_id, control_info in current_controls.items():
                        st.write(f"• {control_info['name']} at {control_info['position']}")
                
                # Option to modify controls
                modify_controls = st.checkbox("Modify Control Samples")
                
                new_control_samples = current_controls.copy()
                if modify_controls:
                    # Clear existing controls
                    if st.button("Clear All Controls"):
                        new_control_samples = {}
                    
                    # Add new controls
                    num_new_controls = st.number_input("Add New Control Samples", min_value=0, max_value=5, value=0)
                    
                    for i in range(num_new_controls):
                        st.write(f"**New Control Sample {i+1}**")
                        control_name = st.text_input(f"Name", key=f"edit_control_name_{i}")
                        control_position = st.text_input(f"Position", value="A1", key=f"edit_control_pos_{i}")
                        
                        if control_name and control_position:
                            new_control_samples[f"control_{len(new_control_samples)+1}"] = {
                                "name": control_name,
                                "position": control_position
                            }
            
            submitted = st.form_submit_button("Update Code", use_container_width=True)
            
            if submitted:
                database[selected_code].update({
                    "description": new_description,
                    "allowed": new_allowed,
                    "default_volume": new_default_volume,
                    "control_samples": new_control_samples,
                    "modified_by": st.session_state.username,
                    "modified_at": datetime.now().isoformat()
                })
                st.success(f"Code '{selected_code}' updated successfully!")
                st.session_state.database_modified = True

def delete_code(database):
    """Delete existing code"""
    st.header("🗑️ Delete Code")
    
    if not database:
        st.warning("No codes to delete")
        return
    
    selected_code = st.selectbox("Select Code to Delete:", list(database.keys()))
    
    if selected_code:
        code_data = database[selected_code]
        
        # Show code details
        st.subheader("Code Details")
        st.write(f"**Code:** {selected_code}")
        st.write(f"**Description:** {code_data.get('description', 'No description')}")
        st.write(f"**Allowed:** {'Yes' if code_data.get('allowed', False) else 'No'}")
        st.write(f"**Control Samples:** {len(code_data.get('control_samples', {}))}")
        
        # Confirmation
        st.warning("⚠️ This action cannot be undone!")
        
        confirm_text = st.text_input("Type 'DELETE' to confirm:")
        
        if st.button("🗑️ Delete Code", type="primary"):
            if confirm_text == "DELETE":
                del database[selected_code]
                st.success(f"Code '{selected_code}' deleted successfully!")
                st.session_state.database_modified = True
                st.rerun()
            else:
                st.error("Please type 'DELETE' to confirm")

def export_database(database):
    """Export database as JSON"""
    st.header("💾 Export Database")
    
    if not database:
        st.warning("Database is empty")
        return
    
    # Show export options
    export_format = st.selectbox("Export Format:", ["JSON", "CSV Summary"])
    
    if export_format == "JSON":
        json_data = json.dumps(database, indent=2)
        
        st.download_button(
            label="📥 Download database.json",
            data=json_data,
            file_name=f"database_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
        
        # Show preview
        with st.expander("Preview JSON"):
            st.code(json_data, language="json")
    
    elif export_format == "CSV Summary":
        # Create CSV summary
        rows = []
        for code, data in database.items():
            control_names = []
            control_positions = []
            
            for control_info in data.get("control_samples", {}).values():
                control_names.append(control_info["name"])
                control_positions.append(control_info["position"])
            
            rows.append({
                "Code": code,
                "Allowed": data.get("allowed", False),
                "Description": data.get("description", ""),
                "Default_Volume": data.get("default_volume", 20),
                "Control_Count": len(data.get("control_samples", {})),
                "Control_Names": "; ".join(control_names),
                "Control_Positions": "; ".join(control_positions)
            })
        
        df = pd.DataFrame(rows)
        csv_data = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download database_summary.csv",
            data=csv_data,
            file_name=f"database_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Show preview
        st.subheader("CSV Preview")
        st.dataframe(df, use_container_width=True)

def main():
    """Main application"""
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'database_modified' not in st.session_state:
        st.session_state.database_modified = False
    
    # Check login
    if not check_login():
        login_page()
        return
    
    # Load database
    if 'database' not in st.session_state:
        st.session_state.database = load_database_from_github("MaGruAGD/AHA_streamlit_app")
    
    database = st.session_state.database
    
    # Create sidebar and get selected action
    selected_action = admin_sidebar()
    
    # Main content based on selected action
    if selected_action == "📊 View Database":
        view_database(database)
    elif selected_action == "➕ Add New Code":
        add_new_code(database)
    elif selected_action == "✏️ Edit Code":
        edit_code(database)
    elif selected_action == "🗑️ Delete Code":
        delete_code(database)
    elif selected_action == "🔄 Refresh Database":
        st.header("🔄 Refresh Database")
        if st.button("Refresh from GitHub", use_container_width=True):
            st.cache_data.clear()
            st.session_state.database = load_database_from_github("MaGruAGD/AHA_streamlit_app")
            st.success("Database refreshed from GitHub!")
            st.rerun()
    elif selected_action == "💾 Export Database":
        export_database(database)
    elif selected_action == "🔑 GitHub Setup":
        github_setup()
    elif selected_action == "🚀 Push to GitHub":
        push_to_github(database)
    
    # Show modification warning with enhanced options
    if st.session_state.database_modified:
        st.warning("⚠️ Database has been modified!")
        
        github_token = get_github_token()
        
        if github_token:
            st.info("💡 You can now push changes directly to GitHub or save locally.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Quick Push to GitHub", use_container_width=True, type="primary"):
                    with st.spinner("Pushing to GitHub..."):
                        if update_github_file(database, github_token):
                            st.success("✅ Successfully pushed to GitHub!")
                            st.session_state.database_modified = False
                            st.rerun()
            
            with col2:
                if st.button("💾 Save Locally", use_container_width=True):
                    if save_database_locally(database):
                        st.success("Database saved locally!")
                        st.session_state.database_modified = False
        else:
            st.info("🔑 Configure GitHub token to enable direct push to repository.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Locally", use_container_width=True):
                    if save_database_locally(database):
                        st.success("Database saved locally!")
                        st.session_state.database_modified = False
            
            with col2:
                if st.button("🔑 Setup GitHub", use_container_width=True):
                    st.session_state.selected_action = "🔑 GitHub Setup"
                    st.rerun()

if __name__ == "__main__":
    main()
