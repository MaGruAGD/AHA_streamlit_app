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
