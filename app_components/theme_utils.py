"""
Theme utilities for consistent dark/light mode support across the application.
"""

import streamlit as st

def apply_dark_mode_css():
    """Apply comprehensive dark mode CSS that works across all components"""
    st.markdown("""
    <style>
        /* Universal dark mode support */
        @media (prefers-color-scheme: dark) {
            /* Main app background */
            .stApp {
                background-color: #0e1117 !important;
                color: #fafafa !important;
            }
            
            /* Sidebar */
            .css-1d391kg {
                background-color: #262730 !important;
            }
            
            /* Forms and containers */
            div[data-testid="stForm"] {
                background-color: #262730 !important;
                border: 1px solid #464852 !important;
                border-radius: 10px !important;
            }
            
            /* Text inputs */
            .stTextInput > div > div > input {
                background-color: #0e1117 !important;
                color: #fafafa !important;
                border: 1px solid #464852 !important;
            }
            
            /* Text areas */
            .stTextArea > div > div > textarea {
                background-color: #0e1117 !important;
                color: #fafafa !important;
                border: 1px solid #464852 !important;
            }
            
            /* Select boxes */
            .stSelectbox > div > div {
                background-color: #0e1117 !important;
                color: #fafafa !important;
                border: 1px solid #464852 !important;
            }
            
            /* Number inputs */
            .stNumberInput > div > div > input {
                background-color: #0e1117 !important;
                color: #fafafa !important;
                border: 1px solid #464852 !important;
            }
            
            /* Date inputs */
            .stDateInput > div > div > input {
                background-color: #0e1117 !important;
                color: #fafafa !important;
                border: 1px solid #464852 !important;
            }
            
            /* File uploader */
            .stFileUploader > div {
                background-color: #262730 !important;
                border: 1px solid #464852 !important;
                border-radius: 10px !important;
            }
            
            .stFileUploader label {
                color: #fafafa !important;
            }
            
            /* Buttons */
            .stButton > button {
                background-color: #262730 !important;
                color: #fafafa !important;
                border: 1px solid #464852 !important;
                border-radius: 8px !important;
            }
            
            .stButton > button:hover {
                background-color: #464852 !important;
                border-color: #fafafa !important;
            }
            
            /* Primary buttons */
            .stButton > button[kind="primary"] {
                background-color: #1f77b4 !important;
                color: white !important;
                border: 1px solid #1f77b4 !important;
            }
            
            .stButton > button[kind="primary"]:hover {
                background-color: #0d5aa7 !important;
                border-color: #0d5aa7 !important;
            }
            
            /* Secondary buttons */
            .stButton > button[kind="secondary"] {
                background-color: transparent !important;
                color: #fafafa !important;
                border: 1px solid #464852 !important;
            }
            
            /* Radio buttons */
            .stRadio > div {
                background-color: transparent !important;
            }
            
            .stRadio label {
                color: #fafafa !important;
            }
            
            /* Checkboxes */
            .stCheckbox label {
                color: #fafafa !important;
            }
            
            /* Multiselect */
            .stMultiSelect > div > div {
                background-color: #0e1117 !important;
                border: 1px solid #464852 !important;
            }
            
            .stMultiSelect label {
                color: #fafafa !important;
            }
            
            /* Sliders */
            .stSlider > div > div > div {
                background-color: #464852 !important;
            }
            
            /* Expanders */
            .streamlit-expanderHeader {
                background-color: #262730 !important;
                border: 1px solid #464852 !important;
                color: #fafafa !important;
                border-radius: 10px 10px 0 0 !important;
            }
            
            .streamlit-expanderContent {
                background-color: #262730 !important;
                border: 1px solid #464852 !important;
                border-top: none !important;
                border-radius: 0 0 10px 10px !important;
            }
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                background-color: #262730 !important;
                border-radius: 10px 10px 0 0 !important;
            }
            
            .stTabs [data-baseweb="tab"] {
                background-color: transparent !important;
                color: #fafafa !important;
                border-radius: 8px 8px 0 0 !important;
            }
            
            .stTabs [aria-selected="true"] {
                background-color: #464852 !important;
                color: #fafafa !important;
            }
            
            /* Tables */
            .stDataFrame {
                background-color: #262730 !important;
                border: 1px solid #464852 !important;
                border-radius: 10px !important;
            }
            
            /* Metrics */
            .css-1xarl3l {
                background-color: #262730 !important;
                border: 1px solid #464852 !important;
                border-radius: 10px !important;
                padding: 1rem !important;
            }
            
            /* Alert boxes */
            .stAlert > div {
                border-radius: 10px !important;
            }
            
            /* Success messages */
            .stSuccess > div {
                background-color: #1d4d1d !important;
                border: 1px solid #28a745 !important;
                color: #90ee90 !important;
            }
            
            /* Error messages */
            .stError > div {
                background-color: #4d1d1d !important;
                border: 1px solid #dc3545 !important;
                color: #ffb3b3 !important;
            }
            
            /* Warning messages */
            .stWarning > div {
                background-color: #4d4d1d !important;
                border: 1px solid #ffc107 !important;
                color: #ffff99 !important;
            }
            
            /* Info messages */
            .stInfo > div {
                background-color: #1d3d4d !important;
                border: 1px solid #17a2b8 !important;
                color: #87ceeb !important;
            }
            
            /* Progress bars */
            .stProgress > div > div {
                background-color: #1f77b4 !important;
            }
            
            /* Code blocks */
            .stCodeBlock {
                background-color: #1e1e1e !important;
                border: 1px solid #464852 !important;
                border-radius: 8px !important;
            }
            
            /* Markdown content */
            .stMarkdown {
                color: #fafafa !important;
            }
            
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
                color: #fafafa !important;
            }
            
            /* Spinner */
            .stSpinner > div {
                border-color: #1f77b4 !important;
            }
            
            /* Map containers */
            .folium-map {
                border: 1px solid #464852 !important;
                border-radius: 10px !important;
            }
            
            /* Plotly charts */
            .js-plotly-plot {
                background-color: #262730 !important;
                border: 1px solid #464852 !important;
                border-radius: 10px !important;
            }
        }
        
        /* Light mode enhancements */
        @media (prefers-color-scheme: light) {
            .stApp {
                background-color: #ffffff !important;
            }
            
            div[data-testid="stForm"] {
                background-color: #ffffff !important;
                border: 1px solid #e0e0e0 !important;
                border-radius: 10px !important;
            }
            
            .stButton > button {
                border-radius: 8px !important;
            }
            
            .stTextInput > div > div > input {
                border-radius: 8px !important;
            }
        }
        
        /* Universal improvements */
        .element-container {
            margin-bottom: 1rem !important;
        }
        
        .stButton > button {
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
        }
        
        /* Custom component styling */
        .auth-container {
            max-width: 800px !important;
            margin: 0 auto !important;
            padding: 1rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

def get_theme_colors():
    """Get theme-appropriate colors for dynamic styling"""
    # Note: This is a simple implementation. In practice, you might want to 
    # detect the actual theme preference via JavaScript if needed
    return {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e', 
        'success': '#28a745',
        'warning': '#ffc107',
        'error': '#dc3545',
        'info': '#17a2b8'
    }
