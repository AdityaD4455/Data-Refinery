import streamlit as st
import pandas as pd
import hashlib
import json
from pathlib import Path
from datetime import datetime

# Page config
st.set_page_config(
    layout="wide",
    page_title="🚀 ML Analytics Pro - Login",
    page_icon="🤖",
    initial_sidebar_state="collapsed"
)

# Optimized CSS with smooth animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Orbitron:wght@400;700;900&display=swap');

    * { 
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #0f0c29);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }

    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .auth-container {
        max-width: 480px;
        margin: 0 auto;
        padding: 40px 20px;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    .auth-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(30px) saturate(180%);
        border-radius: 32px;
        padding: 50px 45px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 20px 80px rgba(0, 0, 0, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
        width: 100%;
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.6s ease-out;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .auth-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
        pointer-events: none;
    }

    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .logo-section {
        text-align: center;
        margin-bottom: 40px;
        position: relative;
        z-index: 1;
    }

    .logo-icon {
        font-size: 72px;
        margin-bottom: 15px;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    .logo-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 36px;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
    }

    .logo-subtitle {
        color: rgba(255, 255, 255, 0.7);
        font-size: 14px;
        letter-spacing: 2px;
        font-weight: 500;
    }

    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        color: white !important;
        padding: 16px 20px !important;
        font-size: 15px !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput > div > div > input:focus {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
        transform: translateY(-2px);
    }

    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.4) !important;
    }

    .stTextInput > label {
        color: white !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        margin-bottom: 8px !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 16px 32px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        letter-spacing: 0.5px !important;
        width: 100% !important;
        margin-top: 10px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6) !important;
    }

    .stButton > button:active {
        transform: translateY(-1px) !important;
    }

    .divider {
        display: flex;
        align-items: center;
        text-align: center;
        margin: 30px 0;
        color: rgba(255, 255, 255, 0.5);
        font-size: 13px;
    }

    .divider::before,
    .divider::after {
        content: '';
        flex: 1;
        border-bottom: 1px solid rgba(255, 255, 255, 0.15);
    }

    .divider span {
        padding: 0 15px;
        font-weight: 500;
    }

    .switch-mode {
        text-align: center;
        margin-top: 25px;
        color: rgba(255, 255, 255, 0.7);
        font-size: 14px;
    }

    .switch-mode a {
        color: #667eea;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .switch-mode a:hover {
        color: #764ba2;
        text-decoration: underline;
    }

    .features {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 15px;
        margin-top: 40px;
        padding-top: 30px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }

    .feature-item {
        background: rgba(255, 255, 255, 0.03);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.3s ease;
    }

    .feature-item:hover {
        background: rgba(255, 255, 255, 0.05);
        transform: translateY(-2px);
    }

    .feature-icon {
        font-size: 24px;
        margin-bottom: 8px;
    }

    .feature-text {
        color: rgba(255, 255, 255, 0.8);
        font-size: 12px;
        font-weight: 500;
    }

    .badge {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        display: inline-block;
        margin-top: 20px;
        animation: pulse 2s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { 
            transform: scale(1);
            box-shadow: 0 5px 15px rgba(0, 210, 255, 0.4);
        }
        50% { 
            transform: scale(1.05);
            box-shadow: 0 8px 25px rgba(0, 210, 255, 0.6);
        }
    }

    /* Toast/Alert styling */
    .stAlert {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }

    /* Loading spinner */
    .stSpinner > div {
        border-color: #667eea transparent transparent transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# User database file
USERS_FILE = Path("users.json")

def load_users():
    """Load users from JSON file"""
    if USERS_FILE.exists():
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    """Verify password against hash"""
    return hash_password(password) == hashed

def create_user(username, email, password):
    """Create new user"""
    users = load_users()
    
    if username in users:
        return False, "Username already exists"
    
    # Check if email exists
    for user_data in users.values():
        if user_data.get('email') == email:
            return False, "Email already registered"
    
    users[username] = {
        'email': email,
        'password': hash_password(password),
        'created_at': datetime.now().isoformat(),
        'last_login': None
    }
    
    save_users(users)
    return True, "Account created successfully!"

def authenticate_user(username, password):
    """Authenticate user"""
    users = load_users()
    
    if username not in users:
        return False, "Invalid username or password"
    
    if verify_password(password, users[username]['password']):
        # Update last login
        users[username]['last_login'] = datetime.now().isoformat()
        save_users(users)
        return True, "Login successful!"
    
    return False, "Invalid username or password"

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'mode' not in st.session_state:
    st.session_state.mode = 'login'

# Check if already authenticated
if st.session_state.authenticated:
    st.switch_page("app.py")

# Main container
st.markdown('<div class="auth-container">', unsafe_allow_html=True)

# Logo section
st.markdown("""
<div class="logo-section">
    <div class="logo-icon">🚀</div>
    <div class="logo-title">ML ANALYTICS PRO</div>
    <div class="logo-subtitle">AI-POWERED PLATFORM</div>
    <div class="badge">🤖 AI EDITION</div>
</div>
""", unsafe_allow_html=True)

# Auth card
st.markdown('<div class="auth-card">', unsafe_allow_html=True)

# Toggle between login and signup
col1, col2 = st.columns(2)
with col1:
    if st.button("🔑 LOGIN", use_container_width=True, 
                 type="primary" if st.session_state.mode == 'login' else "secondary"):
        st.session_state.mode = 'login'
        st.rerun()

with col2:
    if st.button("✨ SIGN UP", use_container_width=True,
                 type="primary" if st.session_state.mode == 'signup' else "secondary"):
        st.session_state.mode = 'signup'
        st.rerun()

st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)

if st.session_state.mode == 'login':
    # Login Form
    st.markdown("### 👋 Welcome Back!")
    st.markdown('<div style="color: rgba(255,255,255,0.6); font-size: 14px; margin-bottom: 25px;">Enter your credentials to continue</div>', unsafe_allow_html=True)
    
    username = st.text_input("Username", placeholder="Enter your username", key="login_username")
    password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        remember = st.checkbox("Remember me", value=True)
    with col2:
        st.markdown('<div style="text-align: right; margin-top: 8px;"><a href="#" style="color: #667eea; font-size: 13px; text-decoration: none;">Forgot?</a></div>', unsafe_allow_html=True)
    
    if st.button("🚀 LOGIN", key="login_btn", use_container_width=True, type="primary"):
        if not username or not password:
            st.error("⚠️ Please fill all fields")
        else:
            with st.spinner("Authenticating..."):
                success, message = authenticate_user(username, password)
                
                if success:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success(f"✅ {message}")
                    st.balloons()
                    st.rerun()
                else:
                    st.error(f"❌ {message}")

else:
    # Signup Form
    st.markdown("### ✨ Create Account")
    st.markdown('<div style="color: rgba(255,255,255,0.6); font-size: 14px; margin-bottom: 25px;">Join ML Analytics Pro today</div>', unsafe_allow_html=True)
    
    new_username = st.text_input("Username", placeholder="Choose a username", key="signup_username")
    new_email = st.text_input("Email", placeholder="your@email.com", key="signup_email")
    new_password = st.text_input("Password", type="password", placeholder="Create a password", key="signup_password")
    confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password", key="signup_confirm")
    
    agree = st.checkbox("I agree to Terms of Service and Privacy Policy")
    
    if st.button("✨ CREATE ACCOUNT", key="signup_btn", use_container_width=True, type="primary"):
        if not all([new_username, new_email, new_password, confirm_password]):
            st.error("⚠️ Please fill all fields")
        elif new_password != confirm_password:
            st.error("❌ Passwords don't match")
        elif len(new_password) < 6:
            st.error("❌ Password must be at least 6 characters")
        elif not agree:
            st.error("⚠️ Please agree to Terms of Service")
        else:
            with st.spinner("Creating your account..."):
                success, message = create_user(new_username, new_email, new_password)
                
                if success:
                    st.success(f"✅ {message}")
                    st.balloons()
                    st.info("👉 Please login with your credentials")
                    st.session_state.mode = 'login'
                    st.rerun()
                else:
                    st.error(f"❌ {message}")

# Features section
st.markdown("""
<div class="features">
    <div class="feature-item">
        <div class="feature-icon">🤖</div>
        <div class="feature-text">Advanced AI</div>
    </div>
    <div class="feature-item">
        <div class="feature-icon">⚡</div>
        <div class="feature-text">Real-time</div>
    </div>
    <div class="feature-item">
        <div class="feature-icon">🎯</div>
        <div class="feature-text">AutoML</div>
    </div>
    <div class="feature-item">
        <div class="feature-icon">🔒</div>
        <div class="feature-text">Secure</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close auth-card
st.markdown('</div>', unsafe_allow_html=True)  # Close auth-container

# Footer
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.5); padding: 30px; font-size: 13px;">
    <p>© 2024 ML Analytics Pro • Powered by AI</p>
</div>
""", unsafe_allow_html=True)
