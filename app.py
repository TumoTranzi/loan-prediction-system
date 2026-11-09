<style>
    /* Global Light Theme */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }
    
    /* Main Title */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    
    /* Input Labels (Gender, Married, etc.) */
    label[data-testid="stWidgetLabel"] p, 
    div[data-testid="stMarkdownContainer"] p {
        color: #475569 !important;
        font-weight: 600 !important;
    }

    /* Primary Buttons */
    .stButton > button {
        background-color: #2563eb !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        width: 100%;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background-color: #1d4ed8 !important;
        transform: translateY(-2px);
    }
    
    /* Result Boxes */
    .result-success { background: #ecfdf5; border-left: 5px solid #10b981; padding: 1.5rem; border-radius: 8px; margin-top: 1rem; color: #065f46; }
    .result-danger { background: #fef2f2; border-left: 5px solid #ef4444; padding: 1.5rem; border-radius: 8px; margin-top: 1rem; color: #991b1b; }
    
    /* Footer */
    .footer { text-align: center; color: #94a3b8; font-size: 0.9rem; padding: 2rem 0 1rem 0; margin-top: 3rem; border-top: 1px solid #e2e8f0; }
</style>