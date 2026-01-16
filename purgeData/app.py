# ========== DISABLE LANGSMITH - MUST BE FIRST ==========
import os
os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# ========== END LANGSMITH DISABLE ==========

import json
import logging
from datetime import datetime

import pandas as pd
import streamlit as st

from database import db_manager
from sql_chain import sql_chain_manager
from chart_generator import chart_generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="SQL Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Session state ----------
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True
    if "active_table" not in st.session_state:
        st.session_state.active_table = None
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

init_state()

# ---------- FIXED CSS - ALWAYS APPLIES BOTH THEMES ----------
def apply_theme():
    st.markdown("""
<style>
    /* ==================== DARK MODE STYLES ==================== */
    .main { background-color: #0e1117; color: #fafafa; }
    .main-header { font-size: 2rem; font-weight: 700; color: #4a9eff; text-align: center; margin: 0.5rem 0; }
    .main-subtitle { text-align: center; color: #a0a0a0; margin-bottom: 1rem; font-size: 0.9rem; }
    
    /* USER MESSAGE - DARK */
    .user-msg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 1rem;
        margin: 0.5rem 0 0.5rem 15%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .user-label { font-weight: 700; font-size: 0.75rem; text-transform: uppercase; opacity: 0.9; margin-bottom: 0.5rem; }
    
    /* ASSISTANT MESSAGE - DARK */
    .assistant-msg {
        background-color: #1a1a1a;
        color: #e0e0e0;
        padding: 1rem;
        border-radius: 1rem;
        margin: 0.5rem 15% 0.5rem 0;
        border-left: 3px solid #4a9eff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .assistant-label { font-weight: 700; font-size: 0.75rem; text-transform: uppercase; color: #4a9eff; margin-bottom: 0.5rem; }
    
    /* SQL BOX - DARK */
    .sql-box {
        background-color: #0d1117;
        color: #79c0ff;
        padding: 0.75rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        margin-top: 0.75rem;
        border: 1px solid #30363d;
        overflow-x: auto;
    }
    .sql-label { color: #8b949e; font-size: 0.7rem; font-weight: 600; margin-bottom: 0.25rem; }
    
    /* CHAT CONTAINER */
    .chat-container {
        height: 60vh;
        overflow-y: auto;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    section[data-testid="stSidebar"] { background-color: #0e1117; }
    .stButton button { background-color: #1a1a1a; color: #e0e0e0; border: 1px solid #333; }
    .stButton button:hover { background-color: #1e3a5f; border-color: #4a9eff; }
    
    /* ==================== LIGHT MODE OVERRIDES ==================== */
    [data-theme="light"] .main { background-color: #ffffff; color: #1a1a1a; }
    [data-theme="light"] .main-header { color: #1e3a8a; }
    [data-theme="light"] .main-subtitle { color: #4b5563; }
    
    [data-theme="light"] .user-msg {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    [data-theme="light"] .assistant-msg {
        background-color: #f3f4f6;
        color: #1f2937;
        border-left: 3px solid #3b82f6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    [data-theme="light"] .assistant-label { color: #1e3a8a; }
    
    [data-theme="light"] .sql-box {
        background-color: #1e293b;
        color: #94a3b8;
        border: 1px solid #334155;
    }
    [data-theme="light"] .sql-label { color: #64748b; }
    
    [data-theme="light"] section[data-testid="stSidebar"] { background-color: #f9fafb; }
    [data-theme="light"] .stButton button { background-color: #ffffff; color: #1f2937; border: 1px solid #d1d5db; }
    [data-theme="light"] .stButton button:hover { background-color: #dbeafe; border-color: #3b82f6; }
</style>
""", unsafe_allow_html=True)

apply_theme()

# ---------- Sidebar ----------
def display_sidebar():
    st.sidebar.title("Data Upload")
    st.sidebar.markdown("---")

    uploaded_file = st.sidebar.file_uploader("Choose file", type=["csv", "xlsx", "xls"])
    table_name = st.sidebar.text_input("Table name", placeholder="e.g., sales_data")
    write_mode = st.sidebar.selectbox("Mode", ["Create/Replace", "Append"], index=0)

    if uploaded_file:
        try:
            if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
                df_preview = pd.read_excel(uploaded_file, nrows=10)
            else:
                df_preview = pd.read_csv(uploaded_file, nrows=10)
            st.sidebar.success(f"{len(df_preview)} rows x {len(df_preview.columns)} cols")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

    if st.sidebar.button("Import", type="primary", use_container_width=True):
        if not uploaded_file or not table_name.strip():
            st.sidebar.error("Select file and enter table name")
            return None
        
        clean_name = table_name.strip().replace(" ", "_").lower()
        try:
            uploaded_file.seek(0)
            if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            # ========== ENHANCED DATE CONVERSION ==========
            date_cols_found = []
            for col in df.columns:
                if df[col].dtype == 'object' or str(df[col].dtype).startswith('datetime'):
                    try:
                        if pd.api.types.is_datetime64_any_dtype(df[col]):
                            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                            date_cols_found.append(col)
                            logger.info(f" Converted datetime column '{col}'")
                            continue
                        
                        parsed = pd.to_datetime(df[col], format='mixed', dayfirst=True, errors='coerce')
                        
                        if parsed.notna().sum() > len(df) * 0.5:
                            df[col] = parsed.dt.strftime('%Y-%m-%d %H:%M:%S')
                            date_cols_found.append(col)
                            logger.info(f" Converted text column '{col}' to SQLite datetime")
                        
                    except Exception as date_err:
                        pass
            
            if date_cols_found:
                logger.info(f" Successfully converted {len(date_cols_found)} date columns: {date_cols_found}")
            # ========== END DATE CONVERSION ==========
            
            mode = "replace" if write_mode.startswith("Create") else "append"
            df.to_sql(clean_name, db_manager.engine, if_exists=mode, index=False)
            
            # Clear metadata cache when new table imported
            if hasattr(sql_chain_manager, 'table_metadata_cache'):
                sql_chain_manager.table_metadata_cache.pop(clean_name, None)
                logger.info(f" Cleared metadata cache for '{clean_name}'")
            
            # Set as active table
            st.session_state.active_table = clean_name
            
            success_msg = f" Imported {len(df):,} rows to '{clean_name}'"
            if date_cols_found:
                success_msg += f"\n Date columns: {', '.join(date_cols_found)}"
            st.sidebar.success(success_msg)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f" Failed: {e}")
            logger.error(f"Import error: {e}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Active Table")
    
    try:
        table_info = db_manager.get_table_info()
        if table_info:
            table_list = list(table_info.keys())
            
            # Set default if none selected
            if st.session_state.active_table is None and table_list:
                st.session_state.active_table = table_list[0]
            
            selected_table = st.sidebar.selectbox(
                "Select table for queries:",
                table_list,
                index=table_list.index(st.session_state.active_table) if st.session_state.active_table in table_list else 0
            )
            
            if selected_table != st.session_state.active_table:
                st.session_state.active_table = selected_table
                st.rerun()
        else:
            st.sidebar.info("No tables. Upload file first.")
            st.session_state.active_table = None
    except Exception as e:
        logger.error(f"Error loading tables: {e}")
        st.session_state.active_table = None

    st.sidebar.markdown("---")
    st.sidebar.subheader("Visualization")
    chart_types = chart_generator.get_available_chart_types()
    chart_options = {"auto": "Auto"} | chart_types
    selected_chart = st.sidebar.selectbox("Chart", list(chart_options.keys()), format_func=lambda x: chart_options[x])

    st.sidebar.markdown("---")
    st.sidebar.subheader("All Tables")
    try:
        table_info = db_manager.get_table_info()
        if table_info:
            for tbl, info in table_info.items():
                with st.sidebar.expander(f" {tbl}", expanded=False):
                    st.text(f"{len(info['columns'])} columns")
                    for col in sorted(info["columns"]):
                        st.text(f"• {col}")
        else:
            st.sidebar.info("No tables available.")
    except Exception as e:
        logger.error(f"Error displaying tables: {e}")
        pass

    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Dark", use_container_width=True, disabled=st.session_state.dark_mode):
            st.session_state.dark_mode = True
            st.rerun()
    with col2:
        if st.button("Light", use_container_width=True, disabled=not st.session_state.dark_mode):
            st.session_state.dark_mode = False
            st.rerun()

    return selected_chart

# ---------- Chat rendering ----------
def render_user_message(content: str):
    st.markdown(f'<div class="user-msg"><div class="user-label">YOU</div><div>{content}</div></div>', unsafe_allow_html=True)

def render_assistant_message(content: str, sql: str = None, df=None, chart_type=None):
    st.markdown(f'<div class="assistant-msg"><div class="assistant-label">ASSISTANT</div><div>{content}</div></div>', unsafe_allow_html=True)
    
    if df is not None and not df.empty:
        with st.expander(" View data table", expanded=False):
            st.dataframe(df, use_container_width=True, height=min(400, len(df) * 35 + 50))
        
        # Only create chart if chart_type is not None
        if chart_type:
            chart_generator.create_chart(df, chart_type, content)
    
    if sql:
        st.markdown(f'<div class="sql-label">Generated SQL:</div><div class="sql-box">{sql}</div>', unsafe_allow_html=True)

# ---------- Query processing ----------
def process_query(user_input: str, chart_type: str):
    """Process user query and generate SQL."""
    if not st.session_state.active_table:
        return None, None, " Please select or upload a table first!", None
    
    progress_placeholder = st.empty()
    
    try:
        # Step 1: Generate SQL
        progress_placeholder.info(" Generating SQL query...")
        sql_query = sql_chain_manager.generate_sql_query(user_input, st.session_state.active_table)
        
        # Step 2: Validate
        progress_placeholder.info(" Validating query...")
        sql_chain_manager.validate_sql_query(sql_query)
        
        # Step 3: Execute
        progress_placeholder.info(" Executing query...")
        df = db_manager.execute_query(sql_query)
        
        progress_placeholder.empty()
        
        if df.empty:
            db_manager.save_chat_history(user_input, sql_query, None, None, "success")
            return None, sql_query, "Query executed successfully but returned no results.", None
        
        # Auto-select chart type
        if chart_type == "auto":
            chart_type = chart_generator.suggest_chart_type(df)
        
        # Generate natural language response using LLM
        progress_placeholder.info(" Generating insights...")
        explanation = sql_chain_manager.explain_query_result(user_input, sql_query, df)
        progress_placeholder.empty()
        
        # Save history
        db_manager.save_chat_history(
            user_input, 
            sql_query, 
            chart_type, 
            df.to_json(orient="records"), 
            "success"
        )
        
        return df, sql_query, explanation, chart_type
        
    except Exception as e:
        progress_placeholder.empty()
        error_msg = str(e)
        logger.error(f"Error: {error_msg}")
        
        db_manager.save_chat_history(
            user_input, 
            sql_query if "sql_query" in locals() else "N/A", 
            chart_type, 
            None, 
            "error"
        )
        
        return None, None, f" Error: {error_msg}", None

# ---------- Main ----------
def main():
    chart_type = display_sidebar()
    
    st.markdown('<h1 class="main-header">Natural Language SQL Assistant</h1>', unsafe_allow_html=True)
    
    # Show active table
    if st.session_state.active_table:
        st.markdown(f'<p class="main-subtitle">Currently querying: <strong>{st.session_state.active_table}</strong></p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="main-subtitle">Upload and select a table to start</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                render_user_message(msg["content"])
            else:
                render_assistant_message(
                    msg["content"], 
                    msg.get("sql"),
                    msg.get("df"),
                    msg.get("chart_type")
                )
    
    # Check if there's a pending query to process
    if st.session_state.pending_query:
        query_text = st.session_state.pending_query
        st.session_state.pending_query = None
        
        # Show loading indicator
        with st.spinner("Thinking..."):
            df, sql, response, chart = process_query(query_text, chart_type)
        
        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "sql": sql,
            "df": df,
            "chart_type": chart,
            "timestamp": datetime.now().isoformat()
        })
        st.rerun()
    
    # Fixed chat input at bottom
    user_input = st.chat_input("Ask anything about your data...")
    
    if user_input:
        # Add user message and mark for processing
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input, 
            "timestamp": datetime.now().isoformat()
        })
        st.session_state.pending_query = user_input
        st.rerun()
    
    st.markdown("---")
    st.markdown('<div style="text-align:center; color:#666; font-size:0.85rem;">Powered by Ollama + LangChain + Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()