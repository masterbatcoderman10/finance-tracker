#!/usr/bin/env python3
"""
Bank Statement Processing - Streamlit GUI
A user-friendly web interface for processing bank statements and generating financial reports
"""

import streamlit as st
import os
import tempfile
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Import our custom modules
from file_utils import get_file_manager, display_file_validation_results, create_download_button
from config_manager import display_environment_status, create_config_from_ui
from streamlit_workflow import process_bank_statement_streamlit

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Finance Tracker - Bank Statement Processor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None
if 'cache_file_path' not in st.session_state:
    st.session_state.cache_file_path = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = "idle"  # idle, processing, complete, error
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'results' not in st.session_state:
    st.session_state.results = None

def main():
    # Header Section
    st.title("📊 Finance Tracker")
    st.subheader("Bank Statement Processing & Financial Analysis")
    
    st.markdown("""
    **Transform your bank statement PDFs into organized financial insights**
    
    This application processes bank statement PDFs, automatically categorizes transactions using AI, 
    and generates comprehensive Excel reports with spending analysis.
    """)
    
    # Requirements and instructions
    with st.expander("📋 Requirements & Instructions", expanded=False):
        st.markdown("""
        **Requirements:**
        - Bank statement in PDF format
        - OpenAI API key (for transaction classification)
        - PDF password (if the statement is encrypted)
        
        **How it works:**
        1. Upload your bank statement PDF
        2. Enter the PDF password if required
        3. Configure processing options (optional)
        4. Click "Process Bank Statement" 
        5. Monitor real-time progress
        6. Download your financial reports
        
        **What you'll get:**
        - Categorized transactions CSV file
        - Comprehensive Excel financial report with charts
        - Monthly spending analysis
        - Updated transaction classification cache (for faster future processing)
        """)
    
    # Environment validation
    if not display_environment_status():
        st.stop()
    
    # Main layout with columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File Upload Section
        st.subheader("📄 Upload Bank Statement")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Select your bank statement PDF file. The file will be processed securely and temporarily stored during analysis."
        )
        
        if uploaded_file is not None:
            # Display file info
            st.success(f"✅ File uploaded: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
            
            # Save uploaded file temporarily
            if st.session_state.uploaded_file_path is None:
                file_manager = get_file_manager()
                temp_file_path = file_manager.save_uploaded_file(uploaded_file)
                st.session_state.uploaded_file_path = temp_file_path
                
                # Validate PDF file
                display_file_validation_results(temp_file_path, "pdf")
        
        # Cache File Upload Section
        st.subheader("🗂️ Upload Classification Cache (Optional)")
        
        cache_file = st.file_uploader(
            "Upload existing classification cache",
            type=['json'],
            help="Upload your existing classification_keywords.json file to speed up processing by reusing previously classified transactions."
        )
        
        if cache_file is not None:
            st.success(f"✅ Cache file uploaded: {cache_file.name} ({cache_file.size:,} bytes)")
            
            # Save cache file temporarily
            if st.session_state.cache_file_path is None:
                file_manager = get_file_manager()
                cache_file_path = file_manager.save_uploaded_file(cache_file)
                st.session_state.cache_file_path = cache_file_path
                
                # Validate and display cache info
                cache_data = display_file_validation_results(cache_file_path, "json")
                if cache_data:
                    # Display cache statistics
                    if isinstance(cache_data, dict):
                        if 'unknown' in cache_data or 'debits' in cache_data or 'credits' in cache_data:
                            total_keywords = sum(len(type_cache) for type_cache in cache_data.values() if isinstance(type_cache, dict))
                            st.info(f"📊 Cache contains {total_keywords} classification keywords")
                        else:
                            total_keywords = len(cache_data)
                            st.info(f"📊 Legacy cache with {total_keywords} keywords (will be converted)")
        else:
            st.info("💡 **Tip**: Upload a previous classification cache to speed up processing!")
        
        # Password Input
        pdf_password = st.text_input(
            "PDF Password (if required)",
            type="password",
            help="Enter the password if your PDF is encrypted. Leave blank if not required."
        )
    
    with col2:
        # Processing Configuration
        st.subheader("⚙️ Configuration")
        
        with st.expander("Processing Options", expanded=False):
            clean_descriptions = st.checkbox(
                "Clean Descriptions", 
                value=True,
                help="Use AI to clean and standardize transaction descriptions"
            )
            
            force_reclassify = st.checkbox(
                "Force Reclassify",
                value=False, 
                help="Ignore existing cache and reclassify all transactions"
            )
            
            generate_reports = st.checkbox(
                "Generate Excel Reports",
                value=True,
                help="Automatically generate comprehensive Excel financial reports"
            )
            
            save_classification_details = st.checkbox(
                "Save Classification Details",
                value=False,
                help="Save detailed classification results with reasoning and confidence scores"
            )
        
        with st.expander("Advanced Settings", expanded=False):
            max_concurrent = st.slider(
                "Max Concurrent Requests",
                min_value=1,
                max_value=20,
                value=8,
                help="Maximum number of concurrent API requests for classification"
            )
            
            model_choice = st.selectbox(
                "OpenAI Model",
                options=["gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini"],
                index=0,
                help="Choose the OpenAI model for transaction classification"
            )
            
            custom_report_name = st.text_input(
                "Custom Report Filename (optional)",
                placeholder="finance_report_custom.xlsx",
                help="Custom name for the Excel report file"
            )
    
    # Processing Section
    st.subheader("🚀 Process Bank Statement")
    
    # Process button
    can_process = (
        uploaded_file is not None and 
        st.session_state.processing_status != "processing"
    )
    
    if st.button(
        "Process Bank Statement", 
        disabled=not can_process,
        use_container_width=True,
        type="primary"
    ):
        if st.session_state.uploaded_file_path:
            st.session_state.processing_status = "processing"
            st.session_state.error_message = None
            st.session_state.processing_complete = False
            
            # Create processing configuration
            config = {
                'pdf_path': st.session_state.uploaded_file_path,
                'password': pdf_password if pdf_password else None,
                'cache_file_path': st.session_state.cache_file_path,
                'clean_descriptions': clean_descriptions,
                'force_reclassify': force_reclassify,
                'generate_reports': generate_reports,
                'save_classification_details': save_classification_details,
                'max_concurrent': max_concurrent,
                'model': model_choice,
                'report_filename': custom_report_name if custom_report_name else None
            }
            
            # Start processing (this will be implemented in the workflow adapter)
            process_bank_statement(config)
        else:
            st.error("Please upload a PDF file first.")
    
    # Processing Status Display
    if st.session_state.processing_status == "processing":
        st.info("🔄 Processing your bank statement...")
        
        # Progress placeholder (will be populated by progress tracker)
        progress_container = st.container()
        with progress_container:
            st.progress(0, text="Initializing...")
            st.empty()  # Placeholder for detailed progress
    
    elif st.session_state.processing_status == "complete":
        st.success("✅ Processing completed successfully!")
        display_results()
    
    elif st.session_state.processing_status == "error":
        st.error(f"❌ Processing failed: {st.session_state.error_message}")
        
        # Error recovery suggestions
        with st.expander("🔧 Troubleshooting", expanded=True):
            st.markdown("""
            **Common issues and solutions:**
            
            1. **PDF Password Error**: Ensure the password is correct and try again
            2. **API Key Error**: Check that your OpenAI API key is valid and has sufficient credits
            3. **File Format Error**: Ensure the file is a valid PDF with extractable text
            4. **Network Error**: Check your internet connection and try again
            5. **Processing Timeout**: Try reducing max concurrent requests or contact support
            """)
        
        if st.button("🔄 Try Again", use_container_width=True):
            st.session_state.processing_status = "idle"
            st.session_state.error_message = None
            st.rerun()

def process_bank_statement(config):
    """
    Process the bank statement with the given configuration
    """
    try:
        # Create configuration for validation
        success, validated_config = create_config_from_ui(config)
        if not success:
            st.session_state.processing_status = "error"
            st.session_state.error_message = "Configuration validation failed"
            return
        
        # Process using the workflow adapter
        results = process_bank_statement_streamlit(config)
        
        # Update session state with results
        st.session_state.processing_status = "complete"
        st.session_state.results = results
        
    except Exception as e:
        st.session_state.processing_status = "error"
        st.session_state.error_message = str(e)

def display_results():
    """Display processing results and download options"""
    if not st.session_state.results:
        return
    
    st.subheader("📊 Processing Results")
    
    # Results summary
    results = st.session_state.results
    files = results.get('files_generated', {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Transactions Processed", 
            results.get('transactions_processed', 0)
        )
    with col2:
        st.metric(
            "Categories Found", 
            results.get('categories_found', 0)
        )
    with col3:
        categorized = results.get('categorized_transactions', 0)
        total = results.get('transactions_processed', 1)
        categorization_rate = f"{(categorized / total * 100):.0f}%" if total > 0 else "0%"
        st.metric(
            "Categorization Rate",
            categorization_rate
        )
    with col4:
        processing_time = results.get('processing_time_formatted', 'Unknown')
        st.metric(
            "Processing Time", 
            processing_time
        )
    
    # Category breakdown
    if 'category_breakdown' in results:
        st.subheader("📋 Category Breakdown")
        category_df = pd.DataFrame(
            list(results['category_breakdown'].items()),
            columns=['Category', 'Transactions']
        ).sort_values('Transactions', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(category_df, use_container_width=True)
        with col2:
            st.bar_chart(category_df.set_index('Category')['Transactions'])
    
    # Download Section
    st.subheader("📥 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Excel Report
        excel_file = files.get('excel_report')
        if excel_file and os.path.exists(excel_file):
            create_download_button(
                label="📊 Excel Report",
                file_path=excel_file,
                filename="finance_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help_text="Comprehensive Excel report with monthly analysis and charts"
            )
        else:
            st.info("Excel report not generated")
    
    with col2:
        # Transactions CSV
        csv_file = files.get('transactions_csv')
        if csv_file and os.path.exists(csv_file):
            create_download_button(
                label="📄 Transactions CSV",
                file_path=csv_file,
                filename="categorized_transactions.csv",
                mime="text/csv",
                help_text="Categorized transactions in CSV format"
            )
        else:
            st.info("CSV file not available")
    
    with col3:
        # Classification Cache
        cache_file = files.get('updated_cache')
        if cache_file and os.path.exists(cache_file):
            create_download_button(
                label="🔧 Updated Cache",
                file_path=cache_file,
                filename="classification_keywords.json",
                mime="application/json",
                help_text="Updated keyword cache for faster future processing"
            )
        else:
            st.info("Cache file not available")
    
    # Optional classification details
    details_file = files.get('classification_details')
    if details_file and os.path.exists(details_file):
        st.subheader("📋 Additional Downloads")
        create_download_button(
            label="📊 Classification Details",
            file_path=details_file,
            filename="classification_details.csv",
            mime="text/csv",
            help_text="Detailed classification results with reasoning and confidence scores"
        )
    
    # Cache statistics
    if 'cache_stats' in results:
        st.subheader("🗂️ Cache Statistics")
        cache_stats = results['cache_stats']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Keywords", cache_stats.get('total_keywords', 0))
        with col2:
            st.metric("Cache Hit Rate", cache_stats.get('cache_hit_rate', '0%'))

# Footer
def show_footer():
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666666; padding: 20px;'>
            <small>
            Finance Tracker v1.0 | Built with Streamlit | 
            Powered by OpenAI GPT for transaction classification
            </small>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    show_footer()