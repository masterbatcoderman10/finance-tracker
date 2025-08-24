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
from config_manager import display_environment_status

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
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = None

# Two-stage workflow session state
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = 1  # 1: inputs, 2: classification review, 3: results
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = None
if 'edited_classifications' not in st.session_state:
    st.session_state.edited_classifications = None
if 'stage1_data' not in st.session_state:
    st.session_state.stage1_data = None
if 'stage1_config' not in st.session_state:
    st.session_state.stage1_config = None
if 'progress_data' not in st.session_state:
    st.session_state.progress_data = {
        'current_step': 0,
        'progress_percentage': 0,
        'current_message': "Initializing...",
        'step_details': {},
        'start_time': None,
        'estimated_completion': None
    }

def main():
    # Header Section
    st.title("📊 Finance Tracker")
    st.subheader("Bank Statement Processing & Financial Analysis")
    
    # Stage indicator
    stage_names = ["📄 File Upload & Configuration", "🔍 Review Classifications", "📊 Results & Downloads"]
    current_stage_name = stage_names[st.session_state.current_stage - 1]
    st.info(f"**Current Stage:** {current_stage_name}")
    
    # Navigation breadcrumbs
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.current_stage == 1:
            st.markdown("🔵 **Stage 1**")
        else:
            st.markdown("✅ Stage 1")
    with col2:
        if st.session_state.current_stage == 2:
            st.markdown("🔵 **Stage 2**")
        elif st.session_state.current_stage > 2:
            st.markdown("✅ Stage 2")
        else:
            st.markdown("⚪ Stage 2")
    with col3:
        if st.session_state.current_stage == 3:
            st.markdown("🔵 **Stage 3**")
        else:
            st.markdown("⚪ Stage 3")
    
    st.markdown("---")
    
    # Conditional rendering based on current stage
    if st.session_state.current_stage == 1:
        render_stage1()
    elif st.session_state.current_stage == 2:
        render_stage2()
    elif st.session_state.current_stage == 3:
        render_stage3()


def render_stage1():
    """Render Stage 1: File Upload & Configuration"""
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
        4. Click "Start Processing" to extract and classify transactions
        5. Review and edit AI classifications
        6. Generate final reports and download results
        
        **What you'll get:**
        - Categorized transactions CSV file
        - Comprehensive Excel financial report with charts
        - Monthly spending analysis
        - Updated transaction classification cache (for faster future processing)
        """)
    
    # Environment validation and API key input
    api_key_from_env = os.getenv('OPENAI_API_KEY')
    
    if not api_key_from_env:
        st.warning("⚠️ OpenAI API key not found in environment variables")
        
        # Show API key input field
        st.subheader("🔑 OpenAI API Key Required")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            api_key_input = st.text_input(
                "Enter your OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Enter your OpenAI API key to enable transaction classification. This key will only be used for this session and will not be saved.",
                value=st.session_state.openai_api_key or ""
            )
        with col2:
            if st.button("💾 Use Key", help="Use this API key for the current session"):
                if api_key_input and len(api_key_input.strip()) >= 20:
                    st.session_state.openai_api_key = api_key_input.strip()
                    st.success("✅ API key accepted")
                    st.rerun()
                else:
                    st.error("❌ Invalid API key format")
        
        # Show instructions for getting API key
        with st.expander("🔧 How to get an OpenAI API Key", expanded=False):
            st.markdown("""
            **To get your OpenAI API key:**
            
            1. Visit [OpenAI API Keys](https://platform.openai.com/account/api-keys)
            2. Sign in to your OpenAI account (or create one)
            3. Click "Create new secret key"
            4. Copy the key and paste it in the field above
            5. Make sure your account has sufficient credits for API usage
            
            **Security Note:** Your API key will only be used during this session and will not be saved or cached.
            """)
        
        # If no valid API key is provided, don't proceed
        if not st.session_state.openai_api_key:
            st.stop()
    else:
        # Show that environment key is being used
        st.success("✅ Using OpenAI API key from environment variables")
    
    # Validate other environment requirements
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
    st.subheader("🚀 Start Processing")
    
    # Process button
    can_process = (
        uploaded_file is not None and 
        st.session_state.processing_status != "processing"
    )
    
    if st.button(
        "Start Processing →", 
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
                'report_filename': custom_report_name if custom_report_name else None,
                'openai_api_key': st.session_state.openai_api_key  # Include API key from session
            }
            
            # Start stage 1 processing
            process_stage1_only(config)
        else:
            st.error("Please upload a PDF file first.")
    
    # Processing Status Display
    if st.session_state.processing_status == "processing":
        st.info("🔄 Processing stage 1: PDF extraction and classification...")
        
        # Show processing indicator
        with st.spinner("Processing your bank statement..."):
            st.write("📄 Extracting transactions from PDF")
            st.write("🔍 Identifying transactions needing classification") 
            st.write("🤖 Running AI classification on new transactions")
            st.write("⏳ This may take a few moments depending on the number of transactions")
    
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
    
    # Show stage completion messages only for stage 1
    elif st.session_state.current_stage == 1:
        if (st.session_state.classification_results is not None and 
            isinstance(st.session_state.classification_results, pd.DataFrame) and 
            not st.session_state.classification_results.empty):
            st.success("✅ Stage 1 completed! Continue to review classifications.")
            
            if st.button("Continue to Review Classifications →", type="primary", use_container_width=True):
                st.session_state.current_stage = 2
                st.rerun()
        
        elif (st.session_state.stage1_data is not None and 
              st.session_state.classification_results is not None and
              isinstance(st.session_state.classification_results, pd.DataFrame) and 
              st.session_state.classification_results.empty):
            st.success("✅ No new classifications needed! Continue to final processing.")
            
            if st.button("Continue to Final Processing →", type="primary", use_container_width=True):
                st.session_state.current_stage = 3
                st.rerun()


def render_stage2():
    """Render Stage 2: Review and Edit Classifications"""
    st.subheader("🔍 Review Classification Results")
    
    if (st.session_state.classification_results is None or 
        (isinstance(st.session_state.classification_results, pd.DataFrame) and st.session_state.classification_results.empty)):
        st.error("No classification results available. Please go back to Stage 1.")
        if st.button("← Back to Stage 1"):
            st.session_state.current_stage = 1
            st.rerun()
        return
    
    st.markdown("""
    Review the AI-generated transaction classifications below. You can edit any category, keyword, or reasoning 
    before proceeding to generate the final reports.
    """)
    
    # Display summary info
    if st.session_state.stage1_data:
        progress_info = st.session_state.stage1_data.get('progress_info', {})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Transactions Found", progress_info.get('transactions_found', 0))
        with col2:
            st.metric("Cached Keywords", progress_info.get('cached_keywords', 0))
        with col3:
            st.metric("Need Classification", progress_info.get('uncached_transactions', 0))
        with col4:
            st.metric("New Classifications", progress_info.get('new_classifications', 0))
    
    st.markdown("---")
    
    # Data editor for classification results
    st.subheader("📝 Edit Classification Results")
    
    # Get available categories for dropdown
    from transaction_analyzer import TransactionCategory
    available_categories = [cat.value for cat in TransactionCategory]
    
    # Configure column types for the data editor
    column_config = {
        "category": st.column_config.SelectboxColumn(
            "Category",
            help="Select the transaction category",
            options=available_categories,
            required=True,
        ),
        "confidence": st.column_config.NumberColumn(
            "Confidence",
            help="Confidence score (0.0 to 1.0)",
            min_value=0.0,
            max_value=1.0,
            step=0.1,
        ),
        "keyword": st.column_config.TextColumn(
            "Keyword",
            help="Key merchant identifier for caching",
            max_chars=100,
        ),
        "reasoning": st.column_config.TextColumn(
            "Reasoning",
            help="Explanation for the classification",
            max_chars=500,
        )
    }
    
    # Display the editable dataframe
    edited_classifications = st.data_editor(
        st.session_state.classification_results,
        column_config=column_config,
        use_container_width=True,
        num_rows="dynamic",
        key="classification_editor"
    )
    
    # Validation
    st.markdown("---")
    
    # Show changes
    try:
        has_changes = not edited_classifications.equals(st.session_state.classification_results)
    except:
        has_changes = True  # Assume changes if comparison fails
    
    if has_changes:
        st.info("🔄 Classifications have been modified")
        
        # Count changes
        changes = 0
        try:
            for idx in edited_classifications.index:
                if idx in st.session_state.classification_results.index:
                    if not edited_classifications.loc[idx].equals(st.session_state.classification_results.loc[idx]):
                        changes += 1
                else:
                    changes += 1
        except:
            changes = len(edited_classifications)
        
        st.write(f"**Changes detected:** {changes} rows modified")
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Stage 1", use_container_width=True):
            st.session_state.current_stage = 1
            st.rerun()
    
    with col2:
        if st.button("Continue to Stage 3 →", type="primary", use_container_width=True):
            # Validate classifications before proceeding
            if edited_classifications.empty:
                st.error("Cannot proceed with empty classifications")
                return
            
            # Check for required columns
            required_columns = ['category', 'keyword', 'confidence']
            missing_columns = [col for col in required_columns if col not in edited_classifications.columns]
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return
            
            # Save edited classifications and proceed
            st.session_state.edited_classifications = edited_classifications
            st.session_state.current_stage = 3
            st.rerun()


def render_stage3():
    """Render Stage 3: Final Processing and Results"""
    st.subheader("🚀 Final Processing & Results")
    
    # Check if we have the required data - edited_classifications can be empty if no new classifications were needed
    if (st.session_state.edited_classifications is None or 
        st.session_state.stage1_data is None):
        st.error("Missing required data from previous stages.")
        if st.button("← Back to Stage 2"):
            st.session_state.current_stage = 2
            st.rerun()
        return
    
    # Check if final processing is complete
    if st.session_state.results is None:
        st.markdown("Ready to complete processing with your edited classifications.")
        
        if st.button("🚀 Complete Processing", type="primary", use_container_width=True):
            # Start stage 2 processing
            try:
                st.session_state.processing_status = "processing"
                st.info("🔄 Applying classifications and generating final reports...")
                
                # Import and use the workflow
                from streamlit_workflow import StreamlitBankStatementWorkflow
                from config_manager import get_config_manager
                
                # Get API key
                api_key = st.session_state.stage1_config.get('openai_api_key') or os.getenv("OPENAI_API_KEY")
                
                # Validate config
                config_manager = get_config_manager()
                is_valid, errors, config = config_manager.validate_config(st.session_state.stage1_config)
                
                if not is_valid:
                    st.error(f"Configuration validation failed: {'; '.join(errors)}")
                    return
                
                # Create workflow and process stage 2
                workflow = StreamlitBankStatementWorkflow(api_key, config.model)
                
                # Run stage 2 processing
                import asyncio
                results = asyncio.run(
                    workflow.process_stage2_with_progress(
                        config, 
                        st.session_state.stage1_data,
                        st.session_state.edited_classifications
                    )
                )
                
                st.session_state.results = results
                st.session_state.processing_status = "complete"
                st.rerun()
                
            except Exception as e:
                st.session_state.processing_status = "error"
                st.session_state.error_message = str(e)
                st.error(f"Processing failed: {str(e)}")
    else:
        # Show results
        st.success("✅ Processing completed successfully!")
        display_results()
        
        # Navigation
        if st.button("🔄 Start New Processing", use_container_width=True):
            # Reset all session state for new processing
            st.session_state.current_stage = 1
            st.session_state.classification_results = None
            st.session_state.edited_classifications = None
            st.session_state.stage1_data = None
            st.session_state.stage1_config = None
            st.session_state.results = None
            st.session_state.processing_status = "idle"
            st.rerun()


def process_stage1_only(config):
    """
    Process stage 1: PDF extraction through classification
    """
    try:
        # Store config for later use
        st.session_state.stage1_config = config
        
        # Import and use the workflow
        from streamlit_workflow import StreamlitBankStatementWorkflow
        from config_manager import get_config_manager
        
        # Validate and create configuration
        config_manager = get_config_manager()
        is_valid, errors, validated_config = config_manager.validate_config(config)

        if not is_valid:
            st.session_state.processing_status = "error"
            st.session_state.error_message = f"Configuration validation failed: {'; '.join(errors)}"
            return
        
        # Get API key
        api_key = config.get('openai_api_key') or os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.session_state.processing_status = "error"
            st.session_state.error_message = "OPENAI_API_KEY not found"
            return

        # Create workflow instance
        workflow = StreamlitBankStatementWorkflow(api_key, validated_config.model)
        
        # Run stage 1 processing
        import asyncio
        stage1_data = asyncio.run(
            workflow.process_stage1_with_progress(validated_config)
        )
        
        # Store results in session state
        st.session_state.stage1_data = stage1_data
        st.session_state.classification_results = stage1_data['classification_results']
        
        # Debug output
        print(f"Stage 1 completed. Classification results type: {type(stage1_data['classification_results'])}")
        if isinstance(stage1_data['classification_results'], pd.DataFrame):
            print(f"Classification results shape: {stage1_data['classification_results'].shape}")
            print(f"Classification results empty: {stage1_data['classification_results'].empty}")
        
        # Move to stage 2 if we have classifications to review
        classification_results = stage1_data['classification_results']
        if (isinstance(classification_results, pd.DataFrame) and not classification_results.empty):
            print("Moving to stage 2 - have classifications to review")
            st.session_state.current_stage = 2
        else:
            print("Staying in stage 1 - no classifications to review, user will manually proceed")
            # If no new classifications, prepare empty DataFrame but stay in stage 1 
            # so user can see the "Continue to Final Processing" button
            st.session_state.edited_classifications = pd.DataFrame()
        
        st.session_state.processing_status = "idle"  # Reset processing status
        st.rerun()  # Trigger rerun to show the next stage
        
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