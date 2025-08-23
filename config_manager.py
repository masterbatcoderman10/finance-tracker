#!/usr/bin/env python3
"""
Configuration Management for Streamlit Finance Tracker
Handles validation and management of processing configuration options
"""

import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import streamlit as st


@dataclass
class ProcessingConfig:
    """Configuration for bank statement processing"""
    pdf_path: str
    password: Optional[str] = None
    cache_file_path: Optional[str] = None
    clean_descriptions: bool = True
    force_reclassify: bool = False
    generate_reports: bool = True
    save_classification_details: bool = False
    max_concurrent: int = 8
    model: str = "gpt-4.1-mini"
    report_filename: Optional[str] = None
    openai_api_key: Optional[str] = None  # Allow API key from UI
    
    # Generated filenames (will be set by ConfigManager)
    output_csv: Optional[str] = None
    cache_output: Optional[str] = None
    classification_details_file: Optional[str] = None


class ConfigManager:
    """
    Manages configuration validation and file naming for the Streamlit app
    """
    
    # Available OpenAI models
    AVAILABLE_MODELS = [
        "gpt-4.1-mini",
        "gpt-4.1-nano", 
        "gpt-4o-mini"
    ]
    
    # File naming patterns
    DEFAULT_FILES = {
        'output_csv': 'categorized_transactions.csv',
        'cache_output': 'classification_keywords.json',
        'classification_details': 'classification_results.csv'
    }
    
    def __init__(self):
        self.temp_dir = None
    
    def validate_config(self, config_dict: Dict[str, Any]) -> Tuple[bool, List[str], Optional[ProcessingConfig]]:
        """
        Validate the processing configuration
        
        Args:
            config_dict: Configuration dictionary from Streamlit inputs
            
        Returns:
            (is_valid, error_messages, validated_config)
        """
        errors = []
        
        # Validate required fields
        if not config_dict.get('pdf_path'):
            errors.append("PDF file path is required")
        elif not os.path.exists(config_dict['pdf_path']):
            errors.append(f"PDF file does not exist: {config_dict['pdf_path']}")
        
        # Validate OpenAI API key (from environment or UI input)
        api_key = config_dict.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            errors.append("OpenAI API key is required (either from environment variable or input field)")
        elif len(api_key.strip()) < 20:
            errors.append("OpenAI API key appears to be invalid (too short)")
        
        # Validate model selection
        model = config_dict.get('model', 'gpt-4.1-mini')
        if model not in self.AVAILABLE_MODELS:
            errors.append(f"Invalid model '{model}'. Available models: {', '.join(self.AVAILABLE_MODELS)}")
        
        # Validate concurrency settings
        max_concurrent = config_dict.get('max_concurrent', 8)
        if not isinstance(max_concurrent, int) or max_concurrent < 1 or max_concurrent > 20:
            errors.append("max_concurrent must be an integer between 1 and 20")
        
        # Validate cache file if provided
        cache_file_path = config_dict.get('cache_file_path')
        if cache_file_path and not os.path.exists(cache_file_path):
            errors.append(f"Cache file does not exist: {cache_file_path}")
        
        # Return early if there are validation errors
        if errors:
            return False, errors, None
        
        # Create validated configuration
        try:
            config = ProcessingConfig(
                pdf_path=config_dict['pdf_path'],
                password=config_dict.get('password'),
                cache_file_path=cache_file_path,
                clean_descriptions=config_dict.get('clean_descriptions', True),
                force_reclassify=config_dict.get('force_reclassify', False),
                generate_reports=config_dict.get('generate_reports', True),
                save_classification_details=config_dict.get('save_classification_details', False),
                max_concurrent=max_concurrent,
                model=model,
                report_filename=config_dict.get('report_filename'),
                openai_api_key=config_dict.get('openai_api_key')
            )
            
            # Generate output file paths
            self._generate_output_paths(config)
            
            return True, [], config
            
        except Exception as e:
            errors.append(f"Error creating configuration: {str(e)}")
            return False, errors, None
    
    def _generate_output_paths(self, config: ProcessingConfig):
        """
        Generate output file paths for the configuration
        
        Args:
            config: ProcessingConfig to update with file paths
        """
        # Create temporary directory for outputs if not exists
        if not self.temp_dir:
            import tempfile
            self.temp_dir = tempfile.mkdtemp()
        
        # Generate unique filenames based on timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set output paths
        config.output_csv = os.path.join(self.temp_dir, f"transactions_{timestamp}.csv")
        config.cache_output = os.path.join(self.temp_dir, f"cache_{timestamp}.json")
        
        if config.save_classification_details:
            config.classification_details_file = os.path.join(
                self.temp_dir, f"classification_details_{timestamp}.csv"
            )
        
        # Handle report filename
        if config.generate_reports:
            if config.report_filename:
                # Use custom filename
                report_path = os.path.join(self.temp_dir, config.report_filename)
            else:
                # Generate default filename
                report_path = os.path.join(self.temp_dir, f"finance_report_{timestamp}.xlsx")
            
            config.report_filename = report_path
    
    def get_configuration_summary(self, config: ProcessingConfig) -> Dict[str, Any]:
        """
        Get a summary of the configuration for display
        
        Args:
            config: ProcessingConfig instance
            
        Returns:
            Dictionary with configuration summary
        """
        summary = {
            "PDF File": os.path.basename(config.pdf_path),
            "Password Protected": "Yes" if config.password else "No",
            "Using Cache File": "Yes" if config.cache_file_path else "No",
            "Clean Descriptions": "Yes" if config.clean_descriptions else "No",
            "Force Reclassify": "Yes" if config.force_reclassify else "No",
            "Generate Reports": "Yes" if config.generate_reports else "No",
            "Save Classification Details": "Yes" if config.save_classification_details else "No",
            "Max Concurrent Requests": config.max_concurrent,
            "OpenAI Model": config.model
        }
        
        if config.cache_file_path:
            summary["Cache File"] = os.path.basename(config.cache_file_path)
        
        return summary
    
    def display_configuration_review(self, config: ProcessingConfig):
        """
        Display configuration review in Streamlit
        
        Args:
            config: ProcessingConfig to display
        """
        st.subheader("📋 Configuration Review")
        
        summary = self.get_configuration_summary(config)
        
        # Create columns for organized display
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**File Settings:**")
            st.write(f"• PDF: {summary['PDF File']}")
            st.write(f"• Password Protected: {summary['Password Protected']}")
            st.write(f"• Using Cache: {summary['Using Cache File']}")
            
            st.write("**Processing Options:**")
            st.write(f"• Clean Descriptions: {summary['Clean Descriptions']}")
            st.write(f"• Force Reclassify: {summary['Force Reclassify']}")
        
        with col2:
            st.write("**Output Options:**")
            st.write(f"• Generate Reports: {summary['Generate Reports']}")
            st.write(f"• Save Details: {summary['Save Classification Details']}")
            
            st.write("**Advanced Settings:**")
            st.write(f"• Max Concurrent: {summary['Max Concurrent Requests']}")
            st.write(f"• AI Model: {summary['OpenAI Model']}")
    
    def validate_environment(self) -> Tuple[bool, List[str]]:
        """
        Validate that the environment is properly configured
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check OpenAI API key (but don't require it at environment level since UI can provide it)
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            # This is not an error anymore - UI can provide the key
            pass
        elif len(api_key.strip()) < 20:
            errors.append("OPENAI_API_KEY environment variable appears to be invalid")
        
        # Check required packages are importable
        required_packages = [
            ('openai', 'OpenAI API client'),
            ('pandas', 'Data processing'),
            ('pdfplumber', 'PDF processing'),
            ('openpyxl', 'Excel file generation')
        ]
        
        for package_name, description in required_packages:
            try:
                __import__(package_name)
            except ImportError:
                errors.append(f"Required package '{package_name}' is not installed ({description})")
        
        return len(errors) == 0, errors
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration values
        
        Returns:
            Dictionary with default values
        """
        return {
            'clean_descriptions': True,
            'force_reclassify': False,
            'generate_reports': True,
            'save_classification_details': False,
            'max_concurrent': 8,
            'model': 'gpt-4.1-mini'
        }
    
    def cleanup_temp_files(self):
        """Clean up temporary files created by this manager"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                self.temp_dir = None
            except Exception as e:
                st.warning(f"Could not clean up temporary directory: {str(e)}")


# Streamlit helper functions
def display_environment_status():
    """Display environment validation status in Streamlit (excluding API key check)"""
    config_manager = ConfigManager()
    is_valid, errors = config_manager.validate_environment()
    
    # Filter out API key errors since those are handled by UI input
    filtered_errors = [error for error in errors if not error.startswith("OPENAI_API_KEY")]
    
    if len(filtered_errors) == 0:
        if len(errors) == 0:
            st.success("✅ Environment is properly configured")
        else:
            st.info("ℹ️ Environment packages are configured (API key handled separately)")
    else:
        st.error("❌ Environment configuration issues detected:")
        for error in filtered_errors:
            st.write(f"• {error}")
        
        with st.expander("🔧 Setup Instructions", expanded=True):
            st.markdown("""
            **To fix these issues:**
            
            1. **Install required packages:**
            ```bash
            pip install -r requirements.txt
            ```
            """)
    
    return len(filtered_errors) == 0


def create_config_from_ui(config_dict: Dict[str, Any]) -> Tuple[bool, ProcessingConfig]:
    """
    Create and validate configuration from Streamlit UI inputs
    
    Args:
        config_dict: Dictionary of configuration values from UI
        
    Returns:
        (success, config_or_none)
    """
    config_manager = ConfigManager()
    is_valid, errors, config = config_manager.validate_config(config_dict)
    
    if not is_valid:
        st.error("❌ Configuration validation failed:")
        for error in errors:
            st.write(f"• {error}")
        return False, None
    
    # Display configuration review
    config_manager.display_configuration_review(config)
    
    return True, config


# Global config manager instance
@st.cache_resource
def get_config_manager() -> ConfigManager:
    """Get or create the configuration manager instance"""
    return ConfigManager()