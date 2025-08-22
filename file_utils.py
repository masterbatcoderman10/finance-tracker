#!/usr/bin/env python3
"""
File Management Utilities for Streamlit Finance Tracker
Handles file upload, validation, temporary storage, and download preparation
"""

import os
import json
import tempfile
import shutil
from typing import Optional, Dict, Any, Tuple
import streamlit as st
from pathlib import Path


class FileManager:
    """
    Manages file operations for the Streamlit application
    """
    
    def __init__(self):
        self.temp_dirs = []
        
    def create_temp_dir(self) -> str:
        """Create a temporary directory and track it for cleanup"""
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def save_uploaded_file(self, uploaded_file, filename: Optional[str] = None) -> str:
        """
        Save an uploaded Streamlit file to temporary storage
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            filename: Custom filename (optional)
            
        Returns:
            Path to saved temporary file
        """
        temp_dir = self.create_temp_dir()
        
        if filename is None:
            filename = uploaded_file.name
            
        temp_file_path = os.path.join(temp_dir, filename)
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        return temp_file_path
    
    def validate_pdf_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate that a file is a readable PDF
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            (is_valid, error_message)
        """
        try:
            if not os.path.exists(file_path):
                return False, "File does not exist"
                
            if not file_path.lower().endswith('.pdf'):
                return False, "File is not a PDF"
                
            # Check file size (reasonable limit: 50MB)
            file_size = os.path.getsize(file_path)
            if file_size > 50 * 1024 * 1024:
                return False, "PDF file is too large (>50MB)"
                
            if file_size == 0:
                return False, "PDF file is empty"
                
            # Try to open with pdfplumber to verify it's a valid PDF
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    # Just try to access the first page
                    if len(pdf.pages) == 0:
                        return False, "PDF has no pages"
            except Exception as e:
                return False, f"Invalid PDF format: {str(e)}"
                
            return True, "Valid PDF file"
            
        except Exception as e:
            return False, f"Error validating PDF: {str(e)}"
    
    def validate_cache_file(self, file_path: str) -> Tuple[bool, str, Optional[Dict]]:
        """
        Validate and load classification cache JSON file
        
        Args:
            file_path: Path to cache JSON file
            
        Returns:
            (is_valid, message, cache_data)
        """
        try:
            if not os.path.exists(file_path):
                return False, "Cache file does not exist", None
                
            if not file_path.lower().endswith('.json'):
                return False, "Cache file is not a JSON file", None
                
            # Check file size (reasonable limit: 10MB for cache)
            file_size = os.path.getsize(file_path)
            if file_size > 10 * 1024 * 1024:
                return False, "Cache file is too large (>10MB)", None
                
            if file_size == 0:
                return False, "Cache file is empty", None
            
            # Try to load and validate JSON structure
            with open(file_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            if not isinstance(cache_data, dict):
                return False, "Cache file must contain a JSON object", None
            
            # Check for hierarchical format vs flat format
            total_keywords = 0
            format_type = "unknown"
            
            if 'unknown' in cache_data or 'debits' in cache_data or 'credits' in cache_data:
                # New hierarchical format
                format_type = "hierarchical"
                for section_name, section_data in cache_data.items():
                    if isinstance(section_data, dict):
                        total_keywords += len(section_data)
                message = f"Valid hierarchical cache with {total_keywords} keywords"
            else:
                # Check if it's a flat format (old style)
                if all(isinstance(v, str) for v in cache_data.values()):
                    format_type = "flat"
                    total_keywords = len(cache_data)
                    message = f"Valid flat cache with {total_keywords} keywords (will be converted)"
                else:
                    return False, "Invalid cache file format", None
            
            return True, message, cache_data
            
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON format: {str(e)}", None
        except Exception as e:
            return False, f"Error validating cache file: {str(e)}", None
    
    def prepare_download_data(self, file_path: str) -> bytes:
        """
        Prepare file data for Streamlit download
        
        Args:
            file_path: Path to file to prepare for download
            
        Returns:
            File data as bytes
        """
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            st.error(f"Error preparing download: {str(e)}")
            return b""
    
    def prepare_json_download(self, data: Dict[str, Any], indent: int = 2) -> bytes:
        """
        Prepare JSON data for download
        
        Args:
            data: Dictionary to convert to JSON
            indent: JSON indentation level
            
        Returns:
            JSON data as bytes
        """
        try:
            json_string = json.dumps(data, indent=indent, ensure_ascii=False)
            return json_string.encode('utf-8')
        except Exception as e:
            st.error(f"Error preparing JSON download: {str(e)}")
            return b""
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get file information for display
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        try:
            if not os.path.exists(file_path):
                return {"exists": False}
                
            stat = os.stat(file_path)
            return {
                "exists": True,
                "name": os.path.basename(file_path),
                "size": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": stat.st_mtime
            }
        except Exception:
            return {"exists": False}
    
    def cleanup_temp_files(self):
        """Clean up all temporary directories created by this manager"""
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                st.warning(f"Could not clean up temporary directory {temp_dir}: {str(e)}")
        self.temp_dirs.clear()
    
    def __del__(self):
        """Cleanup on object destruction"""
        self.cleanup_temp_files()


# Global file manager instance for the Streamlit session
@st.cache_resource
def get_file_manager() -> FileManager:
    """Get or create the file manager instance"""
    return FileManager()


def display_file_validation_results(file_path: str, file_type: str = "pdf"):
    """
    Display file validation results in Streamlit UI
    
    Args:
        file_path: Path to file to validate
        file_type: Type of file ('pdf' or 'json')
    """
    file_manager = get_file_manager()
    
    if file_type == "pdf":
        is_valid, message = file_manager.validate_pdf_file(file_path)
        if is_valid:
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")
    
    elif file_type == "json":
        is_valid, message, cache_data = file_manager.validate_cache_file(file_path)
        if is_valid:
            st.success(f"✅ {message}")
            return cache_data
        else:
            st.error(f"❌ {message}")
            return None
    
    return is_valid


def create_download_button(
    label: str,
    file_path: Optional[str] = None,
    data: Optional[bytes] = None,
    filename: str = "download.txt",
    mime: str = "text/plain",
    help_text: Optional[str] = None
):
    """
    Create a Streamlit download button with file or data
    
    Args:
        label: Button label
        file_path: Path to file to download (optional)
        data: Raw data to download (optional) 
        filename: Download filename
        mime: MIME type
        help_text: Help text for button
    """
    if file_path and os.path.exists(file_path):
        file_manager = get_file_manager()
        download_data = file_manager.prepare_download_data(file_path)
    elif data:
        download_data = data
    else:
        st.error("No file or data provided for download")
        return
    
    return st.download_button(
        label=label,
        data=download_data,
        file_name=filename,
        mime=mime,
        help=help_text
    )