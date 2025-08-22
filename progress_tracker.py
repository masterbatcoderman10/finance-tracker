#!/usr/bin/env python3
"""
Progress Tracking Utilities for Streamlit Finance Tracker
Manages real-time progress updates during bank statement processing
"""

import streamlit as st
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta


class ProgressTracker:
    """
    Manages progress tracking for the Streamlit application
    """
    
    # Processing steps with descriptions
    PROCESSING_STEPS = {
        1: "Extracting transactions from PDF",
        2: "Loading keyword cache", 
        3: "Identifying transactions needing classification",
        4: "Classifying new transaction types",
        5: "Applying categories to all transactions",
        6: "Cleaning transaction descriptions",
        7: "Saving categorized results",
        8: "Generating Excel financial reports",
        9: "Finalizing processing results",
        10: "Processing complete"
    }
    
    def __init__(self):
        self.start_time = None
        self.current_step = 0
        self.total_steps = len(self.PROCESSING_STEPS)
        self.step_start_time = None
        self.progress_container = None
        self.status_container = None
        self.details_container = None
        
    def initialize_display(self):
        """Initialize the progress display containers"""
        self.progress_container = st.container()
        with self.progress_container:
            self.progress_bar = st.progress(0, text="Initializing...")
            self.status_container = st.empty()
            self.details_container = st.empty()
        
        # Initialize session state for progress tracking
        if 'progress_data' not in st.session_state:
            st.session_state.progress_data = {
                'current_step': 0,
                'progress_percentage': 0,
                'current_message': "Initializing...",
                'step_details': {},
                'start_time': None,
                'estimated_completion': None
            }
    
    def start_processing(self):
        """Start the processing timer"""
        self.start_time = time.time()
        self.step_start_time = time.time()
        st.session_state.progress_data['start_time'] = self.start_time
        self.update_display()
    
    def update_step(
        self, 
        step_number: int, 
        message: Optional[str] = None, 
        details: Optional[str] = None,
        progress_within_step: Optional[float] = None
    ):
        """
        Update the current processing step
        
        Args:
            step_number: Current step number (1-10)
            message: Custom message (optional)
            details: Additional details to display
            progress_within_step: Progress within the current step (0.0-1.0)
        """
        self.current_step = step_number
        
        # Calculate overall progress
        base_progress = (step_number - 1) / self.total_steps
        if progress_within_step is not None:
            step_progress = progress_within_step / self.total_steps
            overall_progress = base_progress + step_progress
        else:
            overall_progress = step_number / self.total_steps
        
        # Get step message
        step_message = message or self.PROCESSING_STEPS.get(step_number, f"Step {step_number}")
        
        # Update session state
        st.session_state.progress_data.update({
            'current_step': step_number,
            'progress_percentage': min(overall_progress * 100, 100),
            'current_message': step_message,
            'step_details': details or ""
        })
        
        # Estimate completion time
        if self.start_time and step_number > 1:
            elapsed_time = time.time() - self.start_time
            estimated_total_time = elapsed_time / overall_progress
            estimated_completion = self.start_time + estimated_total_time
            st.session_state.progress_data['estimated_completion'] = estimated_completion
        
        self.update_display()
        
        # Reset step timer for next step
        self.step_start_time = time.time()
    
    def update_step_progress(
        self, 
        current: int, 
        total: int, 
        item_description: str = "items"
    ):
        """
        Update progress within the current step
        
        Args:
            current: Current item number
            total: Total items to process
            item_description: Description of what's being processed
        """
        if total > 0:
            progress_within_step = current / total
            details = f"Processing {item_description}: {current}/{total} ({progress_within_step*100:.0f}%)"
            
            self.update_step(
                self.current_step, 
                details=details,
                progress_within_step=progress_within_step
            )
    
    def update_display(self):
        """Update the Streamlit display with current progress"""
        if not hasattr(self, 'progress_bar') or self.progress_bar is None:
            return
            
        progress_data = st.session_state.progress_data
        
        # Update progress bar
        progress_percentage = progress_data['progress_percentage']
        current_message = progress_data['current_message']
        
        self.progress_bar.progress(
            progress_percentage / 100, 
            text=f"Step {progress_data['current_step']}/{self.total_steps}: {current_message}"
        )
        
        # Update status container
        with self.status_container:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Progress", 
                    f"{progress_percentage:.0f}%",
                    delta=None
                )
            
            with col2:
                if progress_data['start_time']:
                    elapsed = time.time() - progress_data['start_time']
                    elapsed_str = self._format_duration(elapsed)
                    st.metric("Elapsed Time", elapsed_str)
            
            with col3:
                if progress_data.get('estimated_completion'):
                    remaining = progress_data['estimated_completion'] - time.time()
                    if remaining > 0:
                        remaining_str = self._format_duration(remaining)
                        st.metric("Est. Remaining", remaining_str)
        
        # Update details container
        if progress_data['step_details']:
            with self.details_container:
                st.info(f"📋 {progress_data['step_details']}")
    
    def add_step_info(self, step_number: int, info: Dict[str, Any]):
        """
        Add information about a completed step
        
        Args:
            step_number: Step number
            info: Information dictionary
        """
        if 'step_info' not in st.session_state.progress_data:
            st.session_state.progress_data['step_info'] = {}
        
        st.session_state.progress_data['step_info'][step_number] = info
    
    def complete_processing(self, results: Dict[str, Any]):
        """
        Mark processing as complete and display final results
        
        Args:
            results: Final processing results
        """
        self.update_step(10, "Processing completed successfully! ✅")
        
        # Calculate total processing time
        if self.start_time:
            total_time = time.time() - self.start_time
            results['processing_time'] = total_time
            results['processing_time_formatted'] = self._format_duration(total_time)
        
        st.session_state.progress_data['results'] = results
        st.session_state.progress_data['completed'] = True
    
    def handle_error(self, error_message: str, step_number: Optional[int] = None):
        """
        Handle processing errors
        
        Args:
            error_message: Error description
            step_number: Step where error occurred (optional)
        """
        current_step = step_number or self.current_step
        
        st.session_state.progress_data.update({
            'error': True,
            'error_message': error_message,
            'error_step': current_step
        })
        
        # Update display to show error
        if hasattr(self, 'progress_bar') and self.progress_bar is not None:
            self.progress_bar.progress(
                st.session_state.progress_data['progress_percentage'] / 100,
                text=f"❌ Error in Step {current_step}: Processing failed"
            )
        
        with self.status_container:
            st.error(f"🚫 Processing failed at step {current_step}: {error_message}")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to readable string"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            seconds = int(seconds % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of the processing progress"""
        progress_data = st.session_state.get('progress_data', {})
        
        summary = {
            'current_step': progress_data.get('current_step', 0),
            'total_steps': self.total_steps,
            'progress_percentage': progress_data.get('progress_percentage', 0),
            'current_message': progress_data.get('current_message', 'Not started'),
            'is_complete': progress_data.get('completed', False),
            'has_error': progress_data.get('error', False)
        }
        
        if progress_data.get('start_time'):
            elapsed = time.time() - progress_data['start_time']
            summary['elapsed_time'] = elapsed
            summary['elapsed_time_formatted'] = self._format_duration(elapsed)
        
        return summary


# Helper functions for use in the main app
def create_progress_callback(tracker: ProgressTracker):
    """
    Create a callback function that can be passed to the workflow
    
    Args:
        tracker: ProgressTracker instance
        
    Returns:
        Callback function
    """
    def progress_callback(
        step: int, 
        message: str = None, 
        details: str = None,
        current: int = None, 
        total: int = None,
        item_description: str = "items"
    ):
        """
        Progress callback function for the workflow
        
        Args:
            step: Current step number
            message: Step message
            details: Additional details
            current: Current item in step
            total: Total items in step
            item_description: Description of items being processed
        """
        if current is not None and total is not None:
            tracker.update_step_progress(current, total, item_description)
        else:
            tracker.update_step(step, message, details)
    
    return progress_callback


def display_progress_history():
    """Display a summary of processing steps after completion"""
    progress_data = st.session_state.get('progress_data', {})
    step_info = progress_data.get('step_info', {})
    
    if not step_info:
        return
    
    st.subheader("📋 Processing Summary")
    
    for step_num in sorted(step_info.keys()):
        info = step_info[step_num]
        step_name = ProgressTracker.PROCESSING_STEPS.get(step_num, f"Step {step_num}")
        
        with st.expander(f"Step {step_num}: {step_name}", expanded=False):
            for key, value in info.items():
                if isinstance(value, (int, float)):
                    st.metric(key.replace('_', ' ').title(), value)
                else:
                    st.write(f"**{key.replace('_', ' ').title()}**: {value}")