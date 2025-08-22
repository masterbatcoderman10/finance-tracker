#!/usr/bin/env python3
"""
Streamlit Workflow Adapter
Adapts the BankStatementWorkflow for use with Streamlit, including progress tracking
"""

import asyncio
import os
import json
from typing import Optional, Dict, Any, Callable
import pandas as pd
from datetime import datetime

# Import the existing workflow components
from workflow_orchestrator import BankStatementWorkflow
from progress_tracker import ProgressTracker, create_progress_callback
from config_manager import ProcessingConfig
from file_utils import get_file_manager
import streamlit as st


class StreamlitBankStatementWorkflow:
    """
    Streamlit-adapted version of BankStatementWorkflow with progress tracking
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4.1-mini"):
        """
        Initialize the Streamlit workflow adapter
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use for classification
        """
        self.workflow = BankStatementWorkflow(api_key, model)
        self.progress_tracker = ProgressTracker()
        self.file_manager = get_file_manager()
        
    async def process_bank_statement_with_progress(
        self,
        config: ProcessingConfig,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process bank statement with Streamlit progress tracking
        
        Args:
            config: ProcessingConfig with all processing parameters
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with processing results and file paths
        """
        try:
            # Initialize progress tracking
            self.progress_tracker.start_processing()
            
            # Use provided callback or create default one
            if progress_callback is None:
                progress_callback = create_progress_callback(self.progress_tracker)
            
            # Step 1: Extract transactions from PDF
            progress_callback(1, "Extracting transactions from PDF...")
            
            from pdf_processor import BankStatementProcessor
            processor = BankStatementProcessor(config.pdf_path, config.password)
            transactions = processor.process_all_pages(return_only=True)
            
            progress_callback(1, f"Extracted {len(transactions)} transactions from PDF")
            self.progress_tracker.add_step_info(1, {
                'transactions_found': len(transactions),
                'pdf_file': os.path.basename(config.pdf_path),
                'pages_processed': 'Multiple' if len(transactions) > 50 else 'Single'
            })
            
            if not transactions:
                raise ValueError("No transactions found in PDF!")
            
            # Step 2: Load existing keyword cache
            progress_callback(2, "Loading keyword cache...")
            
            if config.force_reclassify:
                progress_callback(2, "Force reclassify enabled - ignoring existing cache")
                cache = {'unknown': {}, 'debits': {}, 'credits': {}}
            else:
                # Use uploaded cache file if available
                cache_file_path = config.cache_file_path or "classification_keywords.json"
                cache = self.workflow.load_keyword_cache(cache_file_path)
            
            total_cached_keywords = sum(len(type_cache) for type_cache in cache.values())
            progress_callback(2, f"Loaded {total_cached_keywords} cached keywords")
            self.progress_tracker.add_step_info(2, {
                'cached_keywords': total_cached_keywords,
                'cache_source': 'uploaded' if config.cache_file_path else 'default'
            })
            
            # Step 3: Identify transactions needing classification
            progress_callback(3, "Identifying transactions needing classification...")
            
            if config.force_reclassify:
                uncached_transactions = [t for t in transactions if t.get('description', '')]
            else:
                uncached_transactions = self.workflow.identify_uncached_transactions(transactions, cache)
            
            progress_callback(3, f"Found {len(uncached_transactions)} transactions needing classification")
            self.progress_tracker.add_step_info(3, {
                'uncached_transactions': len(uncached_transactions),
                'cached_transactions': len(transactions) - len(uncached_transactions),
                'cache_hit_rate': f"{((len(transactions) - len(uncached_transactions)) / len(transactions) * 100):.1f}%" if transactions else "0%"
            })
            
            # Step 4: Classify uncached transactions
            classification_results = pd.DataFrame()
            if uncached_transactions:
                progress_callback(4, f"Classifying {len(uncached_transactions)} new transaction types...")
                
                # Create a custom progress callback for classification
                def classification_progress_callback(current, total, item_description="transactions"):
                    progress_callback(4, 
                        f"Classifying transactions: {current}/{total}",
                        f"Processing {item_description}: {current}/{total} ({current/total*100:.0f}%)",
                        current, total, item_description)
                
                # Monkey patch the classifier to use our progress callback
                original_classify = self.workflow.classifier.classify_transactions
                
                async def classify_with_progress(*args, **kwargs):
                    # Override the progress display for classification
                    return await original_classify(*args, **kwargs)
                
                classification_results = await self.workflow.classifier.classify_and_analyze(
                    transactions=uncached_transactions,
                    max_concurrent=config.max_concurrent,
                    return_only=True,
                    show_progress=False  # We'll handle progress ourselves
                )
                
                # Update cache with new classifications
                new_keywords = 0
                for _, row in classification_results.iterrows():
                    keyword = row['keyword']
                    category = row['category']
                    transaction_type = row.get('transaction_type', 'unknown')
                    
                    if transaction_type not in cache:
                        cache[transaction_type] = {}
                    
                    if keyword and keyword not in cache[transaction_type]:
                        cache[transaction_type][keyword] = category
                        new_keywords += 1
                
                progress_callback(4, f"Added {new_keywords} new keywords to cache")
                self.progress_tracker.add_step_info(4, {
                    'new_classifications': len(classification_results),
                    'new_keywords_added': new_keywords,
                    'classification_model': config.model
                })
            else:
                progress_callback(4, "All transactions already classified in cache - skipping API calls")
                self.progress_tracker.add_step_info(4, {
                    'new_classifications': 0,
                    'cache_hit': '100%'
                })
            
            # Step 5: Apply categories to all transactions
            progress_callback(5, "Applying categories to all transactions...")
            
            categorized_transactions = self.workflow.apply_categories_to_transactions(transactions, cache)
            
            categorized_count = sum(1 for t in categorized_transactions if t.get('category') != 'Uncategorized')
            uncategorized_count = len(categorized_transactions) - categorized_count
            
            progress_callback(5, f"Categorized: {categorized_count} transactions, Uncategorized: {uncategorized_count}")
            self.progress_tracker.add_step_info(5, {
                'categorized_transactions': categorized_count,
                'uncategorized_transactions': uncategorized_count,
                'categorization_rate': f"{(categorized_count / len(categorized_transactions) * 100):.1f}%" if categorized_transactions else "0%"
            })
            
            # Step 6: Clean descriptions using AI (optional)
            if config.clean_descriptions:
                progress_callback(6, "Cleaning transaction descriptions using AI...")
                
                unique_descriptions = list({t.get('description', '') for t in categorized_transactions if t.get('description', '')})
                
                def cleaning_progress_callback(current, total):
                    progress_callback(6, 
                        f"Cleaning descriptions: {current}/{total}",
                        f"Processing descriptions: {current}/{total} ({current/total*100:.0f}%)")
                
                cleaned_descriptions = await self.workflow.clean_descriptions_batch(
                    unique_descriptions, config.max_concurrent
                )
                
                # Create mapping and apply cleaned descriptions
                description_mapping = dict(zip(unique_descriptions, cleaned_descriptions))
                for transaction in categorized_transactions:
                    original_desc = transaction.get('description', '')
                    if original_desc in description_mapping:
                        transaction['cleaned_description'] = description_mapping[original_desc]
                    else:
                        transaction['cleaned_description'] = original_desc
                
                progress_callback(6, f"Cleaned {len(unique_descriptions)} unique descriptions")
                self.progress_tracker.add_step_info(6, {
                    'descriptions_cleaned': len(unique_descriptions),
                    'cleaning_model': 'gpt-4.1-nano'
                })
            else:
                progress_callback(6, "Skipping description cleaning (disabled)")
                self.progress_tracker.add_step_info(6, {'descriptions_cleaned': 0})
            
            # Step 7: Save final results
            progress_callback(7, f"Saving results to {os.path.basename(config.output_csv)}...")
            
            df = pd.DataFrame(categorized_transactions)
            
            # Reorder columns for better readability
            if config.clean_descriptions:
                column_order = ['date', 'description', 'cleaned_description',
                               'category', 'debits', 'credits', 'balance', 'transaction_type']
            else:
                column_order = ['date', 'description',
                               'category', 'debits', 'credits', 'balance', 'transaction_type']
            
            df = df[column_order]
            df.to_csv(config.output_csv, index=False)
            
            progress_callback(7, f"Saved {len(df)} categorized transactions")
            self.progress_tracker.add_step_info(7, {
                'output_file': os.path.basename(config.output_csv),
                'total_transactions_saved': len(df)
            })
            
            # Step 8: Generate Excel reports (optional)
            excel_report_path = None
            if config.generate_reports:
                progress_callback(8, "Generating Excel financial reports...")
                
                try:
                    from finance_report_generator import FinanceReportGenerator
                    report_generator = FinanceReportGenerator(config.output_csv)
                    
                    excel_report_path = config.report_filename
                    report_generator.generate_excel_report(excel_report_path)
                    
                    available_months = report_generator.get_available_months()
                    month_names = [report_generator.format_month_name(m) for m in available_months]
                    
                    progress_callback(8, f"Excel report generated with {len(available_months)} monthly sheets")
                    self.progress_tracker.add_step_info(8, {
                        'excel_report': os.path.basename(excel_report_path),
                        'monthly_sheets': len(available_months),
                        'months_covered': ', '.join(month_names)
                    })
                    
                except Exception as e:
                    progress_callback(8, f"Error generating reports: {str(e)}")
                    self.progress_tracker.add_step_info(8, {
                        'report_generation_error': str(e)
                    })
            else:
                progress_callback(8, "Skipping Excel report generation (disabled)")
                self.progress_tracker.add_step_info(8, {'excel_report_generated': False})
            
            # Step 9: Save updated cache and classification details
            progress_callback(9, "Finalizing processing results...")
            
            # Save updated cache
            with open(config.cache_output, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
            
            # Save classification details if requested
            classification_details_path = None
            if config.save_classification_details and not classification_results.empty:
                classification_details_path = config.classification_details_file
                classification_results.to_csv(classification_details_path, index=False)
            
            # Calculate category breakdown
            category_counts = df['category'].value_counts().to_dict()
            
            progress_callback(9, "Processing results finalized")
            self.progress_tracker.add_step_info(9, {
                'updated_cache_keywords': sum(len(type_cache) for type_cache in cache.values()),
                'category_breakdown': category_counts
            })
            
            # Step 10: Complete processing
            results = {
                'transactions_processed': len(df),
                'categories_found': len(category_counts),
                'categorized_transactions': categorized_count,
                'uncategorized_transactions': uncategorized_count,
                'category_breakdown': category_counts,
                'files_generated': {
                    'transactions_csv': config.output_csv,
                    'updated_cache': config.cache_output,
                    'excel_report': excel_report_path,
                    'classification_details': classification_details_path
                },
                'cache_stats': {
                    'total_keywords': sum(len(type_cache) for type_cache in cache.values()),
                    'cache_hit_rate': f"{((len(transactions) - len(uncached_transactions)) / len(transactions) * 100):.1f}%" if transactions else "0%"
                }
            }
            
            self.progress_tracker.complete_processing(results)
            progress_callback(10, "Processing completed successfully! ✅")
            
            return results
            
        except Exception as e:
            error_message = str(e)
            self.progress_tracker.handle_error(error_message)
            raise e


def run_streamlit_workflow(config: ProcessingConfig) -> Dict[str, Any]:
    """
    Run the workflow adapted for Streamlit
    
    Args:
        config: ProcessingConfig with all parameters
        
    Returns:
        Processing results dictionary
    """
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Create workflow instance
    workflow = StreamlitBankStatementWorkflow(api_key, config.model)
    
    # Initialize progress display
    workflow.progress_tracker.initialize_display()
    
    # Run the async workflow
    try:
        results = asyncio.run(workflow.process_bank_statement_with_progress(config))
        return results
    except Exception as e:
        # Make sure error is properly displayed in progress tracker
        workflow.progress_tracker.handle_error(str(e))
        raise e


# Helper function for use in main Streamlit app
def process_bank_statement_streamlit(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process bank statement from Streamlit UI configuration
    
    Args:
        config_dict: Configuration dictionary from Streamlit UI
        
    Returns:
        Processing results
    """
    from config_manager import get_config_manager
    
    # Validate and create configuration
    config_manager = get_config_manager()
    is_valid, errors, config = config_manager.validate_config(config_dict)
    
    if not is_valid:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    # Run the workflow
    return run_streamlit_workflow(config)