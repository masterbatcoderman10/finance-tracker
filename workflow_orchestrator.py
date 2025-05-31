#!/usr/bin/env python3
"""
Bank Statement Processing Workflow Orchestrator
Consolidates PDF processing and transaction classification with intelligent caching
"""

from transaction_analyzer import AsyncTransactionClassifier
from pdf_processor import BankStatementProcessor
import argparse
import asyncio
import json
import os
import pandas as pd
from typing import List, Dict, Optional, Set
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class BankStatementWorkflow:
    """
    Master workflow orchestrator for bank statement processing and classification
    """

    def __init__(self, api_key: str, model: str = "gpt-4.1-mini"):
        """
        Initialize the workflow orchestrator

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use for classification
        """
        self.classifier = AsyncTransactionClassifier(api_key, model)
        self.keyword_cache = {}

    def load_keyword_cache(self, cache_file: str) -> Dict[str, str]:
        """
        Load existing keyword cache from JSON file

        Args:
            cache_file: Path to keyword cache JSON file

        Returns:
            Dictionary mapping keywords to categories
        """
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                print(f"Loaded {len(cache)} cached keywords from {cache_file}")
                return cache
            except Exception as e:
                print(f"Warning: Could not load cache file {cache_file}: {e}")
        else:
            print(
                f"Cache file {cache_file} not found, starting with empty cache")

        return {}

    def save_keyword_cache(self, cache: Dict[str, str], cache_file: str):
        """
        Save keyword cache to JSON file

        Args:
            cache: Dictionary mapping keywords to categories
            cache_file: Path to save cache file
        """
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        print(
            f"Updated cache with {len(cache)} keywords saved to {cache_file}")

    def classify_transaction_with_cache(self, description: str, cache: Dict[str, str]) -> Optional[str]:
        """
        Try to classify transaction using cached keywords

        Args:
            description: Transaction description
            cache: Keyword to category mapping

        Returns:
            Category if found in cache, None otherwise
        """
        if not description:
            return None

        # Convert description to uppercase for case-insensitive matching
        description_upper = description.upper()

        # Check if any cached keyword is in the description
        for cached_keyword, category in cache.items():
            if cached_keyword in description_upper:
                return category

        return None

    def identify_uncached_transactions(self, transactions: List[Dict], cache: Dict[str, str]) -> List[str]:
        """
        Identify transactions that need classification (not in cache)

        Args:
            transactions: List of transaction dictionaries
            cache: Keyword to category mapping

        Returns:
            List of transaction descriptions needing classification
        """
        uncached_descriptions = []

        for transaction in transactions:
            description = transaction.get('description', '')
            if description and not self.classify_transaction_with_cache(description, cache):
                uncached_descriptions.append(description)

        # Remove duplicates while preserving order
        seen = set()
        unique_uncached = []
        for desc in uncached_descriptions:
            if desc not in seen:
                seen.add(desc)
                unique_uncached.append(desc)

        return unique_uncached

    def apply_categories_to_transactions(self, transactions: List[Dict], cache: Dict[str, str]) -> List[Dict]:
        """
        Apply categories to all transactions using cache

        Args:
            transactions: List of transaction dictionaries
            cache: Keyword to category mapping

        Returns:
            List of transactions with category column added
        """
        categorized_transactions = []

        for transaction in transactions:
            categorized_transaction = transaction.copy()
            description = transaction.get('description', '')

            # Try to get category from cache
            category = self.classify_transaction_with_cache(description, cache)
            categorized_transaction['category'] = category if category else 'Uncategorized'

            categorized_transactions.append(categorized_transaction)

        return categorized_transactions

    async def process_bank_statement(
        self,
        pdf_path: str,
        password: Optional[str] = None,
        cache_file: str = "classification_keywords.json",
        output_file: str = "categorized_transactions.csv",
        max_concurrent: int = 8,
        force_reclassify: bool = False
    ) -> pd.DataFrame:
        """
        Complete workflow: process PDF, classify transactions, and categorize

        Args:
            pdf_path: Path to bank statement PDF
            password: PDF password if encrypted
            cache_file: Path to keyword cache file
            output_file: Output file for categorized transactions
            max_concurrent: Maximum concurrent API calls for classification
            force_reclassify: If True, ignore cache and reclassify all transactions

        Returns:
            DataFrame with categorized transactions
        """
        print("="*60)
        print("BANK STATEMENT PROCESSING WORKFLOW")
        print("="*60)

        # Step 1: Extract transactions from PDF
        print("\n1. Extracting transactions from PDF...")
        processor = BankStatementProcessor(pdf_path, password)
        transactions = processor.process_all_pages(return_only=True)
        print(f"   Extracted {len(transactions)} transactions")

        if not transactions:
            print("No transactions found in PDF!")
            return pd.DataFrame()

        # Step 2: Load existing keyword cache
        print(f"\n2. Loading keyword cache from {cache_file}...")
        if force_reclassify:
            print("   Force reclassify enabled - ignoring existing cache")
            cache = {}
        else:
            cache = self.load_keyword_cache(cache_file)

        # Step 3: Identify transactions needing classification
        print("\n3. Identifying transactions needing classification...")
        if force_reclassify:
            unique_descriptions = list(
                {t.get('description', '') for t in transactions if t.get('description', '')})
            uncached_descriptions = unique_descriptions
        else:
            uncached_descriptions = self.identify_uncached_transactions(
                transactions, cache)

        print(
            f"   Found {len(uncached_descriptions)} unique descriptions needing classification")

        # Step 4: Classify uncached transactions
        if uncached_descriptions:
            print(
                f"\n4. Classifying {len(uncached_descriptions)} new transaction types...")

            classification_results = await self.classifier.classify_and_analyze(
                transactions=uncached_descriptions,
                max_concurrent=max_concurrent,
                return_only=True
            )

            # Update cache with new classifications
            new_keywords = 0
            for _, row in classification_results.iterrows():
                keyword = row['keyword']
                category = row['category']
                if keyword and keyword not in cache:
                    cache[keyword] = category
                    new_keywords += 1

            print(f"   Added {new_keywords} new keywords to cache")

            # Save updated cache
            self.save_keyword_cache(cache, cache_file)
        else:
            print(
                "\n4. All transactions already classified in cache - skipping API calls")

        # Step 5: Apply categories to all transactions
        print("\n5. Applying categories to all transactions...")
        categorized_transactions = self.apply_categories_to_transactions(
            transactions, cache)

        # Count categorization results
        categorized_count = sum(1 for t in categorized_transactions if t.get(
            'category') != 'Uncategorized')
        uncategorized_count = len(categorized_transactions) - categorized_count

        print(f"   Categorized: {categorized_count} transactions")
        print(f"   Uncategorized: {uncategorized_count} transactions")

        # Step 6: Save final results
        print(f"\n6. Saving results to {output_file}...")
        df = pd.DataFrame(categorized_transactions)

        # Reorder columns for better readability
        column_order = ['date', 'description',
                        'category', 'debits', 'credits', 'balance']
        df = df[column_order]

        df.to_csv(output_file, index=False)
        print(f"   Saved {len(df)} categorized transactions")

        # Step 7: Generate summary
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Total transactions processed: {len(df)}")
        print(f"Unique keywords in cache: {len(cache)}")

        # Category breakdown
        category_counts = df['category'].value_counts()
        print(f"\nCategory breakdown:")
        for category, count in category_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")

        return df


async def main():
    """Main function with command line argument parsing"""

    parser = argparse.ArgumentParser(
        description="Process bank statement PDF and classify transactions with caching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python workflow_orchestrator.py bank_statement.pdf
  python workflow_orchestrator.py bank_statement.pdf --password mypassword --output categorized_bank_data.csv
  python workflow_orchestrator.py bank_statement.pdf --force-reclassify --max-concurrent 10

Note: OpenAI API key should be set in .env file as OPENAI_API_KEY=your_api_key_here
        """
    )

    # Required arguments
    parser.add_argument(
        "pdf_path",
        help="Path to the bank statement PDF file"
    )

    # Optional arguments
    parser.add_argument(
        "--password",
        help="Password for encrypted PDF files"
    )

    parser.add_argument(
        "--cache",
        default="classification_keywords.json",
        help="Path to keyword cache file (default: classification_keywords.json)"
    )

    parser.add_argument(
        "--output",
        default="categorized_transactions.csv",
        help="Output CSV file for categorized transactions (default: categorized_transactions.csv)"
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=8,
        help="Maximum number of concurrent API requests (default: 8)"
    )

    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model to use for classification (default: gpt-4.1-mini)"
    )

    parser.add_argument(
        "--force-reclassify",
        action="store_true",
        help="Ignore cache and reclassify all transactions"
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate PDF file exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file '{args.pdf_path}' not found.")
        return None

    # Get API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with: OPENAI_API_KEY=your_api_key_here")
        return None

    # Initialize workflow
    workflow = BankStatementWorkflow(api_key, args.model)

    # Process bank statement
    try:
        df = await workflow.process_bank_statement(
            pdf_path=args.pdf_path,
            password=args.password,
            cache_file=args.cache,
            output_file=args.output,
            max_concurrent=args.max_concurrent,
            force_reclassify=args.force_reclassify
        )

        return df

    except Exception as e:
        print(f"Error processing bank statement: {e}")
        return None


if __name__ == "__main__":
    # Requirements: pip install openai pandas pydantic python-dotenv tqdm pdfplumber
    asyncio.run(main())
