#!/usr/bin/env python3
"""
Bank Statement Processing Workflow Orchestrator
Consolidates PDF processing and transaction classification with intelligent caching
"""

from transaction_analyzer import AsyncTransactionClassifier
from pdf_processor import BankStatementProcessor
from finance_report_generator import FinanceReportGenerator
import argparse
import asyncio
import json
import os
import pandas as pd
from typing import List, Dict, Optional, Set
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI

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
        self.openai_client = AsyncOpenAI(api_key=api_key)
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
            # Skip empty keywords as they would match all descriptions
            if cached_keyword and cached_keyword in description_upper:
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

    async def clean_description(self, description: str, semaphore: asyncio.Semaphore) -> str:
        """
        Clean a single transaction description using AI

        Args:
            description: Original transaction description
            semaphore: Asyncio semaphore for rate limiting

        Returns:
            Cleaned human-readable description
        """
        async with semaphore:
            try:
                system_prompt = """You are a financial transaction description cleaner. Your task is to convert messy bank transaction descriptions into clean, human-readable descriptions.

Rules:
1. Remove technical codes, reference numbers, and card numbers
2. Extract the essential merchant/business name and transaction type
3. Keep the location if meaningful (city/country)
4. Make it concise but informative
5. Use proper capitalization
6. Remove redundant information

Examples:
- "POS-PURCHASE CARD NO. 4005-3XXX-XXXX-3917 ABRAJ AL TAAWUN HYPERMARK SHJ:AE 105.03,AED 275348 23-04-2025" → "Abraj Al Taawun Hypermarket, Sharjah"
- "DR ATM TRANSACTION CARD NO. 400536XXXXXX3917 511613056773 26-04-2025 13:04:02" → "ATM Withdrawal"
- "POS-PURCHASE CARD NO. 4005-3XXX-XXXX-3917 GOOGLE*YOUTUBEPREMIUM G.CO/HELPPAY#:US" → "YouTube Premium Subscription"
- "SDM DEPOSIT CR SDM REF.-E4011-6XXX-XXXX-8678;AB J AL TAAWUN" → "Bank Deposit"

Return only the cleaned description, nothing else."""

                response = await self.openai_client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Clean this transaction description: {description}"}
                    ],
                    max_tokens=100,
                    temperature=0.1
                )

                cleaned = response.choices[0].message.content.strip()
                return cleaned if cleaned else description

            except Exception as e:
                print(
                    f"Error cleaning description '{description[:50]}...': {e}")
                return description

    async def clean_descriptions_batch(
        self,
        descriptions: List[str],
        max_concurrent: int = 10
    ) -> List[str]:
        """
        Clean multiple transaction descriptions concurrently

        Args:
            descriptions: List of transaction descriptions to clean
            max_concurrent: Maximum number of concurrent API calls

        Returns:
            List of cleaned descriptions
        """
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        print(
            f"   Cleaning {len(descriptions)} descriptions with max {max_concurrent} concurrent requests...")

        # Create all tasks
        tasks = [
            asyncio.create_task(self.clean_description(desc, semaphore))
            for desc in descriptions
        ]

        # Execute all tasks concurrently with progress indication
        import tqdm.asyncio
        results = await tqdm.asyncio.tqdm.gather(
            *tasks,
            desc="Cleaning descriptions",
            unit="description"
        )

        return results

    async def process_bank_statement(
        self,
        pdf_path: str,
        password: Optional[str] = None,
        cache_file: str = "classification_keywords.json",
        output_file: str = "categorized_transactions.csv",
        max_concurrent: int = 8,
        force_reclassify: bool = False,
        clean_descriptions: bool = False,
        generate_reports: bool = True,
        report_filename: Optional[str] = None,
        save_classification_details: bool = False,
        classification_details_file: str = "classification_results.csv"
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
            clean_descriptions: If True, clean descriptions using AI
            generate_reports: If True, automatically generate Excel reports
            report_filename: Custom filename for Excel report (optional)
            save_classification_details: If True, save detailed classification results with reasoning and confidence
            classification_details_file: Path to save classification details file

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
        classification_results = pd.DataFrame()  # Initialize empty DataFrame
        if uncached_descriptions:
            print(
                f"\n4. Classifying {len(uncached_descriptions)} new transaction types...")

            classification_results = await self.classifier.classify_and_analyze(
                transactions=uncached_descriptions,
                max_concurrent=max_concurrent,
                return_only=True,
                show_progress=True
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

        # Step 6: Clean descriptions using AI (optional)
        if clean_descriptions:
            print(f"\n6. Cleaning transaction descriptions using AI...")
            unique_descriptions = list({t.get(
                'description', '') for t in categorized_transactions if t.get('description', '')})
            cleaned_descriptions = await self.clean_descriptions_batch(unique_descriptions, max_concurrent)

            # Create mapping from original to cleaned descriptions
            description_mapping = dict(
                zip(unique_descriptions, cleaned_descriptions))

            # Apply cleaned descriptions to all transactions
            for transaction in categorized_transactions:
                original_desc = transaction.get('description', '')
                if original_desc in description_mapping:
                    transaction['cleaned_description'] = description_mapping[original_desc]
                else:
                    transaction['cleaned_description'] = original_desc

            print(f"   Cleaned {len(unique_descriptions)} unique descriptions")
        else:
            print(f"\n6. Skipping description cleaning (disabled)")

        # Step 7: Save final results
        step_num = 7
        print(f"\n{step_num}. Saving results to {output_file}...")
        df = pd.DataFrame(categorized_transactions)

        # Reorder columns for better readability
        if clean_descriptions:
            column_order = ['date', 'description', 'cleaned_description',
                            'category', 'debits', 'credits', 'balance']
        else:
            column_order = ['date', 'description',
                            'category', 'debits', 'credits', 'balance']

        df = df[column_order]

        df.to_csv(output_file, index=False)
        print(f"   Saved {len(df)} categorized transactions")

        # Step 8: Generate Excel reports (optional)
        step_num += 1
        if generate_reports:
            print(f"\n{step_num}. Generating Excel financial reports...")
            try:
                report_generator = FinanceReportGenerator(output_file)

                # Generate custom filename if not provided
                if report_filename is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_filename = f"finance_report_{timestamp}.xlsx"

                report_generator.generate_excel_report(report_filename)
                print(f"   Excel report generated: {report_filename}")

                # Show available months summary
                available_months = report_generator.get_available_months()
                month_names = [report_generator.format_month_name(
                    m) for m in available_months]
                print(
                    f"   Report contains {len(available_months)} monthly sheets: {', '.join(month_names)}")

            except Exception as e:
                print(f"   Error generating reports: {e}")
                print(
                    "   Transaction processing completed successfully, but report generation failed")
        else:
            print(f"\n{step_num}. Skipping Excel report generation (disabled)")

        # Step 9: Generate summary
        step_num += 1
        print(f"\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Total transactions processed: {len(df)}")
        print(f"Unique keywords in cache: {len(cache)}")
        if clean_descriptions:
            print(f"Descriptions cleaned: {len(unique_descriptions)}")

        # Category breakdown
        category_counts = df['category'].value_counts()
        print(f"\nCategory breakdown:")
        for category, count in category_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")

        # Step 10: Save classification details (optional)
        step_num += 1
        if save_classification_details and not classification_results.empty:
            print(
                f"\n{step_num}. Saving classification details to {classification_details_file}...")
            classification_results.to_csv(
                classification_details_file, index=False)
            print(
                f"   Saved {len(classification_results)} classification details")
        elif save_classification_details and classification_results.empty:
            print(
                f"\n{step_num}. No new classifications to save - all transactions were already in cache")

        return df


async def main():
    """Main function with command line argument parsing"""

    parser = argparse.ArgumentParser(
        description="Process bank statement PDF, classify transactions, and generate Excel reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python workflow_orchestrator.py bank_statement.pdf
  python workflow_orchestrator.py bank_statement.pdf --clean-descriptions
  python workflow_orchestrator.py bank_statement.pdf --no-reports --output categorized_bank_data.csv
  python workflow_orchestrator.py bank_statement.pdf --force-reclassify --max-concurrent 10 --report-filename custom_report.xlsx

Note: OpenAI API key should be set in .env file as OPENAI_API_KEY=your_api_key_here
      By default, Excel reports are automatically generated after processing.
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

    parser.add_argument(
        "--clean-descriptions",
        action="store_true",
        help="Clean descriptions using AI"
    )

    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip automatic Excel report generation"
    )

    parser.add_argument(
        "--report-filename",
        help="Custom filename for Excel report (optional)"
    )

    parser.add_argument(
        "--save-classification-details",
        action="store_true",
        help="Save detailed classification results"
    )

    parser.add_argument(
        "--classification-details-file",
        default="classification_results.csv",
        help="Path to save classification details file (default: classification_results.csv)"
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
            force_reclassify=args.force_reclassify,
            clean_descriptions=args.clean_descriptions,
            generate_reports=not args.no_reports,
            report_filename=args.report_filename,
            save_classification_details=args.save_classification_details,
            classification_details_file=args.classification_details_file
        )

        return df

    except Exception as e:
        print(f"Error processing bank statement: {e}")
        return None


if __name__ == "__main__":
    # Requirements: pip install openai pandas pydantic python-dotenv tqdm pdfplumber
    asyncio.run(main())
