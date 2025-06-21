#!/usr/bin/env python3
"""
Async Transaction Classification Class with OpenAI Responses API and Semaphores
Processes transaction descriptions concurrently using OpenAI Structured Outputs
"""

import argparse
import asyncio
import json
import os
import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from dotenv import load_dotenv
import tqdm.asyncio

# Load environment variables from .env file
load_dotenv()


class TransactionCategory(str, Enum):
    """Available transaction categories"""
    GROCERY = "Grocery"
    RESTAURANTS = "Restaurants"
    SUBSCRIPTION = "Subscription"
    RECURRING_PAYMENTS = "Recurring Payments"
    PETROL = "Petrol"
    MONEY_TRANSFER = "Money Transfer"
    SALARY = "Salary"
    SHOPPING = "Shopping"
    INVESTMENT = "Investment"
    DEPOSITS = "Deposits"
    WITHDRAWALS = "Withdrawals"
    SERVICE_FEES = "Service Fees"
    DONATION = "Donation"
    TRANSPORT = "Transport"
    HOSPITALS_AND_MEDICINE = "Hospitals and Medicine"


class TransactionClassification(BaseModel):
    """Pydantic model for structured transaction classification output"""
    reasoning: str = Field(
        description="Explanation of why this transaction fits the chosen category")
    keyword: str = Field(
        description="The key merchant identifier for future automatic classification")
    category: TransactionCategory = Field(
        description="The classified category")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Classification confidence score from 0.0 to 1.0")


class AsyncTransactionClassifier:
    """Async transaction classifier using OpenAI Responses API"""

    def __init__(self, api_key: str, model: str = "o4-mini"):
        """
        Initialize the transaction classifier

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use for classification
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for transaction classification"""
        return """You are a transaction classification expert. Your task is to analyze bank transaction descriptions and classify them into predefined categories.

Available Categories:
- Grocery: Supermarkets, hypermarkets, pharmacies, food stores
- Restaurants: Dining out, food delivery, takeout services
- Subscription: Recurring service payments (Netflix, Spotify, software, cloud services); services with Pro/Premium/Plus tiers
- Recurring Payments: Installment payments, buy-now-pay-later services (Tabby, Tamara)
- Petrol: Gas stations, fuel purchases
- Money Transfer: Remittances, bank transfers, wire transfers
- Salary: Income, wages, payroll deposits
- Shopping: General retail purchases, online shopping, electronics, clothing
- Investment: Stock purchases, trading platforms, investment services
- Deposits: ATM deposits, cash deposits, SDM deposits (Smart Deposit Machine)
- Withdrawals: ATM withdrawals, cash withdrawals
- Service Fees: Bank charges, transaction fees, maintenance fees, processing fees, VAT, etc.
- Donation: Charitable donations, religious contributions, fundraising, zakat
- Transport: Public transport, taxis, ride-sharing (Uber, Careem), parking fees, tolls
- Hospitals and Medicine: Medical expenses, hospital bills, doctor visits, medicines, health insurance

Important Notes:
- SDM stands for Smart Deposit Machine, which is where withdrawals and deposits occur
- Services with Pro/Premium/Plus tiers should be classified as Subscription regardless of base service type (e.g., TALABAT PRO is Subscription, not Restaurants)
- SDM transactions should be classified as either Deposits or Withdrawals based on the transaction type

For each transaction:
1. Analyze the merchant name and transaction details
2. Identify the most distinctive keyword(s) for future matching
3. Classify into the most appropriate category
4. Provide confidence score (0.0-1.0)"""

    async def classify_single_transaction(self, transaction: str, semaphore: asyncio.Semaphore) -> dict:
        """
        Classify a single transaction using semaphore for rate limiting

        Args:
            transaction: Transaction description string
            semaphore: Asyncio semaphore for rate limiting

        Returns:
            Dictionary with classification results
        """
        async with semaphore:
            try:
                response = await self.client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": transaction}
                    ],
                    text_format=TransactionClassification,
                )

                result = response.output_parsed
                return {
                    "transaction": transaction,
                    "reasoning": result.reasoning,
                    "keyword": result.keyword,
                    "category": result.category.value,
                    "confidence": result.confidence
                }

            except Exception as e:
                print(f"Error classifying transaction: {e}")
                return {
                    "transaction": transaction,
                    "reasoning": f"Error: {str(e)}",
                    "keyword": "ERROR",
                    "category": "Shopping",
                    "confidence": 0.0
                }

    async def classify_transactions(
        self,
        transactions: List[str],
        max_concurrent: int = 8,
        return_only: bool = False,
        show_progress: bool = True
    ) -> List[dict]:
        """
        Classify multiple transactions with semaphore-controlled concurrency

        Args:
            transactions: List of transaction description strings
            max_concurrent: Maximum number of concurrent API calls
            return_only: If True, suppress progress output and return data only
            show_progress: If True, show progress bar during classification

        Returns:
            List of classification result dictionaries
        """
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        if not return_only:
            print(
                f"Processing {len(transactions)} transactions with max {max_concurrent} concurrent requests...")

        # Create all tasks explicitly using asyncio.create_task
        tasks = [
            asyncio.create_task(
                self.classify_single_transaction(transaction, semaphore))
            for transaction in transactions
        ]

        # Execute all tasks concurrently with progress bar using tqdm.asyncio.tqdm.gather
        if show_progress:
            # With progress bar
            results = await tqdm.asyncio.tqdm.gather(
                *tasks,
                desc="Classifying transactions",
                unit="transaction"
            )
        else:
            # Silent execution without progress bar
            results = await asyncio.gather(*tasks)

        return results

    def save_results_to_csv(
        self,
        results: List[dict],
        filename: str = "classified_transactions.csv"
    ) -> pd.DataFrame:
        """
        Convert results to DataFrame and save as CSV

        Args:
            results: List of classification result dictionaries
            filename: Output CSV filename

        Returns:
            DataFrame with results
        """
        df = pd.DataFrame(results)

        # Reorder columns for better readability
        column_order = ["transaction", "category",
                        "keyword", "confidence", "reasoning"]
        df = df[column_order]

        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

        return df

    def load_transactions_from_file(self, filename: str) -> List[str]:
        """
        Load transactions from a text file (one per line)

        Args:
            filename: Input file path

        Returns:
            List of transaction strings
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                transactions = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(transactions)} transactions from {filename}")
            return transactions
        except FileNotFoundError:
            print(f"File {filename} not found.")
            return []

    def load_transactions_from_csv(self, filename: str, column_name: str) -> List[str]:
        """
        Load unique transaction descriptions from a CSV file

        Args:
            filename: CSV file path
            column_name: Name of the column containing transaction descriptions

        Returns:
            List of unique transaction strings
        """
        try:
            df = pd.read_csv(filename)

            if column_name not in df.columns:
                raise ValueError(
                    f"Column '{column_name}' not found in CSV. Available columns: {list(df.columns)}")

            # Get unique non-null transaction descriptions
            transactions = df[column_name].dropna().unique().tolist()

            print(
                f"Loaded {len(transactions)} unique transactions from column '{column_name}' in {filename}")
            return transactions

        except FileNotFoundError:
            print(f"CSV file {filename} not found.")
            return []
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return []

    def analyze_results(self, df: pd.DataFrame) -> dict:
        """
        Analyze classification results and return summary statistics

        Args:
            df: DataFrame with classification results

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "total_transactions": len(df),
            "average_confidence": df['confidence'].mean(),
            "category_distribution": df['category'].value_counts().to_dict(),
            "confidence_analysis": {
                "high_confidence": len(df[df['confidence'] >= 0.9]),
                "medium_confidence": len(df[(df['confidence'] >= 0.7) & (df['confidence'] < 0.9)]),
                "low_confidence": len(df[df['confidence'] < 0.7])
            },
            "low_confidence_transactions": df[df['confidence'] < 0.7][['category', 'keyword', 'confidence']].to_dict('records')
        }

        return analysis

    def print_analysis(self, analysis: dict):
        """
        Print formatted analysis results

        Args:
            analysis: Analysis dictionary from analyze_results()
        """
        print(f"\n" + "="*50)
        print("CLASSIFICATION SUMMARY")
        print("="*50)
        print(f"Total transactions: {analysis['total_transactions']}")
        print(f"Average confidence: {analysis['average_confidence']:.3f}")

        print(f"\nCategory distribution:")
        for category, count in analysis['category_distribution'].items():
            print(f"  {category}: {count}")

        conf_analysis = analysis['confidence_analysis']
        print(f"\nConfidence analysis:")
        print(f"  High confidence (≥0.9): {conf_analysis['high_confidence']}")
        print(
            f"  Medium confidence (0.7-0.9): {conf_analysis['medium_confidence']}")
        print(f"  Low confidence (<0.7): {conf_analysis['low_confidence']}")

        if conf_analysis['low_confidence'] > 0:
            print(f"\nLow confidence transactions:")
            for tx in analysis['low_confidence_transactions']:
                print(
                    f"  {tx['category']} - {tx['keyword']} ({tx['confidence']:.2f})")

    def export_keywords(self, df: pd.DataFrame, filename: str = "classification_keywords.json"):
        """
        Export keywords mapped to their categories as JSON

        Args:
            df: DataFrame with classification results
            filename: Output filename for keywords
        """
        # Create keyword-to-category mapping
        keyword_to_category = {}
        for _, row in df.iterrows():
            keyword_to_category[row['keyword']] = row['category']

        # Save as JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(keyword_to_category, f, indent=2, ensure_ascii=False)

        print(f"Keywords exported to '{filename}'")
        print(f"Total unique keywords: {len(keyword_to_category)}")

    async def classify_and_analyze(
        self,
        transactions: Optional[List[str]] = None,
        input_file: Optional[str] = "transactions.txt",
        csv_file: Optional[str] = None,
        csv_column: Optional[str] = None,
        output_file: str = "classified_transactions.csv",
        keywords_file: str = "classification_keywords.json",
        max_concurrent: int = 8,
        return_only: bool = False,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Complete workflow: classify transactions and generate analysis

        Args:
            transactions: List of transactions (if None, loads from file)
            input_file: Input text file path for transactions
            csv_file: Input CSV file path for transactions
            csv_column: Column name in CSV file containing transactions
            output_file: Output CSV file for results
            keywords_file: Output JSON file for keywords
            max_concurrent: Maximum concurrent API calls
            return_only: If True, skip file saving and verbose output
            show_progress: If True, show progress bar during classification

        Returns:
            DataFrame with classification results
        """
        # Load transactions
        if transactions is None:
            if csv_file and csv_column:
                transactions = self.load_transactions_from_csv(
                    csv_file, csv_column)
            else:
                transactions = self.load_transactions_from_file(input_file)

            if not transactions:
                # Use sample data if no file found
                transactions = self._get_sample_transactions()

        # Classify transactions
        results = await self.classify_transactions(transactions, max_concurrent, return_only, show_progress)

        # Convert to DataFrame
        df = pd.DataFrame(results)
        column_order = ["transaction", "category",
                        "keyword", "confidence", "reasoning"]
        df = df[column_order]

        if not return_only:
            # Save to CSV
            df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")

            # Analyze and print results
            analysis = self.analyze_results(df)
            self.print_analysis(analysis)

            # Export keywords
            self.export_keywords(df, keywords_file)

        return df

    def _get_sample_transactions(self) -> List[str]:
        """Get sample transactions for testing"""
        return [
            "POS-PURCHASE CARD NO. 4005-3XXX-XXXX-3917 ABRAJ AL TAAWUN HYPERMARK SHJ:AE 105.03,AED 275348 23-04-2025 VALUE DATE:23-04-2025",
            "POS-PURCHASE CARD NO. 4005-3XXX-XXXX-3917 TALABAT.COM DUBAI:AE 58.35,AED 005125 24-04-2025 VALUE DATE:24-04-2025",
            "POS-PURCHASE CARD NO. 4005-3XXX-XXXX-3917 TABBY FZ LLC DUBAI:AE 30.49,AED 010887 09-05-2025 VALUE DATE:09-05-2025",
            "POS-PURCHASE CARD NO. 4005-3XXX-XXXX-3917 WWW.GETSTAKE.COM DUBAI:AE 1000.00,AED 258382 29-04-2025 VALUE DATE:29-04-2025",
            "DR ATM TRANSACTION CARD NO. 400536XXXXXX3917 511613056773 26-04-2025 13:04:02 ABRAJ AL TAAWUN, SHARJAH DUBAI AE E4011694 556932",
            "POS-PURCHASE CARD NO. 4005-3XXX-XXXX-3917 GOOGLE*YOUTUBEPREMIUM G.CO/HELPPAY#:US 48.99,AED 586818 19-04-2025 VALUE DATE:19-04-2025",
            "POS-PURCHASE CARD NO. 4005-3XXX-XXXX-3917 EMARAT 1390 AL GARHOUD DUBAI:AE 50.00,AED 302014 16-04-2025 VALUE DATE:16-04-2025",
            "SDM DEPOSIT CR SDM REF.-E4011-6XXX-XXXX-8678;AB J AL TAAWUN",
            "IPP CUSTOMER CREDIT IPP REF 20250426WIO6B98111109345684 LN39116482542941 FIN INVEST BUYING AND SELLING REAL ALI SALARY FOR APR25",
            "DIRECTREMIT LIV REF: EPHCOL13106JWOPM MTI CARS"
        ]


async def main():
    """Main function with command line argument parsing"""

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Classify bank transactions using OpenAI API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transaction_analyzer.py --csv-file bank_data.csv --column description
  python transaction_analyzer.py --csv-file transactions.csv --column transaction_desc --max-concurrent 10

Note: OpenAI API key should be set in .env file as OPENAI_API_KEY=your_api_key_here
        """
    )

    # Required arguments
    parser.add_argument(
        "--csv-file",
        required=True,
        help="Path to the input CSV file containing transactions"
    )

    parser.add_argument(
        "--column",
        required=True,
        help="Name of the column containing transaction descriptions"
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        default="classified_transactions.csv",
        help="Output CSV file for classified transactions (default: classified_transactions.csv)"
    )

    parser.add_argument(
        "--keywords",
        default="classification_keywords.json",
        help="Output JSON file for classification keywords (default: classification_keywords.json)"
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

    # Parse arguments
    args = parser.parse_args()

    # Get API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with: OPENAI_API_KEY=your_api_key_here")
        return None

    # Initialize classifier with parsed arguments
    classifier = AsyncTransactionClassifier(
        api_key=api_key,
        model=args.model
    )

    print(f"Loading transactions from: {args.csv_file}")
    print(f"Using column: {args.column}")
    print(f"Maximum concurrent requests: {args.max_concurrent}")
    print(f"Output file: {args.output}")
    print(f"Keywords file: {args.keywords}")
    print("-" * 50)

    # Classify transactions using the provided arguments
    df = await classifier.classify_and_analyze(
        csv_file=args.csv_file,
        csv_column=args.column,
        output_file=args.output,
        keywords_file=args.keywords,
        max_concurrent=args.max_concurrent
    )

    return df


if __name__ == "__main__":
    # Requirements: pip install openai pandas pydantic python-dotenv tqdm
    asyncio.run(main())
