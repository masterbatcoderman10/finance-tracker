#!/usr/bin/env python3
"""
Finance Report Generator
Performs transaction analysis and generates monthly Excel reports with spending analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path
import calendar


class FinanceReportGenerator:
    """
    A class to analyze financial transactions and generate monthly Excel reports
    """

    def __init__(self, data_file: str = "categorized_transactions.csv"):
        """
        Initialize the Finance Report Generator

        Args:
            data_file: Path to the CSV file containing categorized transactions
        """
        self.data_file = data_file
        self.df = None
        self.load_data()

    def load_data(self) -> None:
        """Load and preprocess transaction data"""
        try:
            self.df = pd.read_csv(self.data_file)

            # Check for empty or invalid dates before conversion
            print(f"Loaded {len(self.df)} transactions from {self.data_file}")

            # Check if cleaned descriptions are available
            has_cleaned_descriptions = 'cleaned_description' in self.df.columns
            if has_cleaned_descriptions:
                print(
                    "✓ Cleaned descriptions available - will use AI-cleaned descriptions in reports")
            else:
                print("• Using original descriptions in reports")

            # Convert date column to datetime, handling errors and specifying DD/MM/YYYY format
            self.df['date'] = pd.to_datetime(
                self.df['date'], format='%d/%m/%Y', errors='coerce')

            # Check for NaT values after conversion
            nat_count = self.df['date'].isna().sum()
            if nat_count > 0:
                print(
                    f"Warning: {nat_count} transactions have invalid dates and will be excluded from monthly analysis")
                # Show some examples of problematic rows
                invalid_dates = self.df[self.df['date'].isna()]
                if not invalid_dates.empty:
                    print("Examples of rows with invalid dates:")
                    # Show both description types if available
                    if has_cleaned_descriptions:
                        display_cols = ['date', 'description',
                                        'cleaned_description']
                    else:
                        display_cols = ['date', 'description']
                    print(invalid_dates[display_cols].head(
                    ).to_string(index=False))

            # Create month-year column for grouping (only for valid dates)
            self.df['month_year'] = self.df['date'].dt.to_period('M')

            # Fill NaN values with 0 for debits and credits
            self.df['debits'] = self.df['debits'].fillna(0)
            self.df['credits'] = self.df['credits'].fillna(0)

            # Report final valid transaction count
            valid_transactions = self.df['date'].notna().sum()
            print(
                f"Successfully processed {valid_transactions} transactions with valid dates")

        except FileNotFoundError:
            print(f"Error: File {self.data_file} not found")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def get_spending_by_category(self, month_year: str) -> pd.DataFrame:
        """
        Analyze spending by category for a specific month

        Args:
            month_year: Month-year string (e.g., "2025-04")

        Returns:
            DataFrame with spending by category
        """
        # Filter by month and exclude rows with invalid dates
        month_data = self.df[(self.df['month_year'] == month_year) & (
            self.df['date'].notna())]

        if month_data.empty:
            return pd.DataFrame(columns=['Category', 'Amount (AED)', 'Transactions', 'Percentage'])

        # Group by category and sum debits (spending)
        spending_by_category = month_data.groupby('category').agg({
            'debits': 'sum',
            'description': 'count'
        }).reset_index()

        spending_by_category.columns = [
            'Category', 'Amount (AED)', 'Transactions']

        # Calculate total spending for percentage calculation
        total_spending = spending_by_category['Amount (AED)'].sum()

        if total_spending > 0:
            spending_by_category['Percentage'] = (
                spending_by_category['Amount (AED)'] / total_spending * 100
            ).round(2)
        else:
            spending_by_category['Percentage'] = 0

        # Sort by amount descending
        spending_by_category = spending_by_category.sort_values(
            'Amount (AED)', ascending=False)

        # Format amount to 2 decimal places
        spending_by_category['Amount (AED)'] = spending_by_category['Amount (AED)'].round(
            2)

        return spending_by_category

    def get_sorted_transaction_log(self, month_year: str, transaction_type: str = 'debits') -> pd.DataFrame:
        """
        Generate sorted transaction log for a specific month

        Args:
            month_year: Month-year string (e.g., "2025-04")
            transaction_type: 'debits' for spending or 'credits' for income

        Returns:
            DataFrame with sorted transaction log
        """
        # Filter by month and exclude rows with invalid dates
        month_data = self.df[(self.df['month_year'] == month_year) & (
            self.df['date'].notna())]

        # Determine which description column to use
        description_column = 'cleaned_description' if 'cleaned_description' in self.df.columns else 'description'

        # Determine column names based on transaction type
        if transaction_type == 'credits':
            amount_column = 'credits'
            display_column = 'Credits (AED)'
            columns = ['Date', 'Description', 'Category', 'Credits (AED)']
        else:  # debits
            amount_column = 'debits'
            display_column = 'Debits (AED)'
            columns = ['Date', 'Description', 'Category', 'Debits (AED)']

        if month_data.empty:
            return pd.DataFrame(columns=columns)

        # Sort by category and date
        sorted_df = month_data.sort_values(by=['category', 'date'])

        # Select only required columns
        sorted_df = sorted_df[[
            'date', description_column, 'category', amount_column]]

        # Drop rows where amount is NaN or 0
        sorted_df = sorted_df.dropna(subset=[amount_column])
        sorted_df = sorted_df[sorted_df[amount_column] > 0]

        # Rename columns for better presentation
        sorted_df.columns = columns

        # Format date and amount
        sorted_df['Date'] = sorted_df['Date'].dt.strftime('%Y-%m-%d')
        sorted_df[display_column] = sorted_df[display_column].round(2)

        return sorted_df.reset_index(drop=True)

    def get_available_months(self) -> List[str]:
        """
        Get list of available months in the data

        Returns:
            List of month-year strings
        """
        if self.df is None or self.df.empty:
            return []

        # Filter out NaT values before getting unique months
        valid_months = self.df['month_year'].dropna()
        months = valid_months.unique()
        return sorted([str(month) for month in months if pd.notna(month)])

    def generate_month_report(self, month_year: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate spending by category, income transaction log, and sorted transaction log for a specific month

        Args:
            month_year: Month-year string (e.g., "2025-04")

        Returns:
            Tuple of (spending_by_category_df, income_transaction_log_df, sorted_transaction_log_df)
        """
        spending_df = self.get_spending_by_category(month_year)
        income_log_df = self.get_sorted_transaction_log(month_year, 'credits')
        transaction_log_df = self.get_sorted_transaction_log(
            month_year, 'debits')

        return spending_df, income_log_df, transaction_log_df

    def format_month_name(self, month_year: str) -> str:
        """
        Convert month-year string to formatted name

        Args:
            month_year: Month-year string (e.g., "2025-04")

        Returns:
            Formatted month name (e.g., "April 2025")
        """
        try:
            period = pd.Period(month_year)
            return period.strftime('%B %Y')
        except:
            return month_year

    def generate_excel_report(self, output_filename: str = None) -> None:
        """
        Generate comprehensive Excel report with separate sheets for each month

        Args:
            output_filename: Output Excel filename (optional)
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"finance_report_{timestamp}.xlsx"

        available_months = self.get_available_months()

        if not available_months:
            print("No data available to generate report")
            return

        print(f"Generating Excel report for {len(available_months)} months...")

        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:

            # Generate a sheet for each month
            for month_year in available_months:
                sheet_name = self.format_month_name(month_year)
                print(f"Processing {sheet_name}...")

                spending_df, income_log_df, transaction_log_df = self.generate_month_report(
                    month_year)

                current_row = 0

                # Add title and spending by category table
                title_df1 = pd.DataFrame(
                    [[f'SPENDING BY CATEGORY - {sheet_name.upper()}']])
                title_df1.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                   index=False, header=False)
                current_row += 2  # Title + blank row

                if not spending_df.empty:
                    spending_df.to_excel(
                        writer, sheet_name=sheet_name, startrow=current_row, index=False)
                    # Data + header + 8 blank rows
                    current_row += len(spending_df) + 1 + 8
                else:
                    no_data_df = pd.DataFrame(
                        [['No spending data available for this month']])
                    no_data_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                        index=False, header=False)
                    current_row += 3 + 8  # Message + 8 blank rows

                # Add title and income transaction log table
                title_df2 = pd.DataFrame(
                    [[f'INCOME TRANSACTION LOG - {sheet_name.upper()}']])
                title_df2.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                   index=False, header=False)
                current_row += 2

                if not income_log_df.empty:
                    income_log_df.to_excel(
                        writer, sheet_name=sheet_name, startrow=current_row, index=False)
                    # Data + header + 8 blank rows
                    current_row += len(income_log_df) + 1 + 8
                else:
                    no_data_df = pd.DataFrame(
                        [['No income data available for this month']])
                    no_data_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                        index=False, header=False)
                    current_row += 3 + 8  # Message + 8 blank rows

                # Add title and sorted transaction log table
                title_df3 = pd.DataFrame(
                    [[f'SORTED TRANSACTION LOG - {sheet_name.upper()}']])
                title_df3.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                   index=False, header=False)
                current_row += 2

                if not transaction_log_df.empty:
                    transaction_log_df.to_excel(
                        writer, sheet_name=sheet_name, startrow=current_row, index=False)
                else:
                    no_data_df = pd.DataFrame(
                        [['No transaction data available for this month']])
                    no_data_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row,
                                        index=False, header=False)

        print(f"Excel report generated: {output_filename}")
        print(f"Report contains {len(available_months)} monthly sheets")

    def print_month_analysis(self, month_year: str) -> None:
        """
        Print analysis for a specific month to console

        Args:
            month_year: Month-year string (e.g., "2025-04")
        """
        month_name = self.format_month_name(month_year)
        print(f"\n{'='*60}")
        print(f"FINANCIAL ANALYSIS FOR {month_name.upper()}")
        print(f"{'='*60}")

        spending_df, income_log_df, transaction_log_df = self.generate_month_report(
            month_year)

        print(f"\nSPENDING BY CATEGORY:")
        print("-" * 40)
        if not spending_df.empty:
            print(spending_df.to_string(index=False))
        else:
            print("No spending data available for this month")

        print(f"\nINCOME TRANSACTION LOG:")
        print("-" * 40)
        if not income_log_df.empty:
            print(income_log_df.to_string(index=False))
        else:
            print("No income data available for this month")

        print(f"\nSORTED TRANSACTION LOG:")
        print("-" * 40)
        if not transaction_log_df.empty:
            print(transaction_log_df.to_string(index=False))
        else:
            print("No transaction data available for this month")

    def generate_all_reports(self) -> None:
        """Generate both console output and Excel report for all months"""
        available_months = self.get_available_months()

        if not available_months:
            print("No data available for analysis")
            return

        print(
            f"Found data for {len(available_months)} months: {', '.join([self.format_month_name(m) for m in available_months])}")

        # Print analysis for each month
        for month_year in available_months:
            self.print_month_analysis(month_year)

        # Generate Excel report
        print(f"\n{'='*60}")
        print("GENERATING EXCEL REPORT")
        print(f"{'='*60}")
        self.generate_excel_report()


def main():
    """Main function to run the finance report generator"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate financial analysis reports')
    parser.add_argument('--data-file', '-d', default='categorized_transactions.csv',
                        help='Path to the categorized transactions CSV file')
    parser.add_argument('--output', '-o', default=None,
                        help='Output Excel filename (optional)')
    parser.add_argument('--month', '-m', default=None,
                        help='Analyze specific month (format: 2025-04)')
    parser.add_argument('--console-only', '-c', action='store_true',
                        help='Only print console analysis, skip Excel generation')

    args = parser.parse_args()

    try:
        # Initialize the report generator
        generator = FinanceReportGenerator(args.data_file)

        if args.month:
            # Analyze specific month
            generator.print_month_analysis(args.month)
            if not args.console_only:
                # Generate Excel for all months but focus on the requested one
                generator.generate_excel_report(args.output)
        else:
            # Analyze all available months
            if args.console_only:
                available_months = generator.get_available_months()
                for month_year in available_months:
                    generator.print_month_analysis(month_year)
            else:
                generator.generate_all_reports()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
