import pdfplumber
import pandas as pd
from typing import List, Dict, Optional
import re
import argparse
import os
from datetime import datetime


class BankStatementProcessor:
    def __init__(self, pdf_path: str, password: Optional[str] = None):
        """
        Initialize the processor with PDF path and optional password.

        Args:
            pdf_path: Path to the PDF file
            password: Password for encrypted PDFs
        """
        self.pdf_path = pdf_path
        self.password = password
        self.table_settings = {
            'vertical_strategy': 'lines',
            'horizontal_strategy': 'text',
            'snap_y_tolerance': 7,
            'intersection_tolerance': 5,
        }

    def extract_table_from_page(self, page) -> List[List[str]]:
        """
        Extract table data from a single page.

        Args:
            page: pdfplumber page object

        Returns:
            List of table rows
        """
        try:
            # Try with custom settings first
            table = page.extract_table(self.table_settings)
            if table:
                return table

            # Fallback to default extraction
            table = page.extract_table()
            if table:
                return table

            print(f"Warning: No table found on page {page.page_number}")
            return []

        except Exception as e:
            print(f"Error extracting table from page {page.page_number}: {e}")
            return []

    def clean_cell_data(self, cell: str) -> str:
        """
        Clean individual cell data by removing extra whitespace and newlines.

        Args:
            cell: Raw cell content

        Returns:
            Cleaned cell content
        """
        if not cell:
            return ""

        # Replace multiple whitespace/newlines with single space
        cleaned = re.sub(r'\s+', ' ', cell.strip())
        return cleaned

    def process_table_rows(self, table: List[List[str]], page_number: int = 1) -> List[Dict[str, str]]:
        """
        Process raw table rows into structured transaction data.

        Args:
            table: Raw table data from PDF
            page_number: Page number (1-indexed) to determine how many header rows to skip

        Returns:
            List of processed transaction dictionaries
        """
        if not table or len(table) < 2:
            return []

        processed_rows = []

        # Skip header rows: 1 row for first page, 2 rows for subsequent pages (includes Arabic header)
        if page_number == 1:
            ix = 1  # Skip 1 header row on first page
            print(f"  Skipping 1 header row on page {page_number}")
        else:
            ix = 2  # Skip 2 header rows on subsequent pages (regular + Arabic)
            print(
                f"  Skipping 2 header rows on page {page_number} (regular + Arabic)")

        max_ix = len(table)

        while ix < max_ix:
            row = table[ix]

            # Skip empty rows
            if not any(row):
                ix += 1
                continue

            # Clean the row data
            cleaned_row = [self.clean_cell_data(cell) for cell in row]

            # Ensure we have at least 5 columns (date, description, debits, credits, balance)
            while len(cleaned_row) < 5:
                cleaned_row.append("")

            # Skip rows with "carried forward" or "brought forward" in description
            potential_description = cleaned_row[1] if len(
                cleaned_row) > 1 else ""
            if potential_description and any(phrase in potential_description.upper() for phrase in ['CARRIED FORWARD', 'BROUGHT FORWARD']):
                print(
                    f"  Skipping balance transfer row: {potential_description[:50]}...")
                ix += 1
                continue

            date = None
            description = None
            debits = None
            credits = None
            balance = None

            row_found = False

            while not row_found and ix < max_ix:
                potential_date = cleaned_row[0] if len(cleaned_row) > 0 else ""
                potential_description = cleaned_row[1] if len(
                    cleaned_row) > 1 else ""
                potential_debits = cleaned_row[2] if len(
                    cleaned_row) > 2 else ""
                potential_credits = cleaned_row[3] if len(
                    cleaned_row) > 3 else ""
                potential_balance = cleaned_row[4] if len(
                    cleaned_row) > 4 else ""

                # Check if this is a complete row (has date and at least one amount)
                if potential_date and (potential_debits or potential_credits or potential_balance):
                    processed_rows.append({
                        'date': potential_date,
                        'description': potential_description,
                        'debits': potential_debits,
                        'credits': potential_credits,
                        'balance': potential_balance
                    })
                    ix += 1
                    row_found = True

                elif potential_date and potential_description:
                    # Start of a new multi-line row
                    date = potential_date
                    description = potential_description
                    ix += 1
                    if ix < max_ix:
                        cleaned_row = [self.clean_cell_data(
                            cell) for cell in table[ix]]
                        while len(cleaned_row) < 5:
                            cleaned_row.append("")

                elif potential_description and (potential_debits or potential_credits or potential_balance):
                    # End of a multi-line row with description continuation
                    if description:
                        description = description + ' ' + potential_description
                    else:
                        description = potential_description
                    debits = potential_debits
                    credits = potential_credits
                    balance = potential_balance

                    processed_rows.append({
                        'date': date,
                        'description': description,
                        'debits': debits,
                        'credits': credits,
                        'balance': balance
                    })
                    ix += 1
                    row_found = True

                elif (potential_debits or potential_credits or potential_balance):
                    # End of row with amounts only
                    debits = potential_debits
                    credits = potential_credits
                    balance = potential_balance

                    processed_rows.append({
                        'date': date,
                        'description': description,
                        'debits': debits,
                        'credits': credits,
                        'balance': balance
                    })
                    ix += 1
                    row_found = True

                elif potential_description:
                    # Continuation of description
                    if description:
                        description = description + ' ' + potential_description
                    else:
                        description = potential_description
                    ix += 1
                    if ix < max_ix:
                        cleaned_row = [self.clean_cell_data(
                            cell) for cell in table[ix]]
                        while len(cleaned_row) < 5:
                            cleaned_row.append("")
                else:
                    # Skip this row
                    ix += 1
                    row_found = True

        return processed_rows

    def clean_amount(self, amount_str: str) -> Optional[float]:
        """
        Clean amount string by removing text, commas and converting to float.

        Args:
            amount_str: Raw amount string

        Returns:
            Float value or None if invalid
        """
        if not amount_str or not amount_str.strip():
            return None

        # Remove all non-numeric characters except decimal points and minus signs
        cleaned = re.sub(r'[^\d.-]', '', amount_str.strip())

        if not cleaned:
            return None

        try:
            return float(cleaned)
        except ValueError:
            return None

    def format_date(self, date_str: str) -> Optional[str]:
        """
        Convert date from format like "28APR25" to "2025-04-28".

        Args:
            date_str: Date string in format like "28APR25"

        Returns:
            Formatted date string in YYYY-MM-DD format or None if invalid
        """
        if not date_str or not date_str.strip():
            return None

        date_str = date_str.strip()

        # Month abbreviations mapping
        month_map = {
            'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
            'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
            'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
        }

        # Pattern to match format like "28APR25"
        pattern = r'(\d{1,2})([A-Z]{3})(\d{2})'
        match = re.match(pattern, date_str.upper())

        if not match:
            return date_str  # Return original if doesn't match expected format

        day, month_abbr, year_short = match.groups()

        if month_abbr not in month_map:
            return date_str  # Return original if month not recognized

        # Convert 2-digit year to 4-digit (assuming 20xx)
        year = f"20{year_short}"
        month = month_map[month_abbr]

        # Validate the date
        try:
            datetime.strptime(f"{year}-{month}-{day.zfill(2)}", "%Y-%m-%d")
            return f"{year}-{month}-{day.zfill(2)}"
        except ValueError:
            return date_str  # Return original if invalid date

    def post_process_transactions(self, transactions: List[Dict[str, str]]) -> List[Dict[str, any]]:
        """
        Post-process transactions to clean amounts and format dates.

        Args:
            transactions: Raw transaction data

        Returns:
            Cleaned transaction data with proper types
        """
        processed = []

        for transaction in transactions:
            processed_transaction = {
                'date': self.format_date(transaction.get('date', '')),
                'description': transaction.get('description', '').strip(),
                'debits': self.clean_amount(transaction.get('debits', '')),
                'credits': self.clean_amount(transaction.get('credits', '')),
                'balance': self.clean_amount(transaction.get('balance', ''))
            }
            processed.append(processed_transaction)

        return processed

    def process_all_pages(self, return_only: bool = False) -> List[Dict[str, any]]:
        """
        Process all pages in the PDF and return concatenated results.

        Args:
            return_only: If True, only return data without saving to files

        Returns:
            List of all transactions from all pages
        """
        all_transactions = []

        try:
            with pdfplumber.open(self.pdf_path, password=self.password) as doc:
                if not return_only:
                    print(f"Processing PDF with {len(doc.pages)} pages...")

                for page_num, page in enumerate(doc.pages, 1):
                    if not return_only:
                        print(f"Processing page {page_num}...")

                    # Extract table from current page
                    table = self.extract_table_from_page(page)

                    if table:
                        # Process the table rows with page number info
                        transactions = self.process_table_rows(table, page_num)
                        all_transactions.extend(transactions)
                        if not return_only:
                            print(
                                f"  Found {len(transactions)} transactions on page {page_num}")
                    else:
                        if not return_only:
                            print(
                                f"  No transactions found on page {page_num}")

        except Exception as e:
            if not return_only:
                print(f"Error processing PDF: {e}")
            raise

        # Post-process all transactions to clean amounts and format dates
        if not return_only:
            print(f"Post-processing {len(all_transactions)} transactions...")
        processed_transactions = self.post_process_transactions(
            all_transactions)
        if not return_only:
            print("  - Cleaned amount columns (removed text, commas)")
            print("  - Formatted dates to YYYY-MM-DD")

        return processed_transactions

    def save_to_csv(self, transactions: List[Dict[str, any]], output_path: str):
        """
        Save transactions to CSV file.

        Args:
            transactions: List of transaction dictionaries
            output_path: Path for output CSV file
        """
        if not transactions:
            print("No transactions to save.")
            return

        df = pd.DataFrame(transactions)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(transactions)} transactions to {output_path}")

    def save_to_excel(self, transactions: List[Dict[str, any]], output_path: str):
        """
        Save transactions to Excel file.

        Args:
            transactions: List of transaction dictionaries
            output_path: Path for output Excel file
        """
        if not transactions:
            print("No transactions to save.")
            return

        df = pd.DataFrame(transactions)
        df.to_excel(output_path, index=False)
        print(f"Saved {len(transactions)} transactions to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Process bank statement PDF and extract transactions')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--password', help='Password for encrypted PDF')
    parser.add_argument('--output', '-o', help='Output file path (CSV or Excel)',
                        default='bank_transactions.csv')
    parser.add_argument('--format', choices=['csv', 'excel'], default='csv',
                        help='Output format (default: csv)')

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file '{args.pdf_path}' not found.")
        return

    # Determine output format from file extension if not specified
    if args.format == 'csv' and args.output.endswith('.xlsx'):
        args.format = 'excel'
    elif args.format == 'excel' and args.output.endswith('.csv'):
        args.format = 'csv'

    try:
        # Initialize processor
        processor = BankStatementProcessor(args.pdf_path, args.password)

        # Process all pages
        transactions = processor.process_all_pages()

        if transactions:
            # Save results
            if args.format == 'excel':
                if not args.output.endswith('.xlsx'):
                    args.output = args.output.replace('.csv', '.xlsx')
                processor.save_to_excel(transactions, args.output)
            else:
                if not args.output.endswith('.csv'):
                    args.output = args.output.replace('.xlsx', '.csv')
                processor.save_to_csv(transactions, args.output)

            print(
                f"\nProcessing complete! Total transactions extracted: {len(transactions)}")
        else:
            print("No transactions found in the PDF.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
