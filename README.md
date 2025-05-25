# Bank Statement PDF Processor

This script processes bank statement PDFs and extracts transaction data into structured CSV or Excel files. It handles multi-page PDFs, password protection, and complex table layouts where transaction data may span multiple rows.

## Features

- **Multi-page processing**: Processes all pages in a PDF document
- **Password protection**: Supports encrypted PDFs
- **Smart table extraction**: Handles complex layouts with multi-line transactions
- **Multiple output formats**: Saves to CSV or Excel
- **Robust error handling**: Gracefully handles various PDF formats and extraction issues
- **Command-line interface**: Easy to use from terminal
- **Programmatic API**: Can be imported and used in other Python scripts

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

#### Basic usage:
```bash
python pdf_processor.py path/to/your/statement.pdf
```

#### With password protection:
```bash
python pdf_processor.py path/to/your/statement.pdf --password YOUR_PASSWORD
```

#### Specify output file and format:
```bash
# Save as CSV
python pdf_processor.py statement.pdf --output transactions.csv

# Save as Excel
python pdf_processor.py statement.pdf --output transactions.xlsx --format excel
```

#### Full example:
```bash
python pdf_processor.py data/E-STATEMENT_17May25_9601.pdf --password MOHA1410 --output output/bank_transactions.xlsx --format excel
```

### Programmatic Usage

```python
from pdf_processor import BankStatementProcessor

# Initialize processor
processor = BankStatementProcessor("path/to/statement.pdf", password="YOUR_PASSWORD")

# Process all pages
transactions = processor.process_all_pages()

# Save results
processor.save_to_csv(transactions, "transactions.csv")
processor.save_to_excel(transactions, "transactions.xlsx")
```

See `example_usage.py` for a complete example.

## Output Format

The script extracts the following fields for each transaction:

- **date**: Transaction date
- **description**: Transaction description (may span multiple lines in original PDF)
- **debits**: Debit amount (if applicable)
- **credits**: Credit amount (if applicable)  
- **balance**: Account balance after transaction

## How It Works

1. **PDF Loading**: Opens the PDF with optional password protection
2. **Page Processing**: Iterates through all pages in the document
3. **Table Extraction**: Uses `pdfplumber` with custom settings to extract table data
4. **Data Cleaning**: Removes extra whitespace and normalizes cell content
5. **Row Processing**: Intelligently reconstructs transactions that span multiple table rows
6. **Data Consolidation**: Combines all transactions from all pages
7. **Output Generation**: Saves to CSV or Excel format

## Handling Complex Layouts

The script is designed to handle bank statements where:
- Transaction descriptions span multiple lines
- Some table rows contain only partial information
- Data doesn't always align perfectly in columns
- Headers and footers may interfere with table detection

## Error Handling

The script includes robust error handling for:
- Missing or corrupted PDF files
- Incorrect passwords
- Pages without extractable tables
- Malformed table data
- File I/O errors

## Requirements

- Python 3.7+
- pdfplumber >= 0.9.0
- pandas >= 1.5.0
- openpyxl >= 3.0.0 (for Excel output)

## Troubleshooting

### No tables found
If the script reports "No table found on page X", try:
- Checking if the PDF contains actual tables (not just images)
- Adjusting the table extraction settings in the `BankStatementProcessor` class
- Using a different PDF viewer to verify the content

### Incorrect extraction
If transactions are not extracted correctly:
- The PDF may have a non-standard layout
- You may need to adjust the table extraction settings
- Consider preprocessing the PDF to improve table detection

### Password issues
For encrypted PDFs:
- Ensure the password is correct
- Some PDFs may have restrictions that prevent text extraction

## Customization

You can customize the table extraction by modifying the `table_settings` in the `BankStatementProcessor` class:

```python
self.table_settings = {
    'vertical_strategy': 'lines',      # or 'text'
    'horizontal_strategy': 'text',     # or 'lines'
    'snap_y_tolerance': 7,
    'intersection_tolerance': 5,
}
```

## License

This project is open source and available under the MIT License. 