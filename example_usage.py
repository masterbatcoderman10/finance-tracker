#!/usr/bin/env python3
"""
Example usage of the BankStatementProcessor class.
This shows how to use the processor programmatically without command line arguments.
"""

from pdf_processor import BankStatementProcessor


def main():
    # Example 1: Process a password-protected PDF
    pdf_path = "data/E-STATEMENT_17May25_9601.pdf"
    password = "MOHA1410"

    try:
        # Initialize the processor
        processor = BankStatementProcessor(pdf_path, password)

        # Process all pages
        print("Starting PDF processing...")
        transactions = processor.process_all_pages()

        if transactions:
            # Save to CSV
            processor.save_to_csv(transactions, "output/bank_transactions.csv")

            # Save to Excel
            processor.save_to_excel(
                transactions, "output/bank_transactions.xlsx")

            # Print summary
            print(f"\nSummary:")
            print(f"Total transactions extracted: {len(transactions)}")

            # Show first few transactions
            print(f"\nFirst 3 transactions:")
            for i, transaction in enumerate(transactions[:3], 1):
                print(f"{i}. Date: {transaction['date']}")
                print(f"   Description: {transaction['description'][:50]}...")
                print(f"   Balance: {transaction['balance']}")
                print()

        else:
            print("No transactions found in the PDF.")

    except FileNotFoundError:
        print(f"Error: PDF file '{pdf_path}' not found.")
        print("Please make sure the file exists and the path is correct.")
    except Exception as e:
        print(f"Error processing PDF: {e}")


if __name__ == "__main__":
    main()
