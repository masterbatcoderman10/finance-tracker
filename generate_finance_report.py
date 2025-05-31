#!/usr/bin/env python3
"""
Simple script to generate finance reports using the FinanceReportGenerator class
"""

from finance_report_generator import FinanceReportGenerator


def main():
    """Generate comprehensive finance report"""

    print("=" * 60)
    print("FINANCE REPORT GENERATOR")
    print("=" * 60)

    try:
        # Initialize the generator with the categorized transactions file
        generator = FinanceReportGenerator("categorized_transactions.csv")

        # Generate reports for all available months
        generator.generate_all_reports()

        print("\n" + "=" * 60)
        print("REPORT GENERATION COMPLETED")
        print("=" * 60)
        print("✓ Console analysis printed above")
        print("✓ Excel report generated with separate sheets for each month")
        print("✓ Overview sheet included with cross-month comparison")

    except FileNotFoundError:
        print("Error: categorized_transactions.csv file not found!")
        print("Please ensure the file exists in the current directory.")

    except Exception as e:
        print(f"Error generating reports: {e}")


if __name__ == "__main__":
    main()
