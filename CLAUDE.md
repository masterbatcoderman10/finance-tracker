# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive bank statement processing and financial analysis system that extracts transaction data from PDF statements, classifies transactions using AI, and generates Excel reports with spending analysis.

## Core Components

### 1. PDF Processing (`pdf_processor.py`)
- **Purpose**: Extracts transaction data from bank statement PDFs
- **Key class**: `BankStatementProcessor`
- **Handles**: Multi-page PDFs, password protection, complex table layouts
- **Output**: Raw transaction data with date, description, debits, credits, balance, transaction_type

### 2. Transaction Classification (`transaction_analyzer.py`) 
- **Purpose**: AI-powered transaction categorization using OpenAI GPT models
- **Key class**: `AsyncTransactionClassifier`
- **Features**: Async processing with semaphores, structured outputs via Pydantic
- **Categories**: Grocery, Restaurants, Subscription, Petrol, Money Transfer, etc.
- **Caching**: Keyword-based caching to minimize API calls

### 3. Workflow Orchestration (`workflow_orchestrator.py`)
- **Purpose**: Master controller that combines PDF processing + classification + reporting
- **Key class**: `BankStatementWorkflow`
- **Features**: Intelligent caching, batch processing, description cleaning, progress tracking
- **Output**: Categorized transactions with optional cleaned descriptions

### 4. Report Generation (`finance_report_generator.py`)
- **Purpose**: Creates comprehensive Excel reports with monthly analysis
- **Key class**: `FinanceReportGenerator`
- **Features**: Monthly sheets, spending analysis, category summaries, charts

## Common Development Commands

### Installation
```bash
pip install -r requirements.txt
```

### Environment Setup
Create a `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
```

### Basic PDF Processing
```bash
python pdf_processor.py data/statement.pdf --password PASSWORD --output transactions.csv
```

### Full Workflow (Recommended)
```bash
python workflow_orchestrator.py data/statement.pdf --password PASSWORD
```

### Transaction Classification Only
```bash
python transaction_analyzer.py --csv-file transactions.csv --column description
```

### Common Workflow Options
- `--clean-descriptions`: Use AI to clean transaction descriptions
- `--force-reclassify`: Ignore cache and reclassify all transactions  
- `--no-reports`: Skip Excel report generation
- `--max-concurrent 10`: Adjust API concurrency

## Architecture Notes

### Data Flow
1. **PDF → Raw Transactions**: `pdf_processor.py` extracts structured data
2. **Raw → Classified**: `transaction_analyzer.py` adds categories via AI
3. **Classified → Cached**: Keyword cache prevents re-classification
4. **Cached → Reports**: `finance_report_generator.py` creates Excel analysis

### Caching Strategy
- **File**: `classification_keywords.json`
- **Structure**: Hierarchical by transaction type (debits/credits/unknown)
- **Key Benefit**: Dramatically reduces API costs on subsequent runs
- **Cache Logic**: `classify_transaction_with_cache()` in workflow orchestrator

### Transaction Types
- **debits**: Money going out (expenses, purchases, withdrawals)
- **credits**: Money coming in (salary, deposits, refunds)
- **unknown**: Fallback when type cannot be determined

### File Structure
- **Data inputs**: `data/` directory for PDF files
- **Outputs**: CSV files for transactions, JSON for keyword cache
- **Reports**: Timestamped Excel files with monthly analysis
- **Notebooks**: `analysis.ipynb`, `pdf_test.ipynb` for exploratory work

## Key Development Patterns

### Error Handling
All main components include robust error handling for:
- Missing/corrupt PDF files
- API rate limits and failures  
- Invalid date formats
- Missing required columns

### Async Processing
Transaction classification uses async patterns:
- Semaphores for rate limiting
- Batch processing with progress bars
- Concurrent API calls with proper error isolation

### Data Validation
- Pydantic models for structured AI responses
- Pandas for data manipulation and validation
- Date format handling (DD/MM/YYYY)

## Testing Files
- Use sample transactions in `_get_sample_transactions()` for testing
- `example_usage.py` demonstrates programmatic usage
- Jupyter notebooks for interactive analysis

## Configuration
- OpenAI model: Default `gpt-4.1-mini` (configurable)
- Concurrency: Default 8 concurrent requests (adjustable)
- Cache files: Auto-generated and maintained
- Output formats: CSV for data, Excel for reports