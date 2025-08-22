# Finance Tracker - Streamlit Web Application

A user-friendly web interface for processing bank statements and generating financial reports using AI-powered transaction classification.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.7+** installed on your system
2. **OpenAI API Key** - Get one from [OpenAI Platform](https://platform.openai.com/account/api-keys)

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file in the project directory:
   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```

### Running the Application

1. **Start the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload your bank statement PDF** and configure processing options

4. **Click "Process Bank Statement"** and monitor real-time progress

5. **Download your results** - Excel reports, categorized CSV, and updated cache

## 📋 Features

### File Upload
- **PDF Bank Statements**: Drag-and-drop upload with validation
- **Classification Cache**: Upload existing cache to speed up processing
- **Password Support**: Handle encrypted PDF files

### Processing Options
- **Clean Descriptions**: AI-powered description cleaning for better readability
- **Force Reclassify**: Ignore cache and reclassify all transactions
- **Generate Reports**: Automatic Excel report generation with charts
- **Save Classification Details**: Export detailed AI reasoning and confidence scores

### Advanced Settings
- **Concurrency Control**: Adjust parallel API requests (1-20)
- **Model Selection**: Choose from available OpenAI models
- **Custom Report Names**: Specify custom output filenames

### Real-time Progress Tracking
- **10-Step Processing Pipeline**: Visual progress with detailed status
- **Time Estimates**: Elapsed and estimated remaining time
- **Step-by-Step Details**: Comprehensive progress information
- **Error Handling**: Clear error messages with troubleshooting tips

### Download Options
- **Excel Financial Report**: Comprehensive analysis with monthly breakdowns
- **Categorized Transactions CSV**: All transactions with AI-assigned categories
- **Updated Classification Cache**: JSON file for faster future processing
- **Classification Details**: Detailed AI reasoning (optional)

## 🏗️ Application Architecture

### Core Components

1. **`streamlit_app.py`**: Main application interface
   - File upload and configuration
   - Progress tracking display
   - Results visualization and downloads

2. **`streamlit_workflow.py`**: Workflow adapter
   - Integrates existing workflow with Streamlit
   - Real-time progress updates
   - Async processing management

3. **`file_utils.py`**: File management
   - Upload/download handling
   - File validation and security
   - Temporary file management

4. **`progress_tracker.py`**: Progress tracking
   - Real-time progress updates
   - Time estimation and step tracking
   - Error handling and recovery

5. **`config_manager.py`**: Configuration management
   - Input validation and sanitization
   - Environment checking
   - Default value management

### Integration with Existing Components

The Streamlit app reuses all existing business logic:
- **`workflow_orchestrator.py`**: Core processing workflow
- **`pdf_processor.py`**: PDF extraction logic
- **`transaction_analyzer.py`**: AI classification
- **`finance_report_generator.py`**: Excel report generation

## 📊 Processing Workflow

### Step-by-Step Process

1. **PDF Extraction**: Extract transactions from uploaded PDF
2. **Cache Loading**: Load existing classification keywords
3. **Transaction Analysis**: Identify transactions needing classification
4. **AI Classification**: Classify new transactions using OpenAI
5. **Category Application**: Apply categories to all transactions
6. **Description Cleaning**: Clean transaction descriptions (optional)
7. **Results Saving**: Save categorized transactions to CSV
8. **Report Generation**: Generate Excel reports with analysis
9. **Cache Updates**: Save updated classification cache
10. **Completion**: Finalize and prepare downloads

### AI-Powered Features

- **Transaction Classification**: Automatically categorize transactions
- **Description Cleaning**: Clean messy bank descriptions
- **Keyword Caching**: Learn from previous classifications
- **Confidence Scoring**: Assess classification reliability

## 🔧 Configuration Options

### Processing Options

| Option | Description | Default |
|--------|-------------|---------|
| Clean Descriptions | Use AI to clean transaction descriptions | ✅ Enabled |
| Force Reclassify | Ignore cache and reclassify all | ❌ Disabled |
| Generate Reports | Create Excel reports automatically | ✅ Enabled |
| Save Classification Details | Export detailed AI results | ❌ Disabled |

### Advanced Settings

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Max Concurrent Requests | 1-20 | 8 | Parallel API calls |
| OpenAI Model | Various | gpt-4.1-mini | Classification model |
| Custom Report Name | Text | Auto-generated | Custom filename |

## 🛡️ Security & Privacy

### Data Handling
- **Temporary Storage**: Files stored temporarily during processing
- **Automatic Cleanup**: Temporary files cleaned up after processing
- **No Persistent Storage**: No data stored permanently on server
- **Local Processing**: All processing happens locally

### API Key Security
- **Environment Variables**: API key stored in `.env` file
- **No Transmission**: API key never transmitted in URLs or logs
- **Validation**: API key validated before processing

## 🔍 Troubleshooting

### Common Issues

1. **"OpenAI API Key Required"**
   - Ensure `.env` file exists with valid API key
   - Check API key has sufficient credits

2. **"PDF Validation Failed"**
   - Ensure file is a valid PDF with extractable text
   - Check PDF is not corrupted or image-only

3. **"Processing Failed"**
   - Check internet connection for API calls
   - Verify PDF password is correct
   - Try reducing concurrent requests

4. **"Cache File Invalid"**
   - Ensure JSON file is properly formatted
   - Check file is not corrupted

### Error Recovery
- **Retry Mechanism**: Use "Try Again" button for temporary failures
- **Progress Preservation**: Processing state maintained during errors
- **Clear Error Messages**: Specific guidance for each error type

## 📈 Performance Tips

### Optimization Strategies

1. **Use Classification Cache**: Upload previous cache files to reduce API calls
2. **Adjust Concurrency**: Increase for faster processing, decrease for stability
3. **Process in Batches**: For very large files, consider splitting
4. **Monitor API Usage**: Check OpenAI usage dashboard

### Expected Processing Times

| Transactions | Time (Est.) | Notes |
|-------------|-------------|-------|
| 50-100 | 1-2 minutes | First run without cache |
| 100-500 | 2-5 minutes | Depends on new transactions |
| 500+ | 5-15 minutes | Consider using cache |

## 🆘 Support

### Getting Help

1. **Check Error Messages**: Most issues have specific solutions
2. **Troubleshooting Section**: Review common issues above
3. **Log Files**: Check console for detailed error information
4. **GitHub Issues**: Report bugs or feature requests

### System Requirements

- **Python**: 3.7 or higher
- **Memory**: 2GB+ recommended for large files
- **Internet**: Required for AI classification
- **Browser**: Modern browser with JavaScript enabled

## 🔄 Updates and Maintenance

### Keeping Up to Date

1. **Pull Latest Changes**: `git pull origin main`
2. **Update Dependencies**: `pip install -r requirements.txt --upgrade`
3. **Check CLAUDE.md**: Review any workflow changes

### Cache Management

- **Regular Backups**: Download and save classification cache
- **Cache Migration**: App handles format updates automatically
- **Performance Monitoring**: Check cache hit rates in results

---

## 🎯 Next Steps

After successful setup:

1. **Test with Sample PDF**: Use a small bank statement first
2. **Build Classification Cache**: Process multiple statements to build cache
3. **Customize Categories**: Adjust classification as needed
4. **Automate Workflows**: Consider scheduling regular processing

Enjoy automated financial analysis with the Finance Tracker Streamlit app! 🎉