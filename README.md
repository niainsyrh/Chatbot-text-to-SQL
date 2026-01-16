# Chatbot Text to SQL

A natural language to SQL chatbot application built with Streamlit, LangChain, and Ollama. This application allows you to upload CSV/Excel files, automatically convert them to SQLite tables, and query them using natural language. The chatbot generates SQL queries, executes them, and provides visualizations using intelligent chart generation.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=Python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.0+-00A67E?style=flat)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat&logo=SQLite&logoColor=white)

## Features

-  **Natural Language Queries**: Ask questions about your data in plain English
-  **Automatic Chart Generation**: Intelligent visualization with bar, line, pie, scatter, and area charts
-  **CSV/Excel Upload**: Easily import data from CSV or Excel files
-  **Dynamic Table Analysis**: Automatically detects column types (dates, numbers, categories, text)
-  **SQLite Database**: Stores your data locally in SQLite
-  **Dark/Light Theme**: Toggle between themes for better viewing
-  **Query History**: Tracks all your queries and results
-  **SQL Security**: Validates queries to prevent dangerous operations

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher**
- **Ollama** ([Download here](https://ollama.ai))
- Required Ollama model: `llama3.2:3b` (or compatible model)

### Installing Ollama and Model

1. Download and install Ollama from [ollama.ai](https://ollama.ai)
2. Start Ollama service
3. Pull the required model:
   ```bash
   ollama pull llama3.2:3b
   ```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/niainsyrh/Chatbot-text-to-SQL.git
   cd Chatbot-text-to-SQL
   ```

2. **Navigate to the project directory**:
   ```bash
   cd purgeData
   ```

3. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Create a `.env` file** (optional):
   Create a `.env` file in the `purgeData` directory with the following:
   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.2:3b
   DATABASE_URL=sqlite:///datawarehouse.db
   ```

## Usage

1. **Start Ollama** (if not already running):
   ```bash
   ollama serve
   ```

2. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

3. **Access the application**:
   Open your browser and navigate to `http://localhost:8501`

4. **Upload your data**:
   - Click on "Choose file" in the sidebar
   - Select a CSV or Excel file
   - Enter a table name
   - Choose "Create/Replace" or "Append" mode
   - Click "Import"

5. **Query your data**:
   - Select the active table from the dropdown
   - Type your question in the chat input
   - Examples:
     - "Show me the top 5 records"
     - "Which month has the most sales?"
     - "Count all records by status"
     - "What is the total revenue?"

## Project Structure

```
purgeData/
├── app.py                 # Main Streamlit application
├── sql_chain.py          # SQL generation using LangChain and Ollama
├── database.py           # Database connection and operations
├── chart_generator.py    # Chart generation and visualization
├── global_vars.py        # Configuration and environment variables
├── import_any_csv.py     # CSV import utility script
├── requirements.txt      # Python dependencies
├── datawarehouse.db      # SQLite database (created automatically)
└── READTHIS.txt          # Development notes
```

## How It Works

1. **Table Analysis**: When you upload a table, the system automatically analyzes its structure:
   - Detects date/time columns
   - Identifies numeric, categorical, and text columns
   - Extracts sample values for context

2. **SQL Generation**: When you ask a question:
   - The system builds a comprehensive prompt based on your table structure
   - Uses Ollama LLM to generate SQL queries
   - Validates the query for security (only SELECT queries allowed)

3. **Query Execution**: 
   - Executes the generated SQL on your SQLite database
   - Returns results as a pandas DataFrame

4. **Visualization**:
   - Automatically suggests appropriate chart types
   - Creates interactive visualizations using Plotly
   - Supports bar, line, pie, scatter, and area charts

5. **Natural Language Explanation**:
   - Uses LLM to generate human-readable explanations of the results
   - Provides context and insights based on the data

## Configuration

### Ollama Configuration

Default settings in `global_vars.py`:
- **Base URL**: `http://localhost:11434`
- **Model**: `llama3.2:3b`
- **Temperature**: `0.1` (for deterministic responses)
- **Timeout**: `120` seconds

You can override these by setting environment variables or modifying `.env` file.

### Database Configuration

- Default database: `datawarehouse.db` (SQLite)
- Can be changed via `DATABASE_URL` environment variable

## Example Queries

- "Show me all status types"
- "Which month has the most deletions?"
- "Count records by category"
- "What is the total amount?"
- "Show top 10 records by count"
- "How many records were created in 2024?"

## Limitations & Future Improvements

Currently, the bot:
- ✅ Uses prompt engineering for SQL generation
- ✅ Includes SQL templates in prompts to reduce hallucinations
- ❌ Does not learn from previous queries (static knowledge)

Future enhancements could include:
- Fine-tuning the model on SQL query examples
- Implementing RAG (Retrieval-Augmented Generation) for better context
- Query optimization and caching
- Support for multiple database types (PostgreSQL, MySQL, etc.)

## Troubleshooting

### Ollama Connection Issues

If you see "Ollama connection failed":
1. Ensure Ollama is running: `ollama serve`
2. Verify the model is installed: `ollama list`
3. Check the base URL in your configuration

### Import Errors

If file upload fails:
- Check file format (CSV or Excel)
- Ensure the file is not corrupted
- Verify column names are valid (no special characters)

### SQL Generation Issues

If queries are incorrect:
- Try rephrasing your question
- Be more specific about column names
- Check the table structure in the sidebar

## Dependencies

Key dependencies:
- `streamlit` - Web interface
- `langchain` - LLM orchestration
- `langchain-ollama` - Ollama integration
- `langchain-community` - Community tools
- `sqlalchemy` - Database ORM
- `pandas` - Data manipulation
- `plotly` - Interactive visualizations
- `ollama` - Ollama Python client
- `python-dotenv` - Environment variable management

See `requirements.txt` for the complete list.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available for use.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Ollama](https://ollama.ai/)
- Uses [LangChain](https://www.langchain.com/) for LLM orchestration
- Visualizations by [Plotly](https://plotly.com/)

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Note**: Make sure Ollama is running before starting the application. The chatbot requires an active connection to Ollama to generate SQL queries.

