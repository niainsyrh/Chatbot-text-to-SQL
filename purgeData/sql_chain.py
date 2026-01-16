import logging
import requests
from typing import Dict, List
import pandas as pd
import re

from langchain_community.utilities import SQLDatabase
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from global_vars import OLLAMA_CONFIG, DATABASE_URL

logger = logging.getLogger(__name__)


class SQLChainManager:
    """Dynamic SQL generator that adapts to ANY table structure."""
    
    def __init__(self):
        self.llm = None
        self.db = None
        self.output_parser = StrOutputParser()
        self.table_metadata_cache = {}
        self._setup_chain()

    def _check_ollama_connection(self) -> None:
        """Verify Ollama server + model."""
        try:
            resp = requests.get(
                f"{OLLAMA_CONFIG['base_url']}/api/tags", timeout=5
            )
            resp.raise_for_status()
            models = resp.json().get("models", [])
            names = [m["name"] for m in models]
            if OLLAMA_CONFIG["model"] not in names:
                raise RuntimeError(
                    f"Model {OLLAMA_CONFIG['model']} not found in Ollama."
                )
        except Exception as e:
            logger.error("Ollama connection failed: %s", e)
            raise RuntimeError(
                f"Please ensure Ollama is running and model "
                f"{OLLAMA_CONFIG['model']} is available"
            )

    def _setup_chain(self) -> None:
        """Initialise LLM + SQL database connection."""
        self._check_ollama_connection()

        self.llm = OllamaLLM(
            model=OLLAMA_CONFIG["model"],
            base_url=OLLAMA_CONFIG["base_url"],
            temperature=OLLAMA_CONFIG["temperature"],
            timeout=OLLAMA_CONFIG["timeout"],
        )

        self.db = SQLDatabase.from_uri(DATABASE_URL)
        logger.info("SQLChainManager initialised with dynamic analysis.")

    def _analyze_table_structure(self, table_name: str) -> Dict:
        """
        Dynamically analyze ANY table structure.
        Returns column categorization + sample values.
        """
        try:
            # Get sample data (100 rows for analysis)
            query = f"SELECT * FROM {table_name} LIMIT 100"
            df = pd.read_sql(query, self.db._engine)
            
            metadata = {
                'columns': list(df.columns),
                'date_columns': [],
                'numeric_columns': [],
                'categorical_columns': [],
                'text_columns': [],
                'column_samples': {},
                'column_stats': {},
                'date_formats': {}  # Track detected date formats
            }
            
            for col in df.columns:
                dtype = df[col].dtype
                sample_values = df[col].dropna().head(5).tolist()
                unique_count = df[col].nunique()
                total_count = len(df)
                
                metadata['column_samples'][col] = sample_values
                metadata['column_stats'][col] = {
                    'unique': unique_count,
                    'total': total_count,
                    'cardinality': unique_count / total_count if total_count > 0 else 0
                }
                
                col_lower = col.lower()
                
                # Smart categorization - GENERIC for ANY table
                # Check for date/time columns
                is_date = False
                if any(kw in col_lower for kw in ['date', 'time', 'created', 'updated', 
                                                    'expired', 'purged', 'deleted', 'modified',
                                                    'timestamp', 'at']):
                    is_date = True
                elif dtype == 'object':
                    # Check if column contains date-like patterns
                    sample_str = ' '.join(str(v) for v in sample_values if v)
                    if re.search(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{4}/\d{2}/\d{2}', sample_str):
                        is_date = True
                
                if is_date:
                    metadata['date_columns'].append(col)
                    # Detect date format from samples
                    if sample_values:
                        metadata['date_formats'][col] = self._detect_date_format(sample_values[0])
                
                elif dtype in ['int64', 'float64', 'int32', 'float32', 'int16', 'float16']:
                    metadata['numeric_columns'].append(col)
                
                elif unique_count < total_count * 0.5 and unique_count < 100:
                    metadata['categorical_columns'].append(col)
                
                else:
                    metadata['text_columns'].append(col)
            
            logger.info(f"Analyzed table '{table_name}': {len(metadata['columns'])} columns")
            logger.info(f"  Date cols: {metadata['date_columns']}")
            logger.info(f"  Categorical cols: {metadata['categorical_columns']}")
            logger.info(f"  Numeric cols: {metadata['numeric_columns']}")
            logger.info(f"  Text cols: {metadata['text_columns']}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error analyzing table {table_name}: {e}")
            return {'columns': [], 'date_columns': [], 'numeric_columns': [], 
                    'categorical_columns': [], 'text_columns': [], 
                    'column_samples': {}, 'column_stats': {}, 'date_formats': {}}

    def _detect_date_format(self, sample_value) -> str:
        """Detect date format from sample value."""
        sample_str = str(sample_value).strip()
        
        if re.match(r'\d{4}-\d{2}-\d{2}', sample_str):
            return 'YYYY-MM-DD'
        elif re.match(r'\d{2}/\d{2}/\d{4}', sample_str):
            return 'DD/MM/YYYY'
        elif re.match(r'\d{4}/\d{2}/\d{2}', sample_str):
            return 'YYYY/MM/DD'
        elif re.match(r'\d{2}-\d{2}-\d{4}', sample_str):
            return 'DD-MM-YYYY'
        else:
            return 'UNKNOWN'

    def _extract_date_column_from_query(self, user_query: str, date_columns: List[str]) -> str:
        """
        Extract which date column user wants by matching keywords.
        Returns the matched column name or first date column if no match.
        Uses substring matching to handle timezone suffixes (UTC, MYT, etc).
        """
        query_lower = user_query.lower()
        
        # Map keywords to column patterns
        keyword_patterns = {
            'purged': ['purge'],
            'deleted': ['delete'],
            'created': ['create'],
            'updated': ['update'],
            'expired': ['expire'],
            'modified': ['modif'],
            'at': ['_at']
        }
        
        # Check if user explicitly mentions a timezone or specific column suffix
        timezone_keywords = ['utc', 'myt', 'est', 'pst', 'gmt', 'ist']
        
        # Check which keyword user mentioned
        for keyword, patterns in keyword_patterns.items():
            if keyword in query_lower:
                # Check if user mentioned a specific timezone
                user_timezone = None
                for tz in timezone_keywords:
                    if tz in query_lower:
                        user_timezone = tz
                        break
                
                # Find matching column
                candidates = []
                for col in date_columns:
                    col_lower = col.lower()
                    for pattern in patterns:
                        if pattern in col_lower:
                            candidates.append(col)
                
                # If multiple candidates and user specified timezone, pick the right one
                if candidates:
                    if user_timezone:
                        for col in candidates:
                            if user_timezone in col.lower():
                                logger.info(f"Matched user keyword '{keyword}' with timezone '{user_timezone}' to column '{col}'")
                                return col
                    
                    # If no timezone match, return first candidate
                    logger.info(f"Matched user keyword '{keyword}' to column '{candidates[0]}'")
                    return candidates[0]
        
        # If no match found, return first date column
        if date_columns:
            logger.info(f"No keyword match, using first date column: {date_columns[0]}")
            return date_columns[0]
        
        return None

    def _build_generic_prompt(self, user_query: str, table_name: str) -> str:
        """
        Build intelligent GENERIC prompt that works for ANY table.
        No hardcoded column names or table-specific logic.
        """
        
        # Get or create metadata (with caching)
        if table_name not in self.table_metadata_cache:
            self.table_metadata_cache[table_name] = self._analyze_table_structure(table_name)
        
        metadata = self.table_metadata_cache[table_name]
        
        # Extract which date column user wants
        preferred_date_col = self._extract_date_column_from_query(user_query, metadata['date_columns'])
        
        prompt = f"""You are an expert SQL query generator for SQLite databases.

═══════════════════════════════════════════════════════════════════════════════
TABLE: {table_name}
USER QUESTION: "{user_query}"
═══════════════════════════════════════════════════════════════════════════════

AVAILABLE COLUMNS (USE EXACT NAMES):
{', '.join(metadata['columns'])}

"""
        
        # ============ SECTION 1: COLUMN CATEGORIES (GENERIC) ============
        
        if metadata['date_columns']:
            prompt += "\nDATE/TIME COLUMNS (Auto-detected):\n"
            for col in metadata['date_columns']:
                samples = metadata['column_samples'].get(col, [])[:3]
                date_format = metadata['date_formats'].get(col, 'UNKNOWN')
                is_preferred = " ← USE THIS ONE" if col == preferred_date_col else ""
                prompt += f"   • {col} (Format: {date_format}){is_preferred}\n"
                if samples:
                    prompt += f"     Examples: {samples}\n"
            
            # CRITICAL: Tell LLM which date column to use
            prompt += "\nCRITICAL DATE COLUMN INSTRUCTION:\n"
            if preferred_date_col:
                prompt += f"   USER SPECIFIED: Use '{preferred_date_col}' for all date operations\n"
                prompt += f"   NEVER use other date columns like {[c for c in metadata['date_columns'] if c != preferred_date_col]}\n"
                prompt += f"   ALWAYS use: {preferred_date_col} in WHERE and GROUP BY clauses\n"
            else:
                prompt += f"   - Use the first date column: {metadata['date_columns'][0] if metadata['date_columns'] else 'NONE'}\n"
            prompt += "\n   Remember: If query mentions 'purged' → use purged column\n"
            prompt += "   Remember: If query mentions 'deleted' → use deleted column\n"
            prompt += "   Remember: If query mentions 'created' → use created column\n"
            prompt += "   - Always use strftime() for date operations\n"
            prompt += "   - NEVER hardcode dates - GROUP BY dates to show all periods\n"
        
        if metadata['categorical_columns']:
            prompt += "\nCATEGORICAL COLUMNS (for filtering/grouping):\n"
            for col in metadata['categorical_columns']:
                stats = metadata['column_stats'][col]
                samples = metadata['column_samples'].get(col, [])[:5]
                prompt += f"   • {col} ({stats['unique']} unique values)\n"
                if samples:
                    prompt += f"     Examples: {samples}\n"
        
        if metadata['numeric_columns']:
            prompt += "\nNUMERIC COLUMNS:\n"
            for col in metadata['numeric_columns']:
                samples = metadata['column_samples'].get(col, [])[:3]
                prompt += f"   • {col}\n"
                if samples:
                    prompt += f"     Range: {samples}\n"
        
        if metadata['text_columns']:
            prompt += "\nTEXT COLUMNS:\n"
            for col in metadata['text_columns']:
                stats = metadata['column_stats'][col]
                samples = metadata['column_samples'].get(col, [])[:3]
                prompt += f"   • {col} ({stats['unique']} unique values)\n"
                if samples:
                    prompt += f"     Examples: {samples}\n"
        
        # ============ SECTION 2: GENERIC QUERY PATTERNS ============
        
        prompt += """
═══════════════════════════════════════════════════════════════════════════════
 SQL GENERATION RULES - GENERIC & FLEXIBLE
═══════════════════════════════════════════════════════════════════════════════

CRITICAL: You MUST generate ONLY SELECT queries. NEVER use DELETE, UPDATE, INSERT, 
    DROP, ALTER, CREATE, or TRUNCATE keywords.

STEP 0: EXTRACT NUMBERS FROM USER QUERY

   If user says "top 5", "top 3", "limit 10", etc. → USE THAT EXACT NUMBER!
   NEVER ignore the number or use a different LIMIT.
   ALWAYS use EXACT number from user's question in LIMIT clause.

STEP 1: IDENTIFY QUERY PATTERN

   A) "TOTAL/SUM" → Use SUM() for aggregation
      Example: "total spending"
      → SELECT SUM(amount) AS total_amount FROM table
   
   B) "WHICH MONTH/DATE HAS MOST X" → GROUP BY DATE, NO LIMIT, NO WHERE ON DATE
      Example: "which month has most deletions"
      CORRECT: 
      → SELECT strftime('%Y-%m', date_col) AS month, COUNT(*) AS count
         FROM table
         GROUP BY month
         ORDER BY count DESC
      
      WRONG: DO NOT add WHERE strftime('%Y-%m', date_col) = 'SPECIFIC-MONTH'
      WRONG: DO NOT hardcode a specific month in WHERE clause
      
      (NO LIMIT! NO WHERE ON DATE! Show all months so user can see which is highest)
      (The ORDER BY DESC will automatically show the month with most activity first)
   
   C) "SHOW TOP N X" → GROUP BY with COUNT, ORDER DESC, LIMIT N
      CRITICAL: "Top N" ALWAYS means:
         1. SELECT the column AND COUNT(*)
         2. GROUP BY the column
         3. ORDER BY COUNT(*) DESC
         4. LIMIT N ← Use EXACT N from user request!
      
      Example: "top 5 deletion types"
      → SELECT deletion_type, COUNT(*) AS count
         FROM table
         GROUP BY deletion_type
         ORDER BY COUNT(*) DESC
         LIMIT 5
   
   D) "SHOW ALL TYPES/VALUES OF X" → DISTINCT or GROUP BY, NO LIMIT
      Example: "what are all status types"
      → SELECT DISTINCT status FROM table ORDER BY status
   
   E) "COUNT X" (simple count) → COUNT(*) with no GROUP BY
      Example: "count total records"
      → SELECT COUNT(*) AS count FROM table
   
   F) "COUNT X BY Y" → GROUP BY Y
      Example: "count by status"
      → SELECT status, COUNT(*) AS count
         FROM table
         GROUP BY status
         ORDER BY count DESC
   
   G) "HOW MANY X WHERE condition" → COUNT with WHERE filter
      Example: "how many records where status is active"
      → SELECT COUNT(*) AS count
         FROM table
         WHERE status = 'Active'

STEP 2: HANDLE DATE COLUMNS CORRECTLY (GENERIC)

   CRITICAL DATE RULES (Works for ANY date column):
   
   - When user says "which MONTH/YEAR has most X" → GROUP BY MONTH/YEAR ONLY
     DO NOT add WHERE clause for the date!
     DO NOT hardcode specific months/years!
     CORRECT: GROUP BY strftime('%Y-%m', date_col) ORDER BY count DESC
     WRONG: WHERE strftime('%Y-%m', date_col) = '2025-08'
     
   - When user says "in 2025" or "in February" → USE WHERE to FILTER
     WHERE strftime('%Y', date_col) = '2025'
     WHERE strftime('%Y-%m', date_col) = '2025-02'
   
   - When user says "by DATE/MONTH/YEAR" → USE GROUP BY
     GROUP BY strftime('%Y-%m', date_col)  ← Shows all months
     GROUP BY strftime('%Y', date_col)     ← Shows all years
     GROUP BY DATE(date_col)               ← Shows all days
   
   - NEVER hardcode month/year in WHERE unless user explicitly asks for specific one
   - NEVER use LIMIT for "which month has most" queries
   - Month MUST be zero-padded: '01', '02', ..., '12'
   - ALWAYS use strftime() for date operations (SQLite only)

STEP 3: MATCH COLUMNS EXACTLY

   - Use exact column names from the list above
   - Match user's keywords to column names intelligently
   - If user mentions a specific date column name → use it
   - If user doesn't specify → use the most logical date column
   - NEVER invent column names

STEP 4: TEXT MATCHING - CASE INSENSITIVE

   When filtering text columns:
   CORRECT: WHERE LOWER(TRIM(column)) LIKE LOWER(TRIM('%pattern%'))
   WRONG: WHERE LOWER(column) LIKE LOWER('%Exact Match%')
   
   Use LOWER() and TRIM() for robustness.

STEP 5: AGGREGATE QUERIES - COMMON PATTERNS

   Single value (COUNT or SUM):
   → SELECT COUNT(*) AS count FROM table
   → SELECT SUM(amount) AS total_amount FROM table
   
   Grouped aggregates:
   → SELECT category, COUNT(*) AS count
      FROM table
      GROUP BY category
      ORDER BY count DESC
   
   With date grouping:
   → SELECT strftime('%Y-%m', date_col) AS month, COUNT(*) AS count
      FROM table
      GROUP BY month
      ORDER BY count DESC
   
   With multiple conditions:
   → SELECT category, COUNT(*) AS count
      FROM table
      WHERE status = 'Active' AND date_col >= DATE('2025-01-01')
      GROUP BY category
      ORDER BY count DESC

STEP 6: FINAL CHECKLIST

   - Is it a "top N" query? Use LIMIT N with exact N from user
   - Is it a "which month/date has most" query? GROUP BY, NO LIMIT
   - Is it a "show all types" query? DISTINCT or GROUP BY, NO LIMIT
   - Is it a "count" query? Use COUNT(*)
   - Is it a "total/sum" query? Use SUM()
   - Did I match column names exactly?
   - Did I use strftime() for all date operations?
   - Did I use LOWER(TRIM(col)) for text comparisons?
   - Is my query ONLY a SELECT statement?
   - Did I avoid hardcoding specific dates (except WHERE filters)?

═══════════════════════════════════════════════════════════════════════════════

Now generate ONLY the SQL query. No explanations, no markdown, no backticks.
Just the raw SQL SELECT statement:"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call Ollama LLM."""
        chain = PromptTemplate.from_template("{prompt}") | self.llm | self.output_parser
        return chain.invoke({"prompt": prompt})

    def generate_sql_query(self, natural_language_query: str, table_name: str) -> str:
        """
        Convert natural language to SQL - works for ANY table.
        """
        
        prompt = self._build_generic_prompt(natural_language_query, table_name)
        raw = self._call_llm(prompt)
        sql = raw.strip()

        # Clean up LLM output
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]

        # Extract SELECT statement
        upper = sql.upper()
        if "SELECT" in upper:
            idx = upper.index("SELECT")
            sql = sql[idx:]
        
        sql = sql.strip().rstrip(";").strip()

        logger.info(f"Generated SQL: {sql}")
        return sql

    def validate_sql_query(self, sql_query: str) -> bool:
        """
        Validate SQL is safe.
        Enhanced to detect DELETE even in column/table names.
        """
        upper = sql_query.upper()
        
        dangerous_patterns = [
            " DROP ", " DELETE ", " UPDATE ", " INSERT ", 
            " ALTER ", " CREATE ", " TRUNCATE ",
            ";DROP", ";DELETE", ";UPDATE", ";INSERT"
        ]
        
        for pattern in dangerous_patterns:
            if pattern in f" {upper} ":
                raise ValueError(f"Dangerous SQL operation detected: {pattern.strip()}")
        
        if not upper.lstrip().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")
        
        if ";" in sql_query:
            parts = sql_query.split(";")
            if len(parts) > 1:
                for part in parts[1:]:
                    part_upper = part.strip().upper()
                    if part_upper and not part_upper.startswith("--"):
                        raise ValueError("Multiple statements not allowed")
        
        return True

    def _prepare_data_summary(self, df: pd.DataFrame, currency: str = "RM") -> str:
        """
        Prepare data summary for LLM context - generic for ANY table.
        """
        summary_parts = []
        
        summary_parts.append(f"Total rows returned: {len(df)}")
        summary_parts.append(f"Columns: {', '.join(df.columns)}")
        
        # Check if any column suggests money
        is_money = any(kw in col.lower() for col in df.columns 
                      for kw in ['amount', 'spending', 'cost', 'price', 'revenue', 'total', 'sum'])
        
        # Single value result (e.g., COUNT(*) or SUM(*) returns one number)
        if len(df) == 1 and len(df.columns) == 1:
            col_name = df.columns[0]
            value = df.iloc[0, 0]
            
            if isinstance(value, (int, float)):
                if is_money:
                    summary_parts.append(f"Single value result ({col_name}): {currency} {value:,.2f}")
                else:
                    summary_parts.append(f"Single value result ({col_name}): {value:,}")
            else:
                summary_parts.append(f"Single value result ({col_name}): {value}")
        
        # Count/aggregation queries - generic
        elif any(col.lower() in ['count', 'frequency', 'total_amount', 'total_spending', 'total', 'sum'] for col in df.columns):
            # Find the numeric aggregation column
            agg_col = None
            for col in df.columns:
                if col.lower() in ['count', 'frequency', 'total_amount', 'total_spending', 'total', 'sum']:
                    agg_col = col
                    break
            
            if agg_col and len(df.columns) >= 2:
                group_col = df.columns[0]
                
                # Calculate totals and percentages
                total = df[agg_col].sum()
                
                summary_parts.append(f"\nTotal {agg_col}: {total:,}")
                
                top_results = []
                for idx, row in df.head(10).iterrows():
                    name = row[group_col]
                    value = row[agg_col]
                    percentage = (value / total * 100) if total > 0 else 0
                    
                    top_results.append(f"  - {name}: {value:,} ({percentage:.1f}%)")
                
                summary_parts.append(f"\nTop results (with percentages):")
                summary_parts.extend(top_results)
                
                if len(df) > 10:
                    remaining = len(df) - 10
                    remaining_sum = df[agg_col].iloc[10:].sum()
                    remaining_pct = (remaining_sum / total * 100) if total > 0 else 0
                    summary_parts.append(f"  ... and {remaining} more entries ({remaining_sum:,}, {remaining_pct:.1f}%)")
        
        # General data (not aggregated)
        else:
            summary_parts.append("\nSample data (first 5 rows):")
            for idx, row in df.head(5).iterrows():
                row_str = " | ".join([f"{col}: {val}" for col, val in row.items()])
                summary_parts.append(f"  - {row_str}")
        
        return "\n".join(summary_parts)

    def explain_query_result(self, natural_language_query: str, sql_query: str, df: pd.DataFrame, currency: str = "RM") -> str:
        """
        Generate natural language explanation with LLM.
        Generic for ANY table structure.
        """
        try:
            import streamlit as st
            
            table_name = st.session_state.active_table if 'active_table' in st.session_state.__dict__ else "the data"
            
            data_summary = self._prepare_data_summary(df, currency)
            
            # Check if money column exists
            is_money = any(kw in col.lower() for col in df.columns 
                          for kw in ['amount', 'spending', 'cost', 'price', 'revenue', 'total', 'sum'])
            
            # ============ GENERIC EXPLANATION PROMPT ============
            explanation_prompt = f"""You are a data analyst. Answer the user's question based on the results.

USER QUESTION: "{natural_language_query}"
TABLE: {table_name}
RESULTS:
{data_summary}

RESPONSE STYLE EXAMPLES TO FOLLOW:

Example 1 (Which month has most):
"Based on the provided data, the month with the most deletions using purged date myt in 2025 is July, with a count of 37,570 (30.3%) of the total 124,053 records. This represents nearly one-third of all deletions recorded in 2025, highlighting the significant activity in this month."

Example 2 (Count with filter):
"Based on the provided data, in February 2024, there was a total of 17 domain name purges, specifically due to 'Deletions by request'. This represents a notable number of deletions for that month."

Example 3 (Top N list):
"Based on the provided data, the top 5 TLD names are:
1. my: 103,673 (65.4%)
2. com.my: 51,881 (32.7%)
3. org.my: 1,413 (0.9%)
4. net.my: 1,012 (0.6%)
5. biz.my: 602 (0.4%)
These TLD names account for the majority of domain name purges, with 'my' being the most common by a significant margin."

Example 4 (Show types):
"Based on the provided data, there are X types of [field]: [list them]. These categories account for all [total] rows in the table."

INSTRUCTIONS:
- Always start with "Based on the provided data"
- Use exact numbers with commas (37,570 not 37570)
- Include percentages (e.g., 30.3%)
- For "which month has most" queries: state the month, count, percentage, and add brief context
- For "top N" queries: list all results with counts and percentages, add brief insight
- For "count" queries: state total count and what it represents (NO extra speculation)
- For "show types" queries: list the types found (NO extra interpretation)
- ONLY 2-3 sentences max - be concise
- Do NOT add speculative statements like "suggests", "indicates", "likely", "probably"
- Do NOT mention what the row count means or make assumptions about data
- Do NOT mention SQL or technical details
- Do NOT invent data - only use what's in the results
- ONLY state facts from the data, nothing more

Now provide your response:"""

            response = self._call_llm(explanation_prompt)
            response = response.strip()
            
            # If LLM returns empty, use fallback
            if not response:
                logger.warning("LLM returned empty response, using fallback")
                return f"Based on the provided data, the query returned **{len(df)}** result(s)."
            
            return response
                
        except Exception as e:
            logger.error(f"Explanation error: {e}")
            return f"Based on the provided data, the query executed successfully with **{len(df)}** result(s)."


# Global instance
sql_chain_manager = SQLChainManager()