import logging
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from global_vars import DATABASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self):
        self.engine = None
        self.Session = None
        self.connect()

    def connect(self):
        """Establish database connection and create metadata tables."""
        try:
            self.engine = create_engine(DATABASE_URL, echo=False, future=True)
            self.Session = sessionmaker(bind=self.engine)
            logger.info(f"Database connection established: {DATABASE_URL}")
            self.create_metadata_tables()
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def create_metadata_tables(self):
        """
        Create internal tables if they don't exist:

        - chat_history: stores user queries, SQL, chart type, execution status, etc.
        """
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_query TEXT NOT NULL,
                        generated_sql TEXT NOT NULL,
                        chart_type TEXT,
                        result_data TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        execution_status TEXT DEFAULT 'success'
                    )
                """
                    )
                )
            logger.info("Metadata tables created / verified")
        except Exception as e:
            logger.error(f"Error creating metadata tables: {e}")
            raise

    # ---------- CRUD helpers ----------

    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query and return results as pandas DataFrame."""
        try:
            df = pd.read_sql(sql_query, self.engine)
            return df
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

    def save_chat_history(
        self,
        user_query: str,
        generated_sql: str,
        chart_type: str | None = None,
        result_data: str | None = None,
        status: str = "success",
    ) -> None:
        """Save chat interaction to history."""
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                    INSERT INTO chat_history
                    (user_query, generated_sql, chart_type, result_data, execution_status)
                    VALUES (:user_query, :generated_sql, :chart_type, :result_data, :status)
                """
                    ),
                    {
                        "user_query": user_query,
                        "generated_sql": generated_sql,
                        "chart_type": chart_type,
                        "result_data": result_data,
                        "status": status,
                    },
                )
            logger.info("Chat history saved")
        except Exception as e:
            logger.error(f"Error saving chat history: {e}")

    def get_chat_history(self, limit: int = 10) -> pd.DataFrame:
        """Retrieve recent chat history."""
        try:
            query = """
                SELECT user_query, generated_sql, chart_type, timestamp, execution_status
                FROM chat_history
                ORDER BY timestamp DESC
                LIMIT :limit
            """
            df = pd.read_sql(query, self.engine, params={"limit": limit})
            return df
        except Exception as e:
            logger.error(f"Error retrieving chat history: {e}")
            return pd.DataFrame()

    def get_table_info(self) -> dict:
        """
        Get information about available tables and columns.
        Includes ALL tables (also chat_history), you can filter in UI if needed.
        """
        try:
            inspector = inspect(self.engine)
            tables_info: dict = {}

            for table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                tables_info[table_name] = {
                    "columns": [col["name"] for col in columns],
                    "column_details": columns,
                }

            return tables_info
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            return {}


# Global instance
db_manager = DatabaseManager()
