import logging
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


class ChartGenerator:
    """Smart chart generator that adapts to data structure with fallback support."""

    def get_available_chart_types(self):
        """Return available chart types."""
        return {
            "bar": "Bar Chart",
            "line": "Line Chart",
            "pie": "Pie Chart",
            "scatter": "Scatter Plot",
            "area": "Area Chart",
        }

    def suggest_chart_type(self, df: pd.DataFrame) -> str:
        """
        Intelligently suggest best chart type based on data structure.
        """
        if df.empty:
            return None

        num_rows = len(df)
        num_cols = len(df.columns)
        
        # CRITICAL FIX: Single value results (no counts) → No chart
        if num_rows == 1 and num_cols == 1:
            logger.info("Single value result - skipping chart")
            return None
        
        # Single row with multiple columns but NO count column → No chart
        if num_rows == 1 and num_cols > 1:
            count_like = [col for col in df.columns if any(kw in col.lower() for kw in ["count", "frequency", "total", "sum"])]
            if not count_like:
                logger.info("Single row without count column - skipping chart")
                return None
            return "bar"

        # Check column types
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Count/frequency data (common pattern)
        count_like = [col for col in df.columns if any(kw in col.lower() for kw in ["count", "frequency", "total", "sum"])]
        if count_like:
            if num_rows <= 10:
                return "pie"  # Small categorical → pie
            else:
                return "bar"  # Larger → bar chart

        # Two columns: one categorical + one numeric → bar
        if num_cols == 2 and len(categorical_cols) == 1 and len(numeric_cols) == 1:
            if num_rows <= 15:
                return "pie"
            return "bar"

        # Time series (date column + numeric) → line
        date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
        if date_cols and numeric_cols:
            return "line"

        # Two numeric columns → scatter
        if len(numeric_cols) >= 2:
            return "scatter"

        # Default: bar chart
        return "bar"

    def create_chart(self, df: pd.DataFrame, chart_type: str = None, title: str = "Data Visualization"):
        """
        Create and display appropriate chart with intelligent fallback.
        Tries multiple chart types if primary fails.
        """
        if df.empty:
            st.info("No data to visualize.")
            return

        # CRITICAL FIX: Skip chart for single-value results
        if len(df) == 1 and len(df.columns) == 1:
            logger.info("Skipping chart for single value result")
            return

        # Auto-suggest chart type if not provided
        if not chart_type:
            chart_type = self.suggest_chart_type(df)
        
        if not chart_type:
            logger.info("No suitable chart type for this data")
            return

        # Define fallback chain: try primary, then fallbacks
        fallback_chain = self._get_fallback_chain(chart_type)
        
        fig = None
        last_error = None

        # Try each chart type in the fallback chain
        for attempt_type in fallback_chain:
            try:
                logger.info(f"Attempting to create {attempt_type} chart...")
                
                if attempt_type == "bar":
                    fig = self._create_bar_chart(df, title)
                elif attempt_type == "line":
                    fig = self._create_line_chart(df, title)
                elif attempt_type == "pie":
                    fig = self._create_pie_chart(df, title)
                elif attempt_type == "scatter":
                    fig = self._create_scatter_plot(df, title)
                elif attempt_type == "area":
                    fig = self._create_area_chart(df, title)

                if fig:
                    logger.info(f"Successfully created {attempt_type} chart")
                    st.plotly_chart(fig, use_container_width=True)
                    return  # Success - stop trying

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Failed to create {attempt_type} chart: {e}")
                continue  # Try next in chain

        # All attempts failed
        if last_error:
            logger.error(f"All chart types failed. Last error: {last_error}")

    def _get_fallback_chain(self, primary_chart: str) -> list:
        """
        Get fallback chart types based on primary choice.
        Each primary type has a fallback chain.
        """
        fallback_chains = {
            "bar": ["bar", "pie", "line", "area"],
            "pie": ["pie", "bar", "line", "area"],
            "line": ["line", "area", "bar", "pie"],
            "scatter": ["scatter", "line", "bar", "pie"],
            "area": ["area", "line", "bar", "pie"],
        }
        
        return fallback_chains.get(primary_chart, ["bar", "pie", "line", "area"])

    def _create_bar_chart(self, df: pd.DataFrame, title: str):
        """Create bar chart with smart column selection."""
        try:
            # Find best x and y columns
            x_col, y_col = self._select_xy_columns(df)

            if not x_col or not y_col:
                raise ValueError("Could not determine x/y columns for bar chart")

            # Limit to top 20 for readability
            plot_df = df.nlargest(20, y_col) if len(df) > 20 else df

            fig = px.bar(
                plot_df,
                x=x_col,
                y=y_col,
                title=title,
                labels={x_col: x_col.replace("_", " ").title(), y_col: y_col.replace("_", " ").title()},
                color=y_col,
                color_continuous_scale="Blues",
            )

            fig.update_layout(
                showlegend=False,
                height=500,
                xaxis_tickangle=-45,
                template="plotly_dark" if st.session_state.get("dark_mode", True) else "plotly_white",
            )

            return fig

        except Exception as e:
            logger.error(f"Bar chart error: {e}")
            raise

    def _create_pie_chart(self, df: pd.DataFrame, title: str):
        """Create pie chart for categorical distribution."""
        try:
            # Find label and value columns
            label_col, value_col = self._select_xy_columns(df)

            if not label_col or not value_col:
                raise ValueError("Could not determine label/value columns for pie chart")

            # Limit to top 10 + "Others"
            if len(df) > 10:
                top_df = df.nlargest(10, value_col)
                others_sum = df.nsmallest(len(df) - 10, value_col)[value_col].sum()
                others_row = pd.DataFrame({label_col: ["Others"], value_col: [others_sum]})
                plot_df = pd.concat([top_df, others_row], ignore_index=True)
            else:
                plot_df = df

            fig = px.pie(
                plot_df,
                names=label_col,
                values=value_col,
                title=title,
                hole=0.3,  # Donut chart
            )

            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(
                height=500,
                template="plotly_dark" if st.session_state.get("dark_mode", True) else "plotly_white",
            )

            return fig

        except Exception as e:
            logger.error(f"Pie chart error: {e}")
            raise

    def _create_line_chart(self, df: pd.DataFrame, title: str):
        """Create line chart (typically for time series)."""
        try:
            x_col, y_col = self._select_xy_columns(df)

            if not x_col or not y_col:
                raise ValueError("Could not determine x/y columns for line chart")

            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                title=title,
                labels={x_col: x_col.replace("_", " ").title(), y_col: y_col.replace("_", " ").title()},
                markers=True,
            )

            fig.update_layout(
                height=500,
                template="plotly_dark" if st.session_state.get("dark_mode", True) else "plotly_white",
            )

            return fig

        except Exception as e:
            logger.error(f"Line chart error: {e}")
            raise

    def _create_scatter_plot(self, df: pd.DataFrame, title: str):
        """Create scatter plot for numeric relationships."""
        try:
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

            if len(numeric_cols) < 2:
                raise ValueError("Need at least 2 numeric columns for scatter plot")

            x_col, y_col = numeric_cols[0], numeric_cols[1]

            # Color by categorical if available
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            color_col = categorical_cols[0] if categorical_cols else None

            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                title=title,
                labels={x_col: x_col.replace("_", " ").title(), y_col: y_col.replace("_", " ").title()},
            )

            fig.update_layout(
                height=500,
                template="plotly_dark" if st.session_state.get("dark_mode", True) else "plotly_white",
            )

            return fig

        except Exception as e:
            logger.error(f"Scatter plot error: {e}")
            raise

    def _create_area_chart(self, df: pd.DataFrame, title: str):
        """Create area chart (typically for cumulative data)."""
        try:
            x_col, y_col = self._select_xy_columns(df)

            if not x_col or not y_col:
                raise ValueError("Could not determine x/y columns for area chart")

            fig = px.area(
                df,
                x=x_col,
                y=y_col,
                title=title,
                labels={x_col: x_col.replace("_", " ").title(), y_col: y_col.replace("_", " ").title()},
            )

            fig.update_layout(
                height=500,
                template="plotly_dark" if st.session_state.get("dark_mode", True) else "plotly_white",
            )

            return fig

        except Exception as e:
            logger.error(f"Area chart error: {e}")
            raise

    def _select_xy_columns(self, df: pd.DataFrame):
        """
        Smart column selection for x (categorical) and y (numeric).
        Handles various data structures.
        """
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Pattern 1: count/frequency column present
        count_like = [col for col in df.columns if any(kw in col.lower() for kw in ["count", "frequency", "total", "sum"])]
        if count_like:
            y_col = count_like[0]
            # X is the first non-count column
            x_col = [col for col in df.columns if col != y_col][0] if len(df.columns) > 1 else None
            return x_col, y_col

        # Pattern 2: Standard categorical + numeric
        if categorical_cols and numeric_cols:
            return categorical_cols[0], numeric_cols[0]

        # Pattern 3: Two numeric columns
        if len(numeric_cols) >= 2:
            return numeric_cols[0], numeric_cols[1]

        # Pattern 4: Single numeric (use index as x)
        if numeric_cols:
            return None, numeric_cols[0]

        # Pattern 5: All categorical (count occurrences)
        if categorical_cols:
            return categorical_cols[0], None

        # Fallback
        if len(df.columns) >= 2:
            return df.columns[0], df.columns[1]

        return None, None


# Global instance
chart_generator = ChartGenerator()