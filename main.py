# /// script
# dependencies = [
#    "pandas",
#    "numpy",
#    "nltk",
#    "scipy",
#    "flask",
#    "logging",
#    "openai",
# ]
# ///

import os
import sqlite3
import pandas as pd
import numpy as np
import logging
import openai
from flask import Flask, request, render_template, jsonify

extension_path = os.path.join(os.path.dirname(__file__), "sqlite-extensions")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_PATH = "quis.db"
QUESTION_LIMIT = 3  # Limit the number of questions


class QueryResult:
    def __init__(self, question, query, result, columns=None):
        self.question = question
        self.query = query
        self.result = result
        self.columns = columns

    def to_dict(self):
        return {
            "question": self.question,
            "query": self.query,
            "result": self.result,
            "columns": self.columns
        }


class QUIS:
    def __init__(self, db_path, table_name):
        self.db_path = db_path
        self.table_name = table_name
        self.questions = []
        self.sql_queries = {}
        logging.info("QUIS instance created using SQLite database: %s, Table: %s", self.db_path, self.table_name)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f'SELECT * FROM {self.table_name} LIMIT 1')
            column_names = [description[0] for description in cursor.description]
            self.schema_info = f"Table: {self.table_name}\nColumns:\n" + "\n".join(column_names)
        logging.info("Schema Info: %s", self.schema_info)

    def generate_questions(self):
        """Uses OpenAI's GPT-4 to generate exploratory questions based on database schema."""
        if not OPENAI_API_KEY:
            logging.error("OpenAI API key is not set.")
            raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")

        openai_client = openai.Client()
        logging.info("Generating EDA questions using OpenAI for schema: %s", self.schema_info)
        # prompt = f"""
        # Given the following SQLite database schema:
        # {self.schema_info}
        # Generate a list of {QUESTION_LIMIT} meaningful exploratory data analysis (EDA) questions.
        # """

        prompt = f"""
        Given the following SQLite database schema:
        {self.schema_info}
        Generate a list of {QUESTION_LIMIT} meaningful exploratory data analysis (EDA) questions to look for insights 
        in the data.
        Do not index the questions, just list them one by one.
        Try to generate questions that are useful for finding outliers, trends, patterns, and relationships in the data.
        """

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an expert in SQL and data analysis."},
                      {"role": "user", "content": prompt}]
        )

        self.questions = [q for q in response.choices[0].message.content.split("\n") if q][:QUESTION_LIMIT]
        logging.info("Generated %d questions using OpenAI.", len(self.questions))

    def generate_sql_queries(self):
        """Uses OpenAI's GPT-4 to generate SQL queries based on the questions."""
        logging.info("Generating SQL queries using OpenAI.")

        openai_client = openai.Client()
        for question in self.questions:
            if not question.strip():
                logging.info("Skipping empty question.")
                continue

            prompt = f"""
            Your job is to generate a SELECT SQL query.
            SQL query should be compliant with sqlite syntax.
            Do not return anything other than the query.
            The returned columns should have proper aliases in English.
            Round the numerical values to 2 decimal places.
            Only return the SQL and nothing else.
            Given the following SQLite database schema:
            {self.schema_info}
            Generate an SQL query to answer the following question:
            {question}
            """

            logging.info("Prompt: [%s]", prompt)
            logging.info("Question: [%s]", question)

            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "You are an expert in SQL query generation."},
                          {"role": "user", "content": prompt}]
            )

            query = response.choices[0].message.content
            # Remove any ```sql from beginning of the query
            query = query.replace("```sql", "").strip()
            # Remove any ``` from the end of the query
            query = query.replace("```", "").strip()
            logging.info("Generated Query: [%s]", query)

            self.sql_queries[question] = query
        logging.info("Generated %d SQL queries.", len(self.sql_queries))

    def execute_queries(self):
        """Executes the generated SQL queries on the SQLite database."""
        insights = []
        logging.info("Executing generated SQL queries.")

        with sqlite3.connect(self.db_path) as conn:
            conn.enable_load_extension(True)
            conn.load_extension(os.path.join(extension_path, "stats2.dylib"))
            cursor = conn.cursor()
            for question, query in self.sql_queries.items():
                try:
                    cursor.execute(query)
                    columns = [description[0] for description in cursor.description]
                    result = cursor.fetchall()
                    insights.append(QueryResult(question, query, result, columns))
                except Exception as e:
                    logging.error("Error executing query: %s, Error: %s", query, str(e))
                    insights.append(QueryResult(question, query, str(e)))

        return insights

    def run(self):
        logging.info("Starting QUIS analysis.")
        self.generate_questions()
        self.generate_sql_queries()
        insights = self.execute_queries()
        logging.info("Analysis complete.")
        return insights


@app.route('/')
def home():
    logging.info("Home page accessed.")
    return render_template('index.html')


@app.route('/tables', methods=['GET'])
def get_tables():
    logging.info("Retrieving table names from the database.")
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite%';")
        tables = [row[0] for row in cursor.fetchall()]
    return jsonify({"tables": tables})


@app.route('/analyze', methods=['POST'])
def analyze():
    logging.info("Received analysis request.")
    data = request.get_json()
    selected_table = data.get('table')
    if not selected_table:
        return jsonify({"error": "No table selected"}), 400

    quis = QUIS(DATABASE_PATH, selected_table)
    insights = quis.run()
    logging.info("Returning generated insights.")
    return jsonify({"insights": [insight.to_dict() for insight in insights]})

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    file = request.files['file']
    if not file:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    filename = file.filename
    table_name = os.path.splitext(filename)[0]

    logging.info("Uploading CSV file: %s", filename)

    try:
        df = pd.read_csv(file)
        column_names = df.columns.tolist()

        prompt = f"""
        Given the following column names:
        {' '.join(column_names)}
        Suggest valid SQLite SQL to generate table. 
        Suggest valid SQLite data types for each column.
        Only return the SQL and nothing else.
        Name the table is {table_name}.
        """
        logging.info("Prompt: %s", prompt)

        openai_client = openai.Client()
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an expert in SQL and data analysis."},
                      {"role": "user", "content": prompt}]
        )

        sql_statement = response.choices[0].message.content
        logging.info("Suggested SQL: %s", sql_statement)

        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            cursor.execute(sql_statement)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            logging.info("Data saved to SQLite database.")

        return jsonify({"success": True})
    except Exception as e:
        logging.error("Error uploading CSV file: %s", str(e))
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    logging.info("Starting Flask application.")
    app.run(host="0.0.0.0", port=8000)
