# Data Insights Analyzer

## Overview
The Data Insights Analyzer is a web application designed to facilitate exploratory data analysis (EDA) using SQLite or PostgreSQL databases. It leverages OpenAI's GPT-4 to generate meaningful questions and SQL queries to uncover insights from the data.

## Features
- **Dynamic Table Selection**: Users can select tables from the connected database.
- **Automated EDA Questions**: The application generates exploratory questions based on the database schema.
- **SQL Query Generation**: Automatically generates SQL queries to answer the generated questions.
- **Insight Display**: Displays the results of the SQL queries, including the ability to toggle the visibility of the SQL queries.
- **CSV Upload**: Users can upload CSV files to create new tables in the database.

## Prerequisites
- uv (https://docs.astral.sh/uv/)
- SQLite or PostgreSQL
- OpenAI API Key

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/sgolestane/EDA
    cd EDA
    ```
   
2. Set the OpenAI API key:
    ```sh
    export OPENAI_API_KEY=your_openai_api_key
    ```

## Usage
1. Start the Flask application:
    ```sh
    # How to run
   uv run main.py --database_dialect postgresql --database "postgresql://localhost/eda"
    ```

2. Open your web browser and navigate to `http://localhost:8000`.

3. Select a table from the dropdown and click "Run Analysis" to generate insights.

4. To upload a CSV file, select the file and click "Upload CSV".

## How It Works
- **Schema Analysis**: The application analyzes the schema of the selected table to understand its structure.
- **Question Generation**: Uses OpenAI's GPT-4 to generate exploratory questions based on the schema.
- **SQL Query Generation**: Generates SQL queries to answer the generated questions.
- **Query Execution**: Executes the SQL queries on the database and displays the results.

## License
This project is licensed under the MIT License.
