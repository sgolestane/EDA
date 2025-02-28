# /// script
# dependencies = [
#    "pandas",
# ]
# ///

import pandas as pd
import sqlite3
import sys

# Check if the correct number of arguments is passed
if len(sys.argv) != 4:
    print("Usage: python script.py <path_to_csv> <database_name.db> <table_name>")
    sys.exit(1)

# Assign arguments to variables
csv_file_path = sys.argv[1]
database_name = sys.argv[2]
table_name = sys.argv[3]

# Connect to SQLite database
conn = sqlite3.connect(database_name)

# Read CSV file
df = pd.read_csv(csv_file_path)

# Write the data to a SQLite table
df.to_sql(table_name, conn, if_exists='replace', index=False)

# Close the connection
conn.close()
