-- Create the sample_data table
CREATE TABLE IF NOT EXISTS sample_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER,
    salary REAL,
    department TEXT
);

-- Insert sample data
INSERT INTO sample_data (name, age, salary, department) VALUES
    ('Alice', 30, 70000.00, 'Engineering'),
    ('Bob', 40, 85000.50, 'Marketing'),
    ('Charlie', 35, 90000.75, 'Finance'),
    ('David', 28, 65000.00, 'HR'),
    ('Eve', 45, 120000.00, 'Engineering'),
    ('Frank', 50, 110000.00, 'Finance'),
    ('Grace', 29, 72000.00, 'Marketing'),
    ('Hannah', 38, 98000.00, 'HR'),
    ('Ian', 33, 88000.00, 'Engineering'),
    ('Jack', 41, 102000.00, 'Finance');

-- Verify the inserted data
SELECT * FROM sample_data;
