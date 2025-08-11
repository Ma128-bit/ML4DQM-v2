from flask import Flask, render_template
import sqlite3

app = Flask(__name__)

DB_NAME = "job_database.db"
TABLE_NAME = "jobs"


def fetch_all_jobs_full(db_name, table_name):
    try:
        conn = sqlite3.connect(db_name)
        c = conn.cursor()
        c.execute(f"SELECT * FROM {table_name}")
        rows = c.fetchall()
        col_names = [desc[0] for desc in c.description]
        conn.close()

        cleaned_rows = []
        for row in rows:
            cleaned_rows.append([
                "Not Set" if cell is None or cell == "" else cell for cell in row
            ])

        return col_names, cleaned_rows
    except Exception as e:
        print(f"Error loading jobs: {e}")
        return [], []


@app.route("/")
def index():
    columns, rows = fetch_all_jobs_full(DB_NAME, TABLE_NAME)
    return render_template("table.html", columns=columns, rows=rows)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)