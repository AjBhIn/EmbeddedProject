"""
DocuNode Web Dashboard (dashboard.py)
--------------------------------------
Lightweight web server to display scanned document records from SQLite[cite: 3].
Allows users to view and download PNG files directly from a browser.
"""

from flask import Flask, render_template_string, send_file
import sqlite3
import os
import sys

app = Flask(__name__)
DB_NAME = "docunode.db"

# HTML Template with styling and direct download buttons
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>DocuNode Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 30px; background-color: #f4f6f9; color: #333; }
        h1 { color: #1a252f; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; background: white; border-radius: 8px; overflow: hidden; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #2c3e50; color: white; }
        tr:hover { background-color: #f1f1f1; }
        .btn { padding: 8px 14px; background-color: #27ae60; color: white; text-decoration: none; border-radius: 4px; font-weight: bold; }
        .btn:hover { background-color: #219150; }
        .badge { padding: 4px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold; background-color: #e8f8f5; color: #27ae60; }
    </style>
</head>
<body>
    <h1>DocuNode - Edge Document Dashboard</h1>
    <p>Logged scans retrieved from local SQLite database (<code>docunode.db</code>)</p>
    
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Timestamp</th>
                <th>Status</th>
                <th>Processed Path</th>
                <th>Action</th>
            </tr>
        </thead>
        <tbody>
            {% for scan in scans %}
            <tr>
                <td><strong>#{{ scan[0] }}</strong></td>
                <td>{{ scan[3] }}</td>
                <td><span class="badge">{{ scan[4] }}</span></td>
                <td><code>{{ scan[2] }}</code></td>
                <td>
                    <a class="btn" href="/download/{{ scan[0] }}">Download PNG</a>
                </td>
            </tr>
            {% else %}
            <tr>
                <td colspan="5" style="text-align: center; color: #888;">No scans found in database. Run a scan in cameragui.py first.</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>
"""

def fetch_scans():
    """Reads scan logs from SQLite[cite: 3]."""
    if not os.path.exists(DB_NAME):
        return []
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT id, raw_path, processed_path, timestamp, status FROM scans ORDER BY id DESC;")
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"[DB Error] {e}")
        return []

@app.route("/")
def index():
    scans = fetch_scans()
    return render_template_string(HTML_TEMPLATE, scans=scans)

@app.route("/download/<int:record_id>")
def download(record_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT processed_path FROM scans WHERE id = ?;", (record_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row and os.path.exists(row[0]):
        return send_file(row[0], as_attachment=True)
    return "File not found.", 404

def start_server():
    """Explicit functional startup sequence."""
    print("--------------------------------------------------")
    print("[Flask] Starting DocuNode Web Dashboard...")
    print(f"[Flask] Access locally: http://localhost:5000")
    print("--------------------------------------------------")
    
    # Run WSGI server directly
    from werkzeug.serving import run_simple
    run_simple("0.0.0.0", 5000, app, use_reloader=False, use_debugger=True)

if __name__ == "__main__":
    start_server()