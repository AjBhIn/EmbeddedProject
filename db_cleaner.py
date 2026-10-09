import sqlite3

# Connect to your database
conn = sqlite3.connect("docunode.db")
cursor = conn.cursor()

# Wipe all records from the table
cursor.execute("DELETE FROM scans")
conn.commit()
conn.close()

print("All records successfully deleted. Dashboard is now clean.")