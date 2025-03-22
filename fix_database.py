import sqlite3

# Connect to the database
conn = sqlite3.connect('pest_detection.db')
c = conn.cursor()

# Fetch all rows
c.execute('SELECT id, pest_id FROM pest_detections')
rows = c.fetchall()

# Update each row
for row in rows:
    row_id, pest_id = row
    if isinstance(pest_id, bytes):
        # Convert bytes to integer
        pest_id_int = int.from_bytes(pest_id, byteorder='little')
        # Update the row
        c.execute('UPDATE pest_detections SET pest_id = ? WHERE id = ?', (pest_id_int, row_id))

# Commit the changes
conn.commit()
conn.close()

print("Database updated successfully!")