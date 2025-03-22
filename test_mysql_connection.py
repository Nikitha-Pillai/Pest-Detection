import mysql.connector

try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Leviosa28*",  # Replace with your actual password
        database="pest_detection"
    )
    print("Connected to MySQL successfully!")
    conn.close()
except mysql.connector.Error as err:
    print(f"Error connecting to MySQL: {err}")