
import sqlite3
import pandas as pd

DB_NAME = "maintenance.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS machines (
                machine_id TEXT PRIMARY KEY,
                temperature REAL,
                rotational_speed REAL,
                torque REAL,
                tool_wear REAL,
                failure INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS maintenance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id TEXT,
                maintenance_date TEXT,
                notes TEXT
            )
        """)


def get_machines():
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM machines ORDER BY machine_id", conn
        )


def add_machine(machine):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO machines
            (machine_id, temperature, rotational_speed, torque,
             tool_wear, failure)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            machine["machine_id"],
            machine["temperature"],
            machine["rotational_speed"],
            machine["torque"],
            machine["tool_wear"],
            machine["failure"]
        ))


def update_machine(machine):
    with get_connection() as conn:
        conn.execute("""
            UPDATE machines
            SET temperature=?, rotational_speed=?, torque=?,
                tool_wear=?, failure=?
            WHERE machine_id=?
        """, (
            machine["temperature"],
            machine["rotational_speed"],
            machine["torque"],
            machine["tool_wear"],
            machine["failure"],
            machine["machine_id"]
        ))


def delete_machine(machine_id):
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM machines WHERE machine_id=?",
            (machine_id,)
        )
        conn.execute(
            "DELETE FROM maintenance WHERE machine_id=?",
            (machine_id,)
        )


def save_maintenance(machine_id, maintenance_date, notes):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO maintenance (machine_id, maintenance_date, notes)
            VALUES (?, ?, ?)
        """, (machine_id, maintenance_date, notes))


def get_maintenance():
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM maintenance ORDER BY id DESC", conn
        )


def import_csv_data(data):
    required = [
        "machine_id", "temperature", "rotational_speed",
        "torque", "tool_wear", "failure"
    ]

    data = data[required].copy()

    with get_connection() as conn:
        data.to_sql(
            "machines", conn, if_exists="append",
            index=False
        )
