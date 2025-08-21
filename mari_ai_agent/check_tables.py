#!/usr/bin/env python3
"""
Check what tables exist in the database
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.db.connection import db_manager
from sqlalchemy import text

def check_database():
    """Check what tables and data exist"""
    try:
        print("Checking database tables...")
        
        with db_manager.get_session() as session:
            # List all tables
            query = text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
            result = session.execute(query)
            tables = [row[0] for row in result.fetchall()]
            print(f'Available tables: {tables}')
            
            # If there are tables, show sample data from each
            for table in tables[:5]:  # Check first 5 tables
                try:
                    print(f'\n--- Table: {table} ---')
                    
                    # Get column info
                    col_query = text(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}';")
                    col_result = session.execute(col_query)
                    columns = [(row[0], row[1]) for row in col_result.fetchall()]
                    print(f'Columns: {[f"{name}({dtype})" for name, dtype in columns]}')
                    
                    # Get row count
                    count_query = text(f'SELECT COUNT(*) FROM {table};')
                    count_result = session.execute(count_query)
                    row_count = count_result.fetchone()[0]
                    print(f'Row count: {row_count}')
                    
                    # Show sample data
                    if row_count > 0:
                        sample_query = text(f'SELECT * FROM {table} LIMIT 3;')
                        result = session.execute(sample_query)
                        rows = result.fetchall()
                        print('Sample data:')
                        for row in rows:
                            print('  ', dict(row._mapping))
                    
                except Exception as e:
                    print(f'Error reading table {table}: {e}')
        
        return tables
        
    except Exception as e:
        print(f'Database error: {e}')
        return []

if __name__ == "__main__":
    tables = check_database()