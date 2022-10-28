import psycopg2
from datetime import datetime, timezone

conn = psycopg2.connect('dbname=fall_detection user=postgres password=papagoose host=localhost port=5432')
cursor = conn.cursor()
print(datetime.now(timezone.utc))
sql = '''
INSERT INTO falls (time) VALUES(%s)
'''
cursor.execute(sql, (datetime.now(timezone.utc),))
# records = cursor.fetchall()
# print(time.time())

conn.commit()
cursor.close()
