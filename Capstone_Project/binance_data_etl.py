from time import time
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from binance_etl import Cassandra as cs
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

default_args = {
    'owner': 'roy',
    'depends_on_past': False,
    'start_date': datetime.utcnow(),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'binance_data_dag',
    default_args=default_args,
    description='A DAG to fetch and process Binance data',
    schedule=timedelta(minutes=1),  # Change schedule to '* * * * *' to run every minute
    catchup=True
)


def connect_to_cassandra():
    cs.establish_connection()


def create_returns_table():
    # Create the SQLAlchemy engine
    session = cs.establish_connection()

    # Define the SQL query
    create_table_query = """CREATE TABLE IF NOT EXISTS btcusd.returns_table (timestamp TIMESTAMP PRIMARY KEY,returns DOUBLE);"""
    session.execute(create_table_query)


def get_returns(**kwargs):
    session = cs.establish_connection()
    # Query to retrieve data for the last minute (timestamps in seconds)
    one_minute_ago = time() - 60
    select_query = f"SELECT timestamp, high_price, low_price FROM btcusd.btcusd_time_series WHERE timestamp >= {one_minute_ago} ALLOW FILTERING"

    rows = session.execute(select_query)
    data = []
    for row in rows:
        if hasattr(row, 'timestamp') and hasattr(row, 'high_price') and hasattr(row, 'low_price'):
            data.append({
                'timestamp': row.timestamp,
                'high_price': row.high_price,
                'low_price': row.low_price
            })
        else:
            # Handle the case where the expected columns are not present in the row
            print("Warning: One or more columns missing in the row.")

    # Create a DataFrame from the retrieved data
    df = pd.DataFrame(data)
    print(df)
    # Calculate returns using a simple formula
    df['avg_price'] = (df['high_price'] + df['low_price']) / 2
    df['return'] = df['avg_price'].pct_change()

    # Push the DataFrame to XCom
    kwargs['ti'].xcom_push(key='dataframe', value=df.to_json())


def load_to_db(**kwargs):
    session = cs.establish_connection()
    # Retrieve the DataFrame from XCom
    ti = kwargs['ti']
    df_json = ti.xcom_pull(task_ids='get_returns', key='dataframe')
    df = pd.read_json(df_json)

    # Iterate over rows and insert data into the database
    for index, row in df.iterrows():
        timestamp = row['timestamp']
        returns = row['return']

        # Use a raw SQL query to insert data into the database
        query = f"INSERT INTO btcusd.returns_table (timestamp, returns) VALUES ('{timestamp}', {returns})"
        session.execute(query)


def train_model():
    # Establish Cassandra connection
    session = cs.establish_connection()

    # Execute the query to retrieve data from the Cassandra table
    query = "SELECT timestamp, returns FROM btcusd.returns_table"
    rows = session.execute(query)

    # Convert the result to a DataFrame
    data = pd.DataFrame(rows, columns=["timestamp", "returns"])

    # Assuming 'timestamp' is the index
    data.set_index('timestamp', inplace=True)

    # Train the ARIMA model with your desired order
    model = ARIMA(data['returns'], order=(3, 1, 3))
    fitted_model = model.fit()


cs_task = PythonOperator(
    task_id='cassandra_task',
    python_callable=connect_to_cassandra,
    provide_context=True,
    dag=dag,
)
gt_data = PythonOperator(
    task_id="get_returns",
    python_callable=get_returns,
    provide_context=True,
    dag=dag,
)

load_to_db_task = PythonOperator(
    task_id='load_to_db_task',
    python_callable=load_to_db,
    provide_context=True,
    dag=dag,
)
create_returns_table_task = PythonOperator(
    task_id='create_returns_table_task',
    python_callable=create_returns_table,
    dag=dag,
)
train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag
)

cs_task.set_downstream(create_returns_table_task)
cs_task.set_downstream(gt_data)
gt_data.set_downstream(load_to_db_task)
gt_data.set_downstream(train_model)
