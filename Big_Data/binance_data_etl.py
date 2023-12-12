import websocket
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from binance_etl.WebSocketConnection import WebSocketConnection, Binance

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
    schedule=timedelta(days=1),
    catchup=True
)


def run_websocket_connection(**kwargs):
    # Instantiate WebSocketConnection with your default values
    ws_connection = WebSocketConnection
    try:
        ws_connection.ticker_instance = Binance(ticker=ws_connection.ticker, timeframe=ws_connection.timeframe)
        connection = websocket.WebSocketApp(ws_connection.ticker_instance.ticker_data(),
                                            on_message=ws_connection.on_message)
        print("Date   :Open\t\t:High\t\t:low\t\t:close")
        connection.run_forever()
    except websocket.WebSocketConnectionClosedException:
        print("Error establishing a connection: {error}")

    data_to_push = ws_connection.data_entry

    # Push the data to XCom
    kwargs['ti'].xcom_push(task_ids='websocket_task', key='data', value=data_to_push)


def calculate_highest_lowest(**kwargs):
    ti = kwargs['ti']  # Get the TaskInstance object
    ws_connection = ti.xcom_pull(task_ids='websocket_task')  # Retrieve data from the WebSocket task

    if not ws_connection:
        print("No data received from the WebSocket task. Aborting.")
        return

    data = ws_connection['data']  # Assuming you store the WebSocket data in a 'data' key
    if not data:
        print("No data available. Aborting.")
        return

    # Extract the relevant values for calculation within the last minute
    now = datetime.now()
    last_minute_data = [entry for entry in data if now - timedelta(seconds=60) <= entry['timestamp'] <= now]

    if not last_minute_data:
        print("No data within the last minute. Aborting.")
        return

    open_prices = [float(entry['o']) for entry in last_minute_data]

    if not open_prices:
        print("No open prices available. Aborting.")
        return

    highest_value = max(open_prices)
    lowest_value = min(open_prices)

    print(f"Highest Value in the Last Minute: {highest_value}, Lowest Value: {lowest_value}")


ws_task = PythonOperator(
    task_id='websocket_task',
    python_callable=run_websocket_connection,
    provide_context=True,
    dag=dag,
)
calculate_task = PythonOperator(
    task_id='calculate_task',
    python_callable=calculate_highest_lowest,
    provide_context=True,  # Set to True to pass the context, allowing access to XCom data
    dag=dag,
)

calculate_task.set_upstream(ws_task)
