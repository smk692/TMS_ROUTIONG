import pandas as pd
from sqlalchemy import create_engine
from src.config import DB_CONFIG

def get_engine():
    conf = DB_CONFIG
    url = f"mysql+pymysql://{conf['user']}:{conf['password']}@{conf['host']}:{conf['port']}/{conf['database']}"
    return create_engine(url)

def fetch_orders():
    engine = get_engine()

    # Prepare parameters
    date = '2025-04-17'
    region = f"경기도%"

    query = """
    SELECT DISTINCT I.invoice_no    AS id
         , C.value                  AS invoice_code
         , D.recipient_road_address AS address1
         , D.recipient_address2     AS address2
         , D.recipient_latitude     AS latitude
         , D.recipient_longitude    AS longitude
      FROM DELIVERY_STATUS DS
      JOIN DELIVERY D
        ON D.order_id = DS.order_id
      JOIN INVOICE I
        ON I.reference_id = DS.order_id
       AND I.type = 'ORDER'
       AND I.tracker_type IN ('TMS_ORDER', 'KAKAO')
 LEFT JOIN CODE C 
        ON I.invoice_code = C.code 
       AND `group` = 'DELIVERY_INVOICE_CODE'
     WHERE DS.delivery_type_code = 'M'
       AND D.morning_schedule_date BETWEEN CONCAT(%s, ' 00:00:00') AND CONCAT(%s, ' 23:59:59')
       AND D.recipient_road_address LIKE %s
    """
    params = (date, date, region)

    return pd.read_sql(query, engine, params=params)