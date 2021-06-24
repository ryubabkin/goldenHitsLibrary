import sqlalchemy as sql
import pandas as pd


def create_engine(host, port, login, pwd, db_name):
    connection_str = f'mysql+pymysql://{login}:{pwd}@{host}:{port}/{db_name}'
    engine = sql.create_engine(connection_str, echo=False, pool_recycle=180)
    return engine


def read_sql(table_name, engine):
    data = pd.read_sql_table(table_name=table_name, con=engine)
    return data


def write_sql(data, table_name, engine):
    data.to_sql(name=table_name, con=engine, if_exists='append', index=False)
