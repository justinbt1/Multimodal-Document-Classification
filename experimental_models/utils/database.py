import json
import sqlalchemy
import pandas as pd


def get_db_table():
    """ Loads SQL table as Pandas data frame.

    Returns:
        pd.DataFrame: Data frame containing extraction details.

    """
    config_file = open(r'..\configs\database_config.json', 'rt')
    config = json.loads(config_file.read())
    config_file.close()

    connection_string = '{}+{}://{}:{}@{}:{}/{}'.format(
        config['Dialect'],
        config['Driver'],
        config['Username'],
        config['Password'],
        config['Host'],
        config['Port'],
        config['Database'],
        config['Table']
    )

    engine = sqlalchemy.create_engine(connection_string)
    df = pd.read_sql_table('file_index', con=engine)
    df = df.loc[~df['ext'].isin(('.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx'))]

    return df
