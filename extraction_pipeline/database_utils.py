import json
import sqlalchemy


class Database:
    """ Manages new database connection and operations.

    Attributes:
        connection: SQLAlchemy connection object.

    """
    def __init__(self, config_path=r'../configs/database_config.json'):
        """ Constructor for database objects.

        Args:
            config_path(str): Location of database configuration file.

        """
        config_file = open(config_path, 'rt')
        config = json.loads(config_file.read())
        config_file.close()

        self._database_schema = config['Database']
        self._table_name = config['Table']

        self._connection_string = '{}+{}://{}:{}@{}:{}/{}'.format(
            config['Dialect'],
            config['Driver'],
            config['Username'],
            config['Password'],
            config['Host'],
            config['Port'],
            self._database_schema,
            self._table_name
        )

        self.connection = None
        self._table = None

    def connect(self):
        """ Establishes and tests the connection to the database.

        """
        engine = sqlalchemy.create_engine(self._connection_string)

        if self._table_name not in engine.table_names():
            raise RuntimeError(f'Table {self._table_name} not found.')

        self.connection = engine.connect().execution_options(autocommit=True)

        meta = sqlalchemy.schema.MetaData(schema=self._database_schema)

        self._table = sqlalchemy.schema.Table(
            self._table_name, meta,
            autoload=True,
            autoload_with=engine
        )

    def insert(self, file_properties):
        """ Extracts properties of FileProperties object and inserts to database.

        Args:
            file_properties(FileProperties): FileProperties object.

        """
        insert = self._table.insert().values(
            original_file=file_properties.file_path,
            image_extracted=file_properties.image_extracted,
            text_extracted=file_properties.text_extracted,
            tika_status=file_properties.tika_status,
            ocr=file_properties.ocr,
            image_dir_ref=file_properties.image_path,
            text_json_ref=file_properties.text_path,
            ext=file_properties.ext,
            error=file_properties.error,
            label=file_properties.label
        )

        self.connection.execute(insert)
