## Extraction Pipeline
The [extraction pipeline](https://github.com/justinbt1/Multimodal-Document-Classification/tree/main/extraction_pipeline) 
extracts both text and page image features from the original raw NDR documents, saving them to disk to create a
persistent dataset. Text features are stored as a JSON file, while page image features were stored as a directory of JPEG 
files. The feature extraction process for each document is logged and stored as a row in a database table that maintains 
the relationship between each document, and it's extracted feature files, recorded extraction status and any errors 
encountered during the extraction process.  

The extracted features are available for download [here](https://drive.google.com/drive/folders/1RHbfZAoXNeK5r_JgZk4m1r_u37tjrVKo?usp=sharing) 
as a .zip archive containing page image feature JPEGs, text feature JSON files and an Excel dump of the database.  

The extraction pipeline is written in Python 3.7 and calls the third party Apache Tika Server, Tesseract and Poppler 
libraries. To make the feature extraction process efficient, the native Python multi-processing module is used to 
parallelise the workload into separate processes distributing extraction across multiple CPU cores. The below diagram 
shows the flow for a single process for extracting text and image features from a document:

![License: MIT](https://github.com/justinbt1/Multimodal-Document-Classification/blob/main/media/Extraction%20Flow.jpg)

#### Pipeline Requirements
The data pipeline is dependent on a number of external C, C++ and Java libraries:
- [Java](https://adoptopenjdk.net/) - version 14.0.1 
- [Apache TikaServer](https://repo1.maven.org/maven2/org/apache/tika/tika-server/1.24/) - version 1.24 
- [Tesseract OCR Engine](https://github.com/UB-Mannheim/tesseract/wiki) - version 4.1.0
- [Poppler](https://poppler.freedesktop.org/) - version 20.08.0

#### Database
The data pipeline also requires a relational database to track the extraction process, though this project 
uses [MySQL Community Server](https://dev.mysql.com/downloads/mysql/) the pipeline code should work with most relational 
databases that support multiple concurrent connections.

To configure the data pipeline to work with a database, the 
[database_config.json](https://github.com/justinbt1/Multimodal-Document-Classification/blob/main/configs/database_config.json) file needs to be updated with the database 
name, host, port, credentials and table name. The appropriate dialect and database driver also need to be set in the configuration file, 
the default dialect for MySQL is mysql and the default driver is pymysql.

The SQL script used to create the table used in this project can be found 
[here](https://github.com/justinbt1/Multimodal-Document-Classification/blob/main/scripts/create_table.sql).

#### Pipeline Configuration
The extraction pipeline needs to be configured before use, this can be done by editing the [config.json](https://github.com/justinbt1/Multimodal-Document-Classification/blob/main/configs/config.json)
file 
- File Directory - Path to directory containing original unprocessed NDR documents.
- Image Output Directory - Directory to output converted page images. 
- N Page Images - Number of pages to convert to JPEG images.
- Text Output Directory - Output directory for JSON files containing extracted text. 
- Tika Server - Path to Apache Tika Server .jar file.
- N Cores - Number of cores to use for multiprocessing.
