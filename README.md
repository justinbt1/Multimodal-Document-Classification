# Multimodal Document Classification
[![Python Version](https://img.shields.io/badge/python-3.8-blue.svg)](https://github.com/justinbt1/Multimodal-Document-Classification)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

### Multimodal Visual & Text Based Approach to Multi-Page Document Classification.
MSc project investigating multi-modal fusion approaches to combining textual and visual features for multi-page document 
classification of documents within the [Oil & Gas Authority](https://www.ogauthority.co.uk/) 
(OGA) [National Data Repository](https://ndr.ogauthority.co.uk/dp/controller/PLEASE_LOGIN_PAGE) (NDR) using deep multimodal fusion
convolutional long short term memory (C-LSTM) neural networks. This readme gives a brief overview of the project and the code in 
this repository, see the accompanying 
[report](https://github.com/justinbt1/Multimodal-Document-Classification/blob/main/report/project_report.pdf) for the full 
details. Note this is experimental code for my masters project, I would advise against running any of it in production.

### Requirements
All Python dependencies for the project can be installed by building a conda environment from the 
[environments.yml](https://github.com/justinbt1/Multimodal-Document-Classification/blob/main/environment.yml) file. 

### Document Dataset
The data source used in this project was compiled from a corpus of raw documents uploaded by oil companies to the 
[National Data Repository](https://ndr.ogauthority.co.uk/) (NDR), a data repository for 
UK petroleum exploration and production data maintained by the [Oil & Gas Authority](https://www.ogauthority.co.uk/) 
(OGA). The document corpus used consists of a sample of 6,541 documents, these are mostly PDF, Microsoft Office, text 
and image type files.

The documents in this corpus are split into 6 classes:
- geol_geow - Geological end of well reports.
- geo_sed - Geological sedimentary reports.
- gphys_gen - General geophysical reports.
- log_sum - Well log summaries.
- pre-site - Pre-site reports.
- vsp_file - Vertical seismic profiles.

### Extraction Pipeline
The [extraction pipeline](https://github.com/justinbt1/Multimodal-Document-Classification/tree/main/extraction_pipeline) 
extracts both text and page image features from the original raw NDR documents, saving them to disk to create a
persistent dataset. Text features are stored as a JSON file, while page image features were stored as a directory of JPEG 
files. The feature extraction process for each document is logged and stored as a row in a database table that maintains 
the relationship between each document, and it's extracted feature files, recorded extraction status and any errors 
encountered during the extraction process.

The extracted features are available for download [here](https://drive.google.com/drive/folders/1RHbfZAoXNeK5r_JgZk4m1r_u37tjrVKo?usp=sharing) 
as a .zip archive containing page image feature JPEGs, text feature JSON files and an Excel dump of the database.

<details>
<summary><b>Full Pipeline Details</b></summary>

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

</details>

### Experimental Models
For optimal performance it is recommended to use GPU accelerated computing by setting up TensorFlow with GPU support and 
installing the required Nvidia CUDA dependencies as described in the [install instructions](https://www.tensorflow.org/install/gpu).
