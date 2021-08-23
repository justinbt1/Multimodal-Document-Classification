# Multimodal Document Classification
[![Python Version](https://img.shields.io/badge/python-3.8-blue.svg)](https://github.com/justinbt1/Multimodal-Document-Classification)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

### Multimodal Visual & Text Based Approach to Multi-Page Document Classification.
MSc project investigating multi-modal fusion approaches to combining textual and visual features for multi-page document 
classification of documents within the [Oil & Gas Authority](https://www.ogauthority.co.uk/) 
(OGA) [National Data Repository](https://ndr.ogauthority.co.uk/dp/controller/PLEASE_LOGIN_PAGE) (NDR) using deep multimodal fusion
convolutional long short term memory (C-LSTM) neural networks. This readme gives a brief overview of the project and the code in 
this repository, see the accompanying 
[report](https://github.com/justinbt1/Multimodal-Document-Classification/blob/main/report/project_report.pdf) for more experiment 
details.

### Abstract
This project proposes using a multi-modal approach to document classification, combining both text and visual 
modalities to create a more robust classifier for oil and gas exploration and production documents. Given heavy 
dependence on multi-modal communication of information and high cross-modal intra-class variability within these 
documents, it is possible to hypothesize that a multi-modal classification approach combining text and visual feature 
input streams, will outperform a classifier trained on features from a single modality such as text or visual features. 
To investigate this we will build on previous related work on multi-modal classification in other domains, applying and 
adapting these approaches to the classification of exploration and production documents.

### Requirements
All Python dependencies for the project can be installed by building a conda environment from the 
[environments.yml](https://github.com/justinbt1/Multimodal-Document-Classification/blob/main/environment.yml) file.<br>

#### Data Pipeline
This project is dependent on a number of external C, C++ and Java libraries for compiling the multimodal training data 
from the various document types within the NDR.
- [Java](https://adoptopenjdk.net/), version 14.0.1 
- [Apache TikaServer](https://repo1.maven.org/maven2/org/apache/tika/tika-server/1.24/), version 1.24 
- [Tesseract OCR Engine](https://github.com/UB-Mannheim/tesseract/wiki), version 4.1.0
- [Poppler](https://poppler.freedesktop.org/), version 20.08.0

The data pipeline also requires a relational database to track the extraction process, though this project 
uses [MySQL Community Server](https://dev.mysql.com/downloads/mysql/) the pipeline code should work with most relational 
databases that support multiple concurrent connections.

The SQL script used to create the table used in this project can be found 
[here](https://github.com/justinbt1/Multimodal-Document-Classification/blob/main/scripts/create_table.sql).

#### Machine Learning
For optimal performance it is recommended to use GPU accelerated computing by setting up TensorFlow GPU support and 
installing the required Nvidia CUDA dependencies as described in the [install instructions](https://www.tensorflow.org/install/gpu).

### Data
The data used in this project was compiled from raw documents uploaded by oil companies to the 
[National Data Repository](https://ndr.ogauthority.co.uk/dp/controller/PLEASE_LOGIN_PAGE) (NDR), a data repository for 
UK petroleum exploration and production data maintained by the [Oil & Gas Authority](https://www.ogauthority.co.uk/) 
(OGA).
