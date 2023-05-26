# Multimodal Document Classification
[![Python Version](https://img.shields.io/badge/python-3.8-blue.svg)](https://github.com/justinbt1/Multimodal-Document-Classification)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

### Multimodal Visual & Text Based Approach to Multi-Page Document Classification.
MSc project investigating multi-modal fusion approaches to combining textual and visual features for multi-page document 
classification of documents within the [North Sea Transition Authority](https://www.ogauthority.co.uk/) 
(OGA) [National Data Repository](https://ndr.ogauthority.co.uk/dp/controller/PLEASE_LOGIN_PAGE) (NDR) using deep multimodal fusion
convolutional long short term memory (C-LSTM) neural networks. This readme gives a brief overview of the project and the code in 
this repository, see the accompanying 
[report](https://github.com/justinbt1/Multimodal-Document-Classification/blob/main/report/) for the full 
details. Note this is experimental code for my masters project, I would advise against running any of it in production.

### Requirements
All Python dependencies for the project can be installed by building a conda environment from the 
[environment.yml](https://github.com/justinbt1/Multimodal-Document-Classification/blob/main/environment.yml) file. 

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
