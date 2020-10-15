# SleepAnalyses
Descriptive analyses of SOMNOgraph signals in different sleep stages


The folder contains 2 files,
####1. ETL.py
####2. Analysis notebook

### ETL

This file contains all classes and functions necessary for analyzing data. Two main classes are implemented for,
1. reading and analyzing data, 
2. plotting results.

### Notebook Sleep Project 

This notebook has 2 main section for 2 version of analyses; 
1. First section reads all units and concatinates them to stacked long data then calculated average firing rate and average amplitude per each individual sleep stage separately. 
2. Second section reads all units but analyzes each unit individually and then to summarize results it uses mean or median analysis. 
In both section user can easily remove part of unit by just selecting by using matrix indexing (example is given in notebook).
