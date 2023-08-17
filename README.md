# LIBS_automated_mineralogy

<i> This module is developed since September 2022 by Simon Nachtergaele (under supervision of Prof. Eric Pirard) in the scope of the LIBS-SCReeN project.
This project benefited from stimulating discussions with Christian Burlet, 
Trystan Gailly, Benjamin Delvoye, Dominik Zians and others from the LIBS-SCReeN project and GeMMe group</i>

<h1> LIBS </h1>

<h3> What is LIBS? </h3>
LIBS is a technique that requires the formation and subsequent breakdown of a plasma. 
A decomposing plasma emits a variety of light rays, that are captured with glass fibre and transported to a spectrometer. 
The resulting hyperspectral data is composed of individual spectra that are 16376-dimensional. 

<h3> Goal of the research </h3>

The goal of the LIBS automated mineralogy is to create a data analysis pipeline for these purposes:
- Major, minor and trace element visualisation in drill cores
- Automated mineralogy for drill cores using different (supervised) classification algorithms for some minerals of interest

<h3> General strategy of LIBS core scanning </h3>

Input data:
- Large hypercubes of data (.nc files) acquired by multiple spectrometers (set-up of Geological Survey of Belgium)
- Pickled files that specify the wavelength for each channel
- A so-called 'ROI-file' that defines the wavelength of interest (region of interest) for each chemical element (multiple times!)
- A training dataset (.csv file) with data from different minerals (120 shots per pixel)

Signal preprocessing:
- Standard normal variate (SNV) transformation is beneficial to the subsequent classification accuracy

The LIBS data classification pipeline uses 2 algorithms to automatically classify large hypercubes:
- using fully connected neural networks (PyTorch framework)
- using the Spectral Angle Mapper technique (SAM)

File explanation:
- The main application is called <b> mineral_classification_app.py </b>
- The functions that the main application uses are stored into the <b> mineral_classification_functions </b>
- The neural network architecture is specified in the script <b> mineral_classification_nn.py </b>
- The training dataset is made with the script <b> mineral_classification_database_builder.py </b>
- The folder libs_GUI_gsb contains the GUI that has been used during LIBS acquisition @ Geological Survey of Belgium

<h1> VNIR-SWIR-TIR-Raman </h1>

<h3> What is SWIR? </h3>
SWIR stands for short wave infrared. It is a technique used for drill core scanning.

<h3> Goal of the research </h3>
The goal of the research is to identify minerals using SWIR. The main minerals that can be found are the hydroxyl- or carbonate-group bearing minerals.
It is a pixel-based approach, aiming to classify each pixel in a drill core.

<h3> General strategy of SWIR core scanning </h3>
Following its pixel-based approach, it is necessary to compare each spectrum 
(with ~286 values) to the reference spectra. Standardization of the signal is absolutely necessary, such as previous 
calibration on black reference (i.e. closed shutter) and white reference material (e.g. spectralon).
The classification technique used is the <b> spectral angle mapper </b> algorithm. This algorithm is also 
used for the LIBS classification. The database of infrared reference spectra (including SWIR and TIR) used as 
reference material comes from jpl (NASA, speclib) and ESA.

<h3> Explanation on the file structure </h3>
<h4> <b> Scripts </b> </h4>

- The <b> SWIR_core_scan_visualiser_and_mineral_classifier.py </b> script is the script 
that is able to do mineral classification with VNIR, SWIR and MWIR data from six mineral files (.txt) 
hidden in Data/Hyperspectral_dataset_0000/speclib/SWIR/Six_Minerals
-  The <b> SWIR_functions_data_processing.py </b> includes many functions that are necessary for the data processing
and classification of the individual pixels into minerals. 
- The <b> swir_converter_from_raw_data.py </b> script is used to convert the raw data to <i> npy </i> format
- The <b> SWIR_peak_finder.py </b> is a prototype script used to find back peak locations and heights.
- The <b> SWIR_alteration_zone_finder.py </b> script is a prototype for retrieving information about the mineral alteration zone.

<h4> <b> Data </b> </h4>
This folder includes multiple other folders that contain the Speclib dataset of NASA 
and a particular folder where only 3 .txt files are stored for which a RGB image will be made.


<h1> SEM </h1>
<h3> What is SEM? </h3>
Scanning electron microscopy is a common technique to find out the mineralogy/chemistry of a given rock sample.
The technique serves as a kind of reference technique for automatd mineralogy. 

<h3> Goal of the research </h3>
The goal of the research was to look into a way to deal with chemistry variations in minerals induced by noise effects from the interaction with the surrounding resin.
