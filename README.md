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
