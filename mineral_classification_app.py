'''
The idea of this script is to classify minerals with different techniques:
 - Fully Connected Neural Networks
 - SAM
 Using a ground-truth dataset with different minerals
 Using several classification techniques
 Author: Simon Nachtergaele
'''

"""
To try:
- Incorporate element ratio's in input file
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import seaborn as sn
from sklearn.metrics import confusion_matrix
import datetime
from pickit.utils.spectroscopy.hyperimage import *


# Import the files where our functions are located
import mineral_classification_functions as mc_functions
import mineral_classification_nn as mc_nn

"""
Settings
"""

# Script configuration. Set True to enable. Set False to disable
show_spectrum_pixel_bool = False
uncertainty_quantification = True
certainty_map = True
make_confusion_matrix = True  # bugging
make_composite_image = True
train_model = True
save_model = False
show_unsupervised_learning_image = False
use_snv = True
make_rgb = False

# Choose your algorithm
# classification_algo = "SAM"
classification_algo = "NN"
# classification_algo = "MAX"

# Set parameters for the denoising
kernel_size = 3
kernel_denoising = False #True
# kernel_shape = 'cross'
kernel_shape = 'box'
# kernel_algo = 'median' # Take the most abundant of the surrounding area
kernel_algo = "removeisolatednoise" # If the color pixel is isolated, replace it by the most abundant

# Input settings
mineral_dict = {0: 'MgCa-phase', 1: 'Ca-phase', 2: 'Noise', 3: 'Zn-phase', 4: 'Pb-phase', 5: 'Si-phase', 6: 'Ba-phase', 7: "Fe-phase"} # 7: 'T+B'}

# mineral_colors_dict = {0:'darkgreen', 1:'green', 2:'white', 3:'purple', 4:'red', 5:'black', 6: 'pink', 7: 'yellow'}
mineral_colors_dict = {0:'blue', 1:'aquamarine', 2:'black', 3:'red', 4:'silver', 5:'yellow', 6: 'green', 7: 'orange'}

mineral_list = list(mineral_dict.values())
colors = list(mineral_colors_dict.values())
print(mineral_list)
print('colors', colors)
number_of_minerals = len(mineral_list)
use_lr_scheduler = False # used to be True
pixel_coordinate = 'x16/y1' # first number = vertical axis
show_elemental_maps_boolean = True
pixel_size = 40 #25
number_of_chunks = 10

st = datetime.datetime.now()

# Training dataset
path = r"C:\Users\simon\PycharmProjects\automated-mineralogy\libs\Mineralmaps"

training_dataset_filename = "training_database_8_minerals_03July2023.csv"
shots_per_mineral = 120
validation_fraction = 0.20 #0.20

# Test Datasets
sample = "NB35_1uint.nc" # 160 heigh, 80 wide #1.726*10^3 average intensity?
# sample = "NB35_2_50x100uint.nc" #
# sample = "NB35_2_detailuint.nc"
# sample = "NB32_1uint.nc" # 160 heigh, 80 wide # 1.329e+03
# sample = "NB32_2uint.nc" # 160, 110 # SAM: 2min30s,
# sample = "NB32_3_2023-02-1517-25-31.nc" # 400 height, 110 width,
# sample = "NB32_42023-02-1512-43-44.nc" # 100 height, 160 wide
# sample = "NB32_5_2023-02-16_10-11-58.nc" # 100 height, 300 wide
# sample = "NB32_6_2023-02-1514-30-03.nc" # 110 height, 300 width: use TRIM for trimming 20%

# sample = "vms2023-04-04_12-02-47.nc"
# sample = "Belgium_mvt2023-04-04_14-06-33.nc"
# sample = "Sphalerite AND1 500 x 300.nc" # coordinates bands from 195.5 to 423.5 nm (8188 bands).
# sample = "Sphalerite 400 x 200.nc" # coordinates bands from 195.5 to 423.5 nm (8188 bands). #error: AttributeError: 'DataArray' object has no attribute 'mapping'
# sample = "Schm01uint.nc" # 11379 bands from , x is 388 and y is 107; error: AttributeError: 'DataArray' object has no attribute 'mapping'
# sample = "Granite_minilibs_600x400.nc" # 2048 bands from , x is 400 and y is 600 # error: AttributeError: 'DataArray' object has no attribute 'mapping'
# sample = "mapping_psilo_100x100.nc" # 2048 bands from , x is 100 and y is 100. Problem with the "mapping Data Variable"
# sample = "hassan_spd_laura2023-05-23_10-42-39.nc" #18k bands: training dataset is not suitable for this
# sample = 'NW-11-22-062023-05-23_11-31-24.nc'
# sample = 'NW-11-22-052023-05-23_13-37-21.nc'


# sample = 'Ivan_BRD-27-22-13_A_only16376.nc'  # done, sent elemental map
# sample = 'Ivan_BRD-27-22-13_B_only16376.nc' # done, sent elemental map
# sample = "NW-11-22-052023-05-23_13-37-21_only_16376.nc" # done, sent elemental map,
# sample = "NW-11-22-062023-05-23_11-31-24_only_16376.nc" # done, sent elemental map,
# sample = 'Ivan_BR_49_19_162023-06-23_12-15-52_only_16376.nc' # done, sent elemental map,
# sample = 'Ivan_BR_11_22_162023-06-23_15-09-17_only_16376.nc' # done, sent elemental map
# sample = 'Ivan_BRD_27_22_202023-06-23_13-32-37_only_16376.nc' # done, sent elemental map in second attempt
# sample = "Ivan_BRD_27_22_212023-06-23_11-28-24_only_16376.nc" #done, sent elemental map in second attempt

# Licia Santoro samples
# sample = "Santoro1232023-06-28_15-03-08_only_16376.nc" # Error in bands = True
# sample = "Santoro1342023-06-26_12-03-18_only_16376.nc" # Error in bands = True (0.05, 0.65), (0.20-0.80)
# sample = "Santoro1372023-06-28_12-15-59_only_16376.nc" # Error in bands = True
# sample = "SantoroAp0022023-06-26_14-59-15_only_16376.nc" # Error in bands = True
# sample = "SantoroMP462023-06-28_09-34-52_only_16376.nc" # Error in bands = True

# Element list for Ivan
# element_visualized = ['Zn', 'Pb', 'Si', 'Ag', 'Ni', 'Ca', 'Hg', 'Fe', 'As', 'Cu', 'Mg', 'Cd', 'P', 'Ge', 'Sb', 'Ni'] #['Zn', 'Pb', 'Ca', 'Mg', 'Fe', 'Si', 'Cu', 'Hg', 'Cd', 'Ba', 'P']
element_visualized = ['Ca', 'Mg', 'Zn', 'Pb', 'Si', 'As', 'Cu', 'Ni', 'Cd']
# Element list for Santoro # Santoro: elements in sphalerite: Cr, Co, Tl, S, Mn, Fe, Zn, Cu, Ga, Ge, Mo, As, Se, Ag, Cd, In, Pb, and Bi contents
# element_visualized = ['Ca', 'Zn', 'Fe', 'Pb', 'Si', 'Ti', 'Cr', 'Co', 'Tl', 'Mn', 'Cu', 'Ga', 'Ge', 'Mo', 'As', 'Se',
#                       'Ag', 'Cd', 'In', 'Bi', 'Hg']

"""
Hyperparameters for deep learning
"""

num_epochs = number_of_minerals * 15 #15 used normally, now only 1 for debugging

if classification_algo == "NN":
    learning_rate = 5*10**-5 #2.5*10**-5
else:
    learning_rate = "NA"

batch_size = 128 #64

# Dropout parameters for fully connected and convolutional neural network
dropout_fc = 0.20
dropout_conv = 0.2

step_lr = 10
gamma_lr = 0.8


"""
Load training data into df_tr (=df used for training)
"""

# Read training data
os.chdir(path)
print('Read dataframe now')
print(training_dataset_filename[-4:])
if training_dataset_filename[-4:] == "xlsx":
    print('Excel file found')
    df_tr = pd.read_excel(training_dataset_filename)
elif training_dataset_filename[-3:] == "csv":
    print('Csv file with training data found')
    df_tr = pd.read_csv(training_dataset_filename, sep=",")
else:
    print('No appropriate fileformat found')

print('df_tr.head()', df_tr.head())


print('get current working directory', os.getcwd())
dictionary_filename = 'bands_wavelength_dict_4_spectrometers.pkl'
os.chdir(r'C:\Users\simon\PycharmProjects\automated-mineralogy\libs\Datacubes\Input')
error_in_bands = False

print('Now there needs to be a mistake repaired')
# Load the dictionary from the file
channels_to_wavelengths_dict = mc_functions.load_dictionary(dictionary_filename)
print('channels_to_wavelengths_dict', channels_to_wavelengths_dict)
bands = list(channels_to_wavelengths_dict.values())
print('type bands', type(bands))
bands = np.array(bands)
print('bands', bands) # dict_values
print('---')

# If the roi (elemental maps!) needs to be applied to the training dataset
loc_roi_file = r"C:\Users\simon\PycharmProjects\automated-mineralogy\libs\Datacubes\Input\roi_wavelengths_v31072023.csv"
df_tr, el_list, df_row_names_training_dataset = mc_functions.use_roi_on_libs_training_dataset(df_tr, loc_roi_file, bands, use_snv)
number_of_elements = len(el_list)

print('channels to wavelength array', bands)
print('Loaded dataframe for the training data:')
print(df_tr.head())

# Take the classes and make a dictionary for the mineral names and class_int's
unique_classes_int = df_tr['Class_int'].unique()

# Add the column value for name
df_tr.columns = df_tr.columns.str.replace('Unnamed: 0', 'Name')
print('after name replacement')

# Gather the training spectra and store in X
print('df_tr.shape', df_tr.shape)
print('df_tr', df_tr)
X_training_database_df = df_tr.iloc[:, 1:]
print('X_training_database_df', X_training_database_df)
X_training_database = pd.DataFrame(df_row_names_training_dataset).merge(X_training_database_df,
                                                   left_index = True,
                                                   right_index = True,
                                                   how='inner')
# print('X_training.shape', X_training_database.shape)
print('X_training_database.head()', X_training_database.head())

# Define the labels
y = df_tr['Class_int'].values
print('y', y)

scaler = StandardScaler()

X_train, X_validation, y_train, y_validation = train_test_split(X_training_database, y, test_size=validation_fraction, random_state=42)

X_train_coordinates = X_train.iloc[:, 0]
X_train_coordinates = X_train_coordinates.to_numpy()
# print('X_train_coordinates', X_train_coordinates)
print(type(X_train_coordinates))
X_train = X_train.iloc[:, 1:]
X_train = X_train.to_numpy()
# print('X_train', X_train)
print(type(X_train))

X_validation_coordinates = X_validation.iloc[:, 0]
X_validation_coordinates = X_validation_coordinates.to_numpy()
print('X_validation_coordinates', X_validation_coordinates)
print(type(X_validation_coordinates))
X_validation = X_validation.iloc[:, 1:]
X_validation = X_validation.to_numpy()
print('X_validation', X_validation)
print(type(X_validation))

# X_validation_coordinates =
# X_validation_ =


print('X_train')
print(X_train)

print('X_validation')
print(X_validation)

print('y_train')
print(y_train)

print('y_validation')
print(y_validation)

"""
Enabling GPU/CUDA support: store the training dataset as float tensors to make them be able to work on the GPU unit
"""

print('Test GPU support')
if torch.cuda.is_available() == True: #: Returns True if CUDA is supported by your system, else False
    print('Cuda is available')
    print(torch.cuda.current_device()) # Returns ID of current device
    cuda_id = torch.cuda.current_device()
    print(cuda_id)
    print(torch.cuda.get_device_name(cuda_id))

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

print('Available device is')
print(device)

# Make floattensor from the to put it on the GPU
X_train = torch.FloatTensor(X_train)
X_train = X_train.to(torch.device(device))
X_validation = torch.FloatTensor(X_validation)
X_validation = X_validation.to(torch.device(device))
y_train = torch.LongTensor(y_train)
y_train = y_train.to(torch.device(device))
y_validation = torch.LongTensor(y_validation)
y_validation = y_validation.to(torch.device(device))

"""
Show the average spectra of each class in a plot 
"""

mc_functions.show_average_spectra_for_each_class_of_training_database(df_tr,
                                                                      X_training_database,
                                                                      number_of_minerals,
                                                                      shots_per_mineral,
                                                                      colors,
                                                                      number_of_elements,
                                                                      mineral_list,
                                                                      mineral_dict)

"""
Training/loading of the network
"""

# If the setting is chosen that the model should be trained
if train_model:
    print('train_model')
    if classification_algo == "NN":
        model = mc_nn.NN(number_of_elements, dropout_fc, number_of_minerals)
        # model = NeuralNetworkClassificationModel(number_of_elements, output_dim) #, number_of_minerals)
        model = model.to(torch.device(device))

# If the setting was chosen to not use such a model
else:
    # Load models
    print('load models in stead of training a model')
    if classification_algo == "NN":
        model_path = "trained_modelNN2023-03-24 11_57_42_751489.pth"
        model = torch.load(model_path)
        print('model', model)
        model = mc_nn.NN(number_of_elements, dropout_fc, number_of_minerals)

    else:
        pass

    if classification_algo == "CNN" or classification_algo == "NN":

        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        model = model.to(torch.device(device))
        print('load model please')

    else:
        pass


train_losses = np.zeros(num_epochs)
validation_losses = np.zeros(num_epochs)

# Make tensor from the model so that it can be run on GPU
# if not classification_algo == "SAM":
#     model = model.to(device)
#     model = model.to(torch.device(device)) # Try putting it on the GPU
#
#     # Start to train network
#     criterion = nn.CrossEntropyLoss()
#     print('model.paramters()', model.parameters())
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     if use_lr_scheduler:
#         scheduler = StepLR(optimizer, step_size=step_lr, gamma=gamma_lr)

# Define some variables that need to be created anyway
predictions_train = []
predictions_test = []

# The train_model needs to be automatically set to False if it's using SAM
if classification_algo == "SAM" or classification_algo == "MAX":
    train_model = False
else:
    pass
print("classification_algo is ", classification_algo)

# If a model needs to be trained (for CNN and NN)
if train_model == True:
    print("Start training network")
    if classification_algo == "NN":
        criterion = nn.CrossEntropyLoss()
        print('model.paramters()', model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        print('Start training NN')
        print('model', model)
        print('optimizer', optimizer)
        print('criterion', criterion)
        print('X_train', X_train)
        print('y_train', y_train)
        print('X_validation', X_validation)
        print('y_validation', y_validation)

        # Define training dataset
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print('train_dataset_loader before the training', train_dataset_loader)

        # Define validation dataset
        validation_dataset = torch.utils.data.TensorDataset(X_validation, y_validation)
        validation_dataset_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

        model = mc_functions.train_network_nn(model,
                                              optimizer,
                                              criterion,
                                              train_dataset_loader,
                                              num_epochs,
                                              train_losses,
                                              validation_losses,
                                              validation_dataset_loader,
                                              use_lr_scheduler,
                                              learning_rate,
                                              step_lr,
                                              gamma_lr,
                                              batch_size,
                                              dropout_fc)

    else:
        pass

    plt.figure(figsize=(20,20))
    plt.plot(train_losses, label='train loss')
    plt.plot(validation_losses, label='validation loss')
    plt.axhline(y=0, color='r', linestyle='dashed')

    plt.legend()
    plt.show()

    with torch.no_grad():
        print('no_grad')

        # Compute statistics for the model on the training dataset
        X_train_testNN = list()
        for data in enumerate(X_train):
            data = data[1]
            data = torch.unsqueeze(data, dim=0)
            data = torch.unsqueeze(data, dim=0)
            predictions_train = model(data)
            X_train_testNN.append(predictions_train)
        predictions_train = torch.cat(X_train_testNN, dim=0)

        # Compute statistics for the X_test dataset
        X_test_testNN = list()
        for data in enumerate(X_validation):
            data = data[1]
            data = torch.unsqueeze(data, dim=0)
            data = torch.unsqueeze(data, dim=0)
            predictions_test = model(data)
            X_test_testNN.append(predictions_test)
        predictions_test = torch.cat(X_test_testNN, dim=0)

    train_acc = mc_functions.get_accuracy_multiclass(predictions_train, y_train)
    validation_acc = mc_functions.get_accuracy_multiclass(predictions_test, y_validation)

    print(f"Training Accuracy: {round(train_acc*100,3)}")
    print(f"Validation Accuracy: {round(validation_acc*100,3)}")


"""
Prepare some lists for testing the classification technique on unknown data
"""

# Make empty lists FOR THE WHOLE CORE
class_list = list()
coords_list = list()
certainty_list = list()
coords_list = list()

nc = NCHyperImage(sample, load_into_RAM=False)
figure_height = nc.dataset.dims['x'] # 160
figure_width = nc.dataset.dims['y'] # 80
figure_nb_pixels = figure_height * figure_width # 160*80 = 12800
chunk_size = 2000

print('ds_array_total')
ds_array_total = np.empty((figure_width, figure_height, number_of_elements))
print('ds_array_total.shape', ds_array_total.shape) # (160,80,51)

for index, chunk in enumerate(nc.get_chunks_np(nbr_chunks=number_of_chunks)):

    # Transform the xarray to a np.ndarray
    chunk_ndarray = chunk.values

    # Transpose
    chunk_ndarray = np.transpose(chunk_ndarray, (1, 0, 2))
    print('chunk_ndarray.shape', chunk_ndarray.shape) #(16, 80, 51)

    # Make empty list
    coords_list_run = list()

    # Fill up the coords_list
    for i in range(chunk_ndarray.shape[0]):
        print('i is...', i)
        previous_distance = int(round(index * (figure_width / number_of_chunks)))
        print('previous distance', previous_distance)

        hor = i + previous_distance

        for vert in range(chunk_ndarray.shape[1]):
            coord = list()
            coord.append(hor)
            coord.append(vert)
            coords_list.append(coord)
            coords_list_run.append(coord)


    print('coords_list', coords_list)
    print('coords_list_run', coords_list_run)

    chunk_vertical_location_upper = int(previous_distance)
    chunk_vertical_location_lower = int(previous_distance + figure_width/number_of_chunks)

    channels = chunk.bands
    chunk_ndarray, el_list = mc_functions.use_roi_on_loaded_nc(chunk_ndarray, loc_roi_file, number_of_elements, bands)
    print('chunk_ndarray.shape', chunk_ndarray.shape)


    # Add the chunk_ndarray to the ds_array_total
    ds_array_total[chunk_vertical_location_upper: chunk_vertical_location_lower, 0:figure_width, : ] = chunk_ndarray # (45, 20, 51)
    print('And now we go to the classification part')

    # Transform so that it is a 2D entity through which we can easily iterate
    chunk_ndarray_transformed = chunk_ndarray.view().reshape(chunk_ndarray.shape[0] * chunk_ndarray.shape[1], chunk_ndarray.shape[2])
    X_unknown = torch.FloatTensor(chunk_ndarray_transformed)
    X_unknown = X_unknown.to(torch.device(device))  # Try putting it on the GPU

    # Perform classification using NN
    if classification_algo == "NN":
        class_list, certainty_list = mc_functions.classify_data_using_nn(coords_list_run, X_unknown, model,
                                                                         uncertainty_quantification, class_list,
                                                                         certainty_list)
    # Perform classification using SAM
    elif classification_algo == "SAM":
        class_list, certainty_list = mc_functions.classify_data_using_sam(coords_list_run, X_unknown, uncertainty_quantification, class_list, certainty_list, number_of_minerals, X_training_database, shots_per_mineral)

    elif classification_algo == "MAX":
        class_list, certainty_list = mc_functions.classify_data_using_max(coords_list_run, X_unknown,
                                            uncertainty_quantification, class_list, certainty_list, number_of_minerals)


"""
Postprocessing after the image map is made
"""
if kernel_denoising:
    class_list = mc_functions.kernel_denoising(coords_list, class_list, figure_width, figure_height, kernel_size, kernel_shape, kernel_algo)

"""
Provide the user with the most abundant minerals
"""

mineral_mapping_results = mc_functions.mineral_counter(class_list, number_of_minerals, mineral_dict)
print('mineral_mapping_results', mineral_mapping_results)

"""
Make map of the classified dataset 
"""

# figure_width = ds.shape[0]
print('figure_width', figure_width)
# figure_height = ds.shape[1]
print('figure_height', figure_height)
figname = sample[:-3]

fig, ax = plt.subplots(figsize=(figure_width, figure_height))
print('Make mineral map now')
# mc_functions.make_colored_mineral_map()
mineral_map = mc_functions.make_colored_mineral_map(classification_algo,
                                                                 learning_rate,
                                                                 num_epochs,
                                                                 coords_list,
                                                                 mineral_dict,
                                                                 mineral_colors_dict,
                                                                 class_list,
                                                                 training_dataset_filename,
                                                                 certainty_list,
                                                                 uncertainty_quantification,
                                                                 figname,
                                                                 figure_width,
                                                                 figure_height,
                                                                pixel_size,
                                                                show_spectrum_pixel_bool,
                                                                pixel_coordinate,
                                                                kernel_denoising,
                                                                kernel_size,
                                                                kernel_shape, kernel_algo)
# Save the figure with legend and title somewhere
mineral_map.show()
mineral_map.savefig(figname[:-3] + 'with_legend.png')
print('image saved at', os.getcwd())
mineral_map.show()
"""
Save NN model
"""

if save_model:
    currentDateAndTime = str(datetime.datetime.now())
    currentDateAndTime = currentDateAndTime.replace(".", "_")
    currentDateAndTime = currentDateAndTime.replace(":", "_")

    name = "trained_model"+str(classification_algo)+str(currentDateAndTime)+str('.pth')
    torch.save(model.state_dict(), name)
    print('model saved with name', name)
else:
    pass

"""
Show the spectrum of a pixel
"""

if show_spectrum_pixel_bool:
    mc_functions.show_spectrum_pixel(pixel_coordinate, ds_array_total, figure_width, figure_height, el_list)

"""
Confusion matrix
"""

if classification_algo != "SAM" and classification_algo != "MAX":
    if make_confusion_matrix:
        if isinstance(predictions_test, torch.Tensor) == False:
            print('Convert predictions_test now')
            predictions_test = torch.tensor(predictions_test)

        print('predictions_test after exp', predictions_test)
        y_pred = torch.max(predictions_test, 1) # original was 1
        print('y_pred after max', y_pred)
        y_pred = y_pred[0] # original: y_pred[1]
        print('y_pred [1]', y_pred[1])

        y_pred = y_pred.argmax(dim=1)
        print('y_pred after argmax', y_pred)
        y_pred = y_pred.data.cpu().numpy() # error here when using NN classication
        print('y_pred', y_pred)# prediction

        y_true = y_validation.cpu().detach().numpy() # truth labels torch_tensor.cpu().detach().numpy()
        print('y_true', y_true)
        cf_matrix = confusion_matrix(y_true, y_pred)
        print('cf_matrix', cf_matrix)

        classes = mineral_dict.values()
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1),
                             index = [i for i in classes],
                             columns = [i for i in classes])
        print('df_cm', df_cm)
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True)
        plt.show()


"""
Display the wrongly classified spectra
"""
if classification_algo == 'NN':
    indexes_wrongly_classified_data = list()
    for index, value in enumerate(y_pred):
        if value == y_true[index]:
            # print('Well classfied')
            pass
        else:
            indexes_wrongly_classified_data.append(index)
            print('Wrongly classified')

    for index in indexes_wrongly_classified_data:
        fig1 = plt.figure(figsize=(16, 8))
        ax1 = fig1.add_subplot(1, 1, 1)
        spectrum = X_validation[index].cpu()
        print('spectrum', spectrum)
        label_true = y_validation[index]
        label_predicted = y_pred[index]
        print('label_true', label_true, 'label predicted', label_predicted)
        mineral_name_true = list(mineral_dict.values())[label_true]
        mineral_name_predicted = list(mineral_dict.values())[label_predicted]
        print('mineral_name_true', mineral_name_true, 'mineral_name_predicted', mineral_name_predicted)
        ax1.plot(el_list, spectrum)
        title = 'predicted: '+str(mineral_name_predicted) + ', true label: ' + str(mineral_name_true) + str(' ') + str(X_validation_coordinates[index])
        ax1.set_title(title)
        fig1.show()

"""
Show elemental maps
"""
use_snv_for_element_visualized = True

if show_elemental_maps_boolean:
    # Make new folder
    print('Make new directory for elemental map storage')
    print(os.getcwd())
    os.chdir(r"C:\Users\simon\PycharmProjects\automated-mineralogy\libs\Datacubes\Output")
    try:
        os.mkdir(str(sample[:-3]))
    except:
        pass
    path_element_maps = os.path.join(r"C:\Users\simon\PycharmProjects\automated-mineralogy\libs\Datacubes\Output", str(sample[:-3]))
    os.chdir(path_element_maps)
    print(os.getcwd())
    print('ds_array_totals.shape is', ds_array_total.shape)

    if use_snv_for_element_visualized:
        ds_array_total_snv = np.zeros_like(ds_array_total)
        print('ds_array_total.shape', ds_array_total.shape)
        for y in range(ds_array_total.shape[0]):
            for x in range(ds_array_total.shape[1]):
                ds_array_total_snv[y, x, :] = mc_functions.snv(ds_array_total[y, x, :])
        ds_array_total = ds_array_total_snv

    for element in element_visualized:
        index_element = el_list.index(element)
        print('index_element', index_element)
        print('min', np.min(ds_array_total[:, :, index_element]))
        print('max', np.max(ds_array_total[:, :, index_element]))
        plt.imshow(ds_array_total[:, :, index_element], cmap='seismic') #, vmax=10000, vmin=0)
        plt.colorbar()
        plt.title(el_list[index_element])
        plt.tight_layout()
        filename_elemental_map = str(sample) + str(element) + str('.jpeg')
        plt.savefig(filename_elemental_map)
        plt.show()
        plt.close()


"""
Make composite image
"""

if make_composite_image == True:

    columns = 3
    rows = 3 #5

    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(12, 15)) #15,21: width, height

    for i, ax_row in enumerate(axes):
        print('row number', i)
        for j, ax in enumerate(ax_row):
            print('column', j)

            # Compute index of the element
            index_element = i*columns+j
            print('index_element', index_element)

            # Find element
            print('element_visualized', element_visualized)
            element = element_visualized[index_element]

            # Find index in hypercube
            index_element_in_hypercube = el_list.index(element)
            print('index_element_in_hypercube', index_element_in_hypercube)

            # Select a slice of the hypercube
            slice_data = ds_array_total[:, :, index_element_in_hypercube]

            # Plot the slice
            img = ax.imshow(slice_data, cmap='seismic')
            # ax.invert_yaxis()
            ax.set_title(element)

            # Add color bar
            cbar = fig.colorbar(img, ax=ax, shrink=0.6)
            print('plot composite figure now...')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the figure
    filename_composite = str(sample)+str(' composite plot.png')
    print('current working directory', os.getcwd())
    plt.savefig(filename_composite)
    plt.show()
    plt.close()


"""
Make average and stdev of entire image
"""

mc_functions.compute_mean_and_stdev_spectrum_from_hyperspectral_image(ds_array_total, el_list)


"""
Make RGB image from different elements
"""


if make_rgb:
    element_rgb = ['Zn', 'Ba', 'Hg']

    element_rgb_idx = list()
    for index, value in enumerate(element_rgb):
        index_element = el_list.index(value)
        element_rgb_idx.append(index_element)

    # Make empty image
    rgb_image = np.zeros((ds_array_total.shape[0], ds_array_total.shape[1], 3))

    for index, value in enumerate(element_rgb_idx):
        slice_ds = ds_array_total[:, :, value]
        slice_normalized = (slice_ds - np.min(slice_ds)) / (np.max(slice_ds) - np.min(slice_ds))
        rgb_image[:, :, index] = slice_normalized

    plt.imshow(rgb_image)
    plt.title(element_rgb)
    plt.show()
    plt.close()


# Make correlation coefficiënt matrix
# Reshape the array to 2D with the 3rd dimension as rows
reshaped_data = np.reshape(ds_array_total[:, :, :number_of_elements], (ds_array_total[:, :, :number_of_elements].shape[0] * ds_array_total[:, :, :number_of_elements].shape[1], ds_array_total[:, :, :number_of_elements].shape[2]))

# Calculate the correlation matrix
correlation_matrix = np.corrcoef(reshaped_data, rowvar=False)

print('correlation matrix', correlation_matrix)

# Plot the correlation matrix as an image
plt.imshow(correlation_matrix, cmap='coolwarm', vmin=0, vmax=1)
plt.colorbar()

# Set the tick labels
labels = el_list[:number_of_elements]
plt.xticks(range(len(labels)), labels, fontsize=5)
plt.yticks(range(len(labels)), labels, fontsize=5)

# Add a title
plt.title('Correlation Matrix')

# Display the image
plt.show()
plt.close()

# Get the end datetime
et = datetime.datetime.now()

# Get execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

