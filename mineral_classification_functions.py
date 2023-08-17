"""
This script includes functions for mineral classification
Author: Simon Nachtergaele
"""

import datetime
import matplotlib.pyplot as plt
import numpy as np
import math
import collections
from matplotlib.lines import Line2D
from statistics import mode
import pandas as pd
import os
import pickle
from torch.optim.lr_scheduler import StepLR
import torch
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
import cv2

# Check if cuda support is available or not
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def load_dictionary(file_path):
    """ Loads dictionary using pickle library
    :param file_path: str
    :return: loaded pickle
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def decrease_brightness(rgb, factor):
    r, g, b = rgb
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    return r, g, b

def snv(input_data):
    """ Does SNV correction

    Parameters
    -----------
    input_data: np.ndarray
        Data

    Returns
    ----------
    SNV corrected input_data
    """

    input_data = np.asarray(input_data)

    # Define a new array and populate it with the corrected data
    data_snv = np.zeros_like(input_data)

    for i in range(data_snv.shape[0]):
        # Apply correction
        # print(input_data[i])
        # print(np.mean(input_data[i]))
        # print(np.std(input_data[i]))

        data_snv[i] = (input_data[i] - np.mean(input_data)) / np.std(input_data)
    return data_snv

def expand_array(array, factor):
    # Get the dimensions of the original array
    height, width, depth = array.shape

    # Create a new array with expanded dimensions
    expanded_array = np.zeros((height * factor, width * factor, depth), dtype=array.dtype)

    # Expand the original array along height and width dimensions
    for d in range(depth):
        for h in range(height):
            for w in range(width):
                expanded_array[h * factor:(h + 1) * factor,
                                w * factor:(w + 1) * factor, d] = array[h, w, d]

    return expanded_array

def computeSA_custom(groundTruth, recovered):
    """ Compute angle for Spectral Angle Mapper technique between the recovered and the corresponding ground-truth spectrum

    Parameters
    -----------
    groundTruth: list
    recovered: list or np.ndarray

    Returns
    --------
    spectral angle

    """

    if type(groundTruth) == list:
        # print('type is list')
        # Make arrays from input which are lists typically
        groundtruth_arr = np.array(groundTruth)
        # print('groundtruth_arr', groundtruth_arr)
        # print('groundtruth_arr.shape', groundtruth_arr.shape) # (46,)
        recovered_arr = np.array(recovered)
        # print('recovered_arr', recovered_arr)
        # print('recovered_arr.shape', recovered_arr.shape) # problem: (45,)

    elif type(groundTruth) == type(np.ndarray):
        # print('type is array')
        pass
    else:
        print('type is nor array nor list')
        pass

    assert groundtruth_arr.shape == recovered_arr.shape, \
        "Size not match for groundtruth and recovered spectral images"

    H = groundtruth_arr.shape[0] # number of channels in this case

    nom = np.sum(np.multiply(groundtruth_arr, recovered_arr), axis=0)
    denom1 = np.sqrt(np.sum(np.power(groundtruth_arr, 2), axis=0))
    denom2 = np.sqrt(np.sum(np.power(recovered_arr, 2), axis=0))
    sam = np.arccos(np.divide(nom, np.multiply(denom1, denom2)))
    sam = np.multiply(np.divide(sam, np.pi), 180)
    sam = np.divide(np.sum(sam), H)
    return sam

def kernel_denoising(coords_list, class_list, width, height, kernel_size, kernel_shape, kernel_algo):
    """ Denoise a given image using a certain kernel operation

    Parameters
    ---------
    coords_list: list
        A list of list with coordinates [x,y]
    class_list: list
        A list with integers that refer to the label
    width: int
        Figure width
    height: int
        Figure height
    kernel_size: int
        Size (in 1D) (e.g. 3 or 4 or 5) of the kernel that does the denoising operations. Currently working with only 3.
    kernel_shape: str
        Choose between 'cross' or 'box'.
    kernel_algo: str
        Kernel algorithm: choose between 'median' or 'removeisolatednoise'

    Returns
    --------
    class_list: list
        A new class_list which is denoised.

    """
    class_list_adapted = list()
    print('start denoising now')
    # Loop over the coordinate list (start with 0,0 until 0,80 and continue from 1,0 to 1,80 etc.)
    for index, value in enumerate(coords_list):

        # Take x and y
        x = float(value[0])
        y = float(value[1])

        # Make new list to store the colors of the surrounding pixels
        colors_surrounding_pixels = list()

        # If the kernel is not outside the window, do the kernel operation
        if x >= (kernel_size - 2) and x <= (width - kernel_size + 1) and y >= (kernel_size - 2) and y <= (height - kernel_size + 1): # this does not work well enough

            # Take color of the pixel in the center of the kernel
            c = class_list[index]

            if kernel_shape == 'box':  # Take colors of the 8 surrounding pixels
                c_above = class_list[index + height]
                c_right_above = class_list[index + height + 1]
                c_left_above = class_list[index + height - 1]
                c_below = class_list[index - height]
                c_right = class_list[index + 1]
                c_left = class_list[index - 1]
                c_right_below = class_list[index - height + 1]
                c_left_below = class_list[index - height - 1]

                # Store in a list
                colors_surrounding_pixels.append(c_above)
                colors_surrounding_pixels.append(c_below)
                colors_surrounding_pixels.append(c_right)
                colors_surrounding_pixels.append(c_left)
                colors_surrounding_pixels.append(c_right_above)
                colors_surrounding_pixels.append(c_left_above)
                colors_surrounding_pixels.append(c_right_below)
                colors_surrounding_pixels.append(c_left_below)

            elif kernel_shape == 'cross':
                c_above = class_list[index + height]
                c_below = class_list[index - height]
                c_right = class_list[index + 1]
                c_left = class_list[index - 1]

                # Store in a list
                colors_surrounding_pixels.append(c_above)
                colors_surrounding_pixels.append(c_below)
                colors_surrounding_pixels.append(c_right)
                colors_surrounding_pixels.append(c_left)

            else:
                pass

            # Take the most abundant color
            class_most_abundant = mode(colors_surrounding_pixels)

            if kernel_algo == "median":
                class_new = class_most_abundant
                class_list_adapted.append(class_new)

            elif kernel_algo == "removeisolatednoise":
                # If the color of the pixel is NOT found in the surrounding pixels
                if not c in colors_surrounding_pixels:
                    # print('removeisolatednoise')
                    class_new = class_most_abundant
                    class_list_adapted.append(class_new)

                # If the color is found in one of the neighbouring pixels
                else:
                    class_list_adapted.append(c)
            else: # when no kernel algoritm is specified
                class_list_adapted.append(c)

        # If the kernel falls on the edge of the image
        else:
            # print('kernel outside', index)
            class_list_adapted.append(class_list[index])

    # Overwrite the class_list to update the mineral names (after kernel usage)
    class_list = class_list_adapted

    # Provide the user with some information on the noise correction technique
    # Loop over the coordinate list and count the number of pixels that have been changed
    pixels_changed_after_noise_correction = 0
    for index, value in enumerate(class_list):
        if value != class_list_adapted[index]:
            pixels_changed_after_noise_correction += 1
        else:
            pass

    print('pixels changed after noise correction', pixels_changed_after_noise_correction)

    return class_list

def make_colored_mineral_map(classification_algo, learning_rate, num_epochs, coords_list, mineral_dict, mineral_colors_dict,
                             class_list, training_dataset_filename, certainty_list,
                             uncertainty_quantification, figname, figure_width, figure_height,
                             pixel_size, show_spectrum_pixel_bool, pixel_coordinate,
                            kernel_denoising, kernel_size, kernel_shape, kernel_algo):

    """ Makes a colored map from the classified LIBS dataset

    :param classification_algo: str
        Classification algorithm such as NN or SAM
    :param learning_rate: float
        Learning rate of the NN
    :param num_epochs: int
        Number of epochs that the NN is trained
    :param coords_list: list
        List of lists with coordinates, e.g. [[0, 0], [0,1], etc. ]
    :param mineral_dict: dict
        Dictionary with
    :param mineral_colors_dict: dict
        Dictionary with the explanation of the mineral int and their colors, e.g. {0:'blue', 1:'aquamarine', 2:'black', ...
    :param class_list: list
        List with the classified labels made by the classification method.
    :param training_dataset_filename: str
        File with the training dataset
    :param certainty_list: list
        List with certainty values
    :param uncertainty_quantification: Bool
        Boolean value indicating the uncertainty_quantification
    :param figname: str
        Figure name (normally nc file + .jpg)
    :param figure_width: int
        Figure width
    :param figure_height: int
        Figure height
    :param pixel_size: int
        Pixel size for the image
    :param pixel_coordinate: str
        Pixel coordinate written in the style of: 'x0/y36'
    :return: figure
    """

    colors = list(mineral_colors_dict.values())
    mineral_list = mineral_dict.values()
    width_height_ratio = figure_width/figure_height # For 160 and 80, this is 2
    print('width_height_ratio', width_height_ratio)
    mean_dimension = figure_height*0.5+figure_width*0.5
    size = 10 * (mean_dimension/150)
    if figure_width > figure_height:
        print('size condition A')
        fig, ax = plt.subplots(
            figsize=(size, round(size*width_height_ratio)+0.1*size))  # specify width and height, respectively, (8,14) for NB35_1
    else:
        print('size condition B')
        fig, ax = plt.subplots(
            figsize=(size, round(size*width_height_ratio)+0.1*size))

    x_list = list()
    y_list = list()
    c_list = list()

    # Calculate certainty
    # Rescale certainty_list on axis 0 to 1
    certainty_list_scaled = list()
    max_list = max(certainty_list)
    min_list = min(certainty_list)
    print('certainty list:', certainty_list)

    # Compute the alpha weights for the pixels
    if classification_algo == "SAM":
        if uncertainty_quantification:
            for certainty in certainty_list:
                c = (certainty - min_list) / (max_list - min_list)  # zi = (xi – min(x)) / (max(x) – min(x))
                certainty_list_scaled.append(round(math.sqrt(c), 3))
        else:
            certainty_list_scaled = certainty_list

        print('certainty_list_scaled', certainty_list_scaled)
    else:
        certainty_list_scaled = certainty_list

    # No uncertainty weighted image
    img = np.zeros((figure_width, figure_height, 3), dtype=np.uint8)
    img_without_uncertainty = np.zeros((figure_width, figure_height, 3), dtype=np.uint8)

    # Iterate over the coordinates list and determine the color of the pixel
    for count, coord in enumerate(coords_list):
        # Get color
        c = class_list[count]
        c = int(c)
        color = colors[c]

        # Save coordinates and append it for the color maps
        x = coord[0]
        y = coord[1]
        y_list.append(y)
        x_list.append(x)
        c_list.append(color)
        print('color', color)

        # Find certainty
        certainty = certainty_list_scaled[count]
        print('certainty', certainty)

        if color == "blue":
            img[x, y] = decrease_brightness((254, 0, 0), certainty)
            img_without_uncertainty[x, y] = (254, 0, 0)
        elif color =='green':
            img[x, y] = decrease_brightness((0, 254, 0), certainty)
            img_without_uncertainty[x, y] = (0, 254, 0)
        elif color == 'red':
            img[x, y] = decrease_brightness((0, 0, 255), certainty)
            img_without_uncertainty[x, y] = (0, 0, 255)
        elif color == 'silver':
            img[x, y] = decrease_brightness((128, 128, 128), certainty)
            img_without_uncertainty[x, y] = (128, 128, 128)
        elif color == 'black':
            img[x, y] = decrease_brightness((0, 0, 0), certainty)
            img_without_uncertainty[x, y] = (0, 0, 0)
        elif color == 'yellow':
            img[x, y] = decrease_brightness((0, 255, 255), certainty)
            img_without_uncertainty[x, y] = (0, 255, 255)
        elif color == 'orange':
            img[x, y] = decrease_brightness((0, 125, 255), certainty)
            img_without_uncertainty[x, y] = (0, 125, 125)
        elif color == 'aquamarine':
            img[x, y] = decrease_brightness((208, 224, 64), certainty)
            img_without_uncertainty[x, y] = (208, 224, 64)

        else:
            pass

    # Now add a scale on top, down, left and right
    edge_width = 2
    block_length = 20
    new_figure_height = img.shape[0] + 2 * edge_width
    new_figure_width = img.shape[1] + 2 * edge_width
    new_img = np.zeros((new_figure_height, new_figure_width, 3), dtype=np.uint8)
    print('new_img', new_img.shape)

    # First make a chess-board pattern
    for y in range(new_figure_height):
        for x in range(new_figure_width):
            # For the numbers with rest value of 10-19
            if x % block_length >= block_length*0.5:
                if y % block_length >= block_length * 0.5:
                    new_img[y, x, :] = (255, 255, 255)
            # For the numbers going from 0-9
            elif x % block_length < block_length*0.5:
                if y % block_length < 0.5 * block_length:
                    new_img[y, x, :] = (255, 255, 255)
            else:
                pass

    print('img.shape', img.shape)
    print('new figure height', new_figure_height) # 104 for test image
    print('new figure width', new_figure_width) # 49 for test image

    # Then copy the image on top of the chess board pattern
    for y in range(new_figure_height):
        for x in range(new_figure_width):
            print(' ')
            print('x', x, 'y', y)
            # If it's in the middle of the image
            if y > edge_width and y <= (new_figure_height - edge_width) and x > edge_width and x <= (new_figure_width-edge_width):
                new_img[y-1, x-1, :] = img_without_uncertainty[y-edge_width-1, x-edge_width-1, :]


    # Make the image a bit bigger
    multiplication_factor_img = 5
    img_resized = expand_array(img, multiplication_factor_img)
    img_resized_scale = expand_array(new_img, multiplication_factor_img)
    img_resized_without_uncertainty = expand_array(img_without_uncertainty, multiplication_factor_img)

    cv2.imshow('mineral map showing now...', img_resized)
    cv2.waitKey(0)
    cv2.imshow('mineral map showing now...', img_resized_scale)
    cv2.waitKey(0)
    cv2.imshow('mineral map showing now...', img_resized_without_uncertainty)
    cv2.waitKey(0)
    print('image saved at', os.getcwd())

    title = figname[:-4] + classification_algo + str('kernel=') + str(kernel_denoising) \
            + str(kernel_shape) + str(kernel_size) + str(kernel_algo) + '.png'


    os.chdir(r"C:\Users\simon\PycharmProjects\automated-mineralogy\libs\Datacubes\Output")

    cv2.imwrite(title, img_resized)

    title_scale_bar = figname[:-4] + classification_algo + str('kernel=') + str(kernel_denoising) \
            + str(kernel_shape) + str(kernel_size) + str(kernel_algo) + 'scale.png'

    cv2.imwrite(title_scale_bar, img_resized_scale)

    title_without_uncertainty = figname[:-4] + classification_algo + str('kernel=') + str(kernel_denoising) \
            + str(kernel_shape) + str(kernel_size) + str(kernel_algo) + 'no_uncertainty.png'

    cv2.imwrite(title_without_uncertainty, img_resized_without_uncertainty)

    cv2.destroyAllWindows()

    # Draw image
    ax.scatter(y_list,
               x_list,
               c=c_list,
               marker=',', # square shaped is ','
               alpha = 1, #certainty_list_scaled, # specify transparancy here.
               s = pixel_size #22 too large # specify the pixel size here # 50 for step size 2
               )


    # Insert legend
    legend_custom_lines = list()
    for index, value in enumerate(colors):
        legend_custom_lines.append(Line2D([0], [0], color = value, lw=2))
    plt.legend(legend_custom_lines, mineral_list, loc='best')

    # Insert title
    if classification_algo == 'NN':
        plt.title(str(figname)+' mineral mapping with NN' + str(learning_rate) + str(' ') + str(num_epochs) + ' epochs')
    elif classification_algo == 'SAM':
        plt.title(str(figname)+' mineral mapping with SAM')
    else:
        print('test')

    plt.xlabel('Width')
    plt.ylabel('Height')

    plt.gca().invert_yaxis()
    plt.tight_layout()

    pixel_coordinate_split = pixel_coordinate.split('/')
    x = pixel_coordinate_split[0]
    x = int(x.replace('x', ''))
    print('x is', x)
    y = pixel_coordinate_split[1]
    y = int(y.replace('y', ''))
    print('y is', y)

    if show_spectrum_pixel_bool:
        plt.plot(y, x, marker='*')

    return fig

def show_average_spectra_for_each_class_of_training_database(df_tr, X_training_database, number_of_minerals, shots_per_mineral, colors, input_dim, mineral_list, mineral_dict):
    """ Shows the average spectra for all classes + each class individually

    :param df_tr: pd.DataFrame:
        Panda Dataframe with the data of the training database file (csv file)
    :param X_training_database: pd. Dataframe
        Training dataset spectral data
    :param number_of_minerals: int
        Number of minerals
    :param shots_per_mineral: int
        Number of shots per mineral in the training database
    :param colors: list
        List with colors
    :param input_dim: int
        Size of the input data (in spectral dimension)
    :param mineral_list: list
        List with different minerals
    :param mineral_dict: dict
        Dictionary with the mineral names and number referring to that (e.g. {0: 'MgCa-phase', 1: 'Ca-phase', 2: 'Noise', 3: 'Zn-phase', 4: 'Pb-phase', 5: 'Si-phase', 6: 'Ba-phase', 7: "Fe-phase"})

    :return: none
    """
    # Make a figure to display the MEANS of the training dataset
    fig1 = plt.figure(figsize=(16, 8))
    ax1 = fig1.add_subplot(1, 1, 1)

    X_training_database = X_training_database.iloc[:, 1:]
    print('X_training_database head', X_training_database.head())



    for i in range(number_of_minerals):
        # Take the part of the dataframe, specify the number of shots per mineral
        a = shots_per_mineral * i
        b = a + shots_per_mineral
        mineral_all_spectra = X_training_database[a:b]

        # Compute mean or median of some minerals
        mineral_mean_spectrum = list(mineral_all_spectra.mean(axis=0))

        # Use the median (which is a bit less dependant on outliers)
        mineral_median_spectrum = np.median(mineral_all_spectra, axis=0)
        # print(mineral_mean_spectrum)

        # Try do an overlay representing the stdev of the mean
        # to do

        # Plot this for each mineral
        headers_tr = df_tr.columns.values.tolist()
        list_of_channels = headers_tr[1:]

        ax1.plot(list_of_channels,
                 mineral_mean_spectrum,
                 c=colors[i])

        ax1.set_xlim(0, input_dim)
        ax1.set_title("Mean spectra of different minerals in the database")

        # Insert legend
        legend_custom_lines = list()
        for index, value in enumerate(colors):
            print('index ', index)
            print('value', value)
            legend_custom_lines.append(Line2D([0], [0], color=value, lw=2))
        plt.legend(legend_custom_lines, mineral_list, loc='best')
    fig1.show()

    # Make a figure to display the MEANS of the training dataset
    fig2 = plt.figure(figsize=(16, 8))
    ax2 = fig2.add_subplot(1, 1, 1)

    for i in range(number_of_minerals):
        # Take the part of the dataframe, specify the number of shots per mineral
        a = shots_per_mineral * i
        b = a + (shots_per_mineral)
        mineral_all_spectra = X_training_database[a:b]
        print('mineral_all_spectra')
        print(mineral_all_spectra)

        # Compute mean or median of some minerals
        mineral_mean_spectrum = mineral_all_spectra.mean(axis=0)
        # Use the median (which is a bit less dependant on outliers)
        mineral_median_spectrum = np.median(mineral_all_spectra, axis=0)

        # Make a SECOND figure to display the training dataset
        ax2.plot(list_of_channels,
                 mineral_median_spectrum,
                 c=colors[i])

        ax2.set_xlim(0, input_dim)
        ax2.set_title("Median spectra of different minerals in the database")

        # Insert legend
        legend_custom_lines = list()
        for index, value in enumerate(colors):
            print('index ', index)
            print('value', value)
            legend_custom_lines.append(Line2D([0], [0], color=value, lw=2))
        plt.legend(legend_custom_lines, mineral_list, loc='best')

    # Make a third figure to display the training database
    for i in range(number_of_minerals):

        # Make each time a new figure
        fig3 = plt.figure(figsize=(16, 8))
        ax3 = fig3.add_subplot(1, 1, 1)

        # Take the part of the dataframe, specify the number of shots per mineral
        a = shots_per_mineral * i
        b = a + (shots_per_mineral)
        mineral_all_spectra = X_training_database[a:b]

        print('list_of_channels', list_of_channels)
        print('mineral_all_spectra', mineral_all_spectra)

        # Make another figure to display the training dataset
        for index, mineral_spectrum in mineral_all_spectra.iterrows():
            print('mineral_spectrum is', mineral_spectrum)
            ax3.plot(list_of_channels,
                     mineral_spectrum)

        ax3.set_xlim(0, input_dim)
        mineral_names = list(mineral_dict.values())
        print(mineral_names)
        ax3.set_title(mineral_names[i])

        # Insert legend
        legend_custom_lines = list()
        for index, value in enumerate(colors):
            print('index ', index)
            print('value', value)
            legend_custom_lines.append(Line2D([0], [0], color=value, lw=2))
        plt.legend(legend_custom_lines, mineral_list, loc='best')

        fig3.show()

    return

def classify_data_using_nn(coords_list_run, X_unknown, model, uncertainty_quantification, class_list, certainty_list):
    """
    :param coords_list_run: list
    :param X_unknown: array
    :param model: Pytorch model
    :param uncertainty_quantification: Bool
    :param class_list: list
    :param certainty_list: list

    :return: class_list (list), certainty_list (list)
    """

    @torch.no_grad()
    def predict_unknown_NN(X_unknown, model):
        y_pred = model(X_unknown)
        certainty = y_pred.max().cpu().numpy()  # added cpu()
        return y_pred.argmax()

    @torch.no_grad()
    def predict_unknown_NN_uncertainty(X_unknown):
        y_pred = model(X_unknown)
        certainty = y_pred.max().cpu().numpy()
        # print('certainty is ...')
        # print(certainty)
        return y_pred

    # Loop through the dataset and classify
    for count, coordinate in enumerate(coords_list_run):  # maybe replace it by .iter (Trystan's suggestion)
        print('coordinate is', coordinate, 'count is', count)
        # Take row
        unknown_spectrum = X_unknown[count]

        # Necessary to put it on the cpu first
        unknown_spectrum = unknown_spectrum.cpu()
        unknown_spectrum = snv(np.array(unknown_spectrum))

        # Convert to tensor
        # print('convert to tensor')
        unknown_spectrum = torch.tensor(unknown_spectrum)
        unknown_spectrum = unknown_spectrum.to(torch.device(device))  # Try putting it on the GPU

        # Predict mineral now and append to class_list
        # print(unknown_spectrum.shape)
        classification_result = predict_unknown_NN(unknown_spectrum, model)
        mineral = classification_result.cpu().tolist()
        class_list.append(mineral)
        # print('class_list', class_list)

        # Try to quantify uncertainty (to do)
        # print('uncertainty quantification')
        if uncertainty_quantification:

            # Quantify certainty
            y_pred = predict_unknown_NN_uncertainty(unknown_spectrum)
            y_pred = y_pred.cpu()

            softmax = torch.nn.Softmax(dim=0)
            y_pred_softmax = softmax(y_pred)
            # print('y_pred after softmax', y_pred_softmax) #tensor([3.3858e-05, 2.1925e-03, 9.9567e-01, 1.6650e-05, 7.0610e-04, 5.3277e-05, 7.3105e-04, 6.0012e-04])

            y_pred_prob = y_pred_softmax.tolist() # 3.3857628295663744e-05
            # print('y_pred_prob', y_pred_prob)

            y_pred_prob_sorted = sorted(y_pred_prob)
            # print('y_pred_sorted')
            min_list = min(y_pred_prob_sorted)
            max_list = max(y_pred_prob_sorted)
            second_max_from_list = y_pred_prob_sorted[1]

            # Compute difference between maximum and second heighest
            # difference = max_list - second_max_from_list
            #
            # difference = max_list - min_list
            # print('difference', difference)
            certainty_list.append(round(max_list, 3))

        else:
            certainty_list.append(1)
        # print('pixel classified with NN')

    return class_list, certainty_list

def classify_data_using_sam(coords_list_run, X_unknown, uncertainty_quantification,class_list, certainty_list, number_of_minerals, X_training_database, shots_per_mineral):
    """ Function that classifies a 2D array using spectral angle mapper (SAM) and returns the label list (class_list) and uncertainty list

    :param coords_list_run: list of lists
    :param X_unknown: 2D array
    :param uncertainty_quantification: Boolean
    :param class_list: list
    :param certainty_list: list
    :param number_of_minerals: int
    :param X_training_database: pd.DataFrame
    :param shots_per_mineral: int

    :return: class_list, uncertainty list
    """
    class_mean_spectra_dict = dict()

    # Compute average classes from groundtruth database
    for i in range(number_of_minerals):
        a = shots_per_mineral * i
        b = a + (shots_per_mineral - 1)
        mineral_all_spectra = X_training_database[a:b]

        # Compute mean of each mineral and store in dict # to do: evaluate the influence of changing to median
        mineral_mean_spectrum = mineral_all_spectra.mean(axis=0)
        print(mineral_mean_spectrum)
        class_mean_spectra_dict[i] = mineral_mean_spectrum

    # Loop through the dataset and classify
    for count, coordinate in enumerate(coords_list_run):
        # Take row from df_te (test data)
        # unknown_row = df_te.iloc[count]
        # unknown_row = unknown_row.tolist()
        print('coordinate', coordinate, 'count', count)
        x = int(coordinate[0])
        print('x', x)
        y = int(coordinate[1])
        print('y', y)

        unknown_spectrum = X_unknown[count]  # ds.pixel_spectrum((x, y))
        unknown_spectrum = unknown_spectrum.cpu()

        classification_result_angles = list()
        # print('number of minerals', number_of_minerals)
        for i in range(number_of_minerals):
            s_GT = list(class_mean_spectra_dict[i])
            angle = computeSA_custom(s_GT, unknown_spectrum)
            classification_result_angles.append(angle)

        # Predict mineral now
        print(classification_result_angles)
        mineral = np.array(classification_result_angles).argmin()
        # print('mineral ', mineral)
        class_list.append(mineral)

        if uncertainty_quantification:

            # Quantify certainty
            angles = list(classification_result_angles)
            print('angles ', angles)

            # Take the values from the list
            min_angle = min(angles)
            max_angle = max(angles)
            angles_sorted = sorted(angles)
            # print('angles_sorted', angles_sorted)
            second_min_from_list = angles_sorted[1]
            # print('second_min_from_list', second_min_from_list)

            # Compute difference between maximum and second heighest
            difference = max_angle - min_angle
            # print('difference is', difference)
            certainty_list.append(difference)

        else:
            certainty_list.append(1)

    return class_list, certainty_list

def classify_data_using_max(coords_list_run,
                            X_unknown,
                            uncertainty_quantification,
                            class_list,
                            certainty_list,
                            mineral_dict):
    # el_list:
    #     'Ag', 'Al', 'As', 'Au', 'B',
    #     'Ba', 'Be', 'Br', 'Bi', 'C',
    #     'Ca', 'Cd', 'Cl', 'Co', 'Cr',
    #     'Cs', 'Cu', 'Fe', 'Ga', 'Ge',
    #     'H', 'Hf', 'Hg', 'In', 'K',
    #     'Li', 'Mg', 'Mn', 'Mo', 'Na',
    #     'Ni', 'La', 'P', 'Pb', 'Rb',
    #     'S', 'Sb', 'Sc', 'Se', 'Sm',
    #     'Sn', 'Si', 'Sr', 'Th', 'Tl',
    #     'Ti', 'U', 'V', 'Y', 'Zn',
    #     'Zr']
    # Now make a dictionary where the mineral_dict phases are linked to the
    index_el_list = {0: 26, 1: 10, 2: 20, 3: 49, 4: 33, 5: 41, 6: 5, 7: 17} #for noise I took Hydrogen
    index_el_list_indexes = list(index_el_list.values())
    print('index_el_list_indexes', index_el_list_indexes)
    print(X_unknown.shape)
    X_unknown_light = X_unknown[:, index_el_list_indexes]
    print(X_unknown_light.shape)

    # Loop through the dataset and classify
    for count, coordinate in enumerate(coords_list_run):
        # Take row from df_te (test data)
        print('coordinate', coordinate, 'count', count)
        x = int(coordinate[0])
        print('x', x)
        y = int(coordinate[1])
        print('y', y)

        # Get unknown spectrum
        unknown_spectrum = X_unknown_light[count]
        unknown_spectrum = unknown_spectrum.cpu()
        unknown_spectrum = list(unknown_spectrum)
        print('unknown_spectrum', unknown_spectrum)
        print(np.mean(unknown_spectrum))

        # Predict mineral now: determine index from maximum value
        if np.mean(unknown_spectrum) > 2000:
            mineral = unknown_spectrum.index(max(unknown_spectrum))
            print('mineral ', mineral)
            class_list.append(mineral)
        else:
            mineral = '2' # noise
            class_list.append(mineral)

        # Add a "1" to the certainty_list
        if uncertainty_quantification:
            certainty_list.append(1)

    return class_list, certainty_list

def mineral_counter(class_list, number_of_minerals, mineral_dict):
    """ Counts the percentage of minerals in your sample. Does not take into account the noise percentage.
    :param class_list: list
    :param number_of_minerals: int
    :param mineral_dict: dict
    :return: dict with volumetric percentages (The sum of the elements (without noise) is 100%)
    """

    print('mineral_dict', mineral_dict)
    counter = collections.Counter(class_list)
    print("Counter", counter)

    counter_all_minerals = dict()
    for i in range(number_of_minerals):
        if counter.get(i) is not None:
            value = counter[i]
            counter_all_minerals[i] = value
        else:
            counter_all_minerals[i] = 0
    counter = counter_all_minerals
    print('modified counter', counter)

    print('Mineral_dict', mineral_dict)
    mineral_vol_percentage = dict()
    total_number_of_pixels = sum(counter.values()) # take sum of the values of the counter dict
    print('total number of pixels is', total_number_of_pixels)

    for key in counter:
        # Get the mineral name using the key in the counter
        mineral = mineral_dict[int(key)]
        # Print mineral name and abundance
        print(mineral, '->', counter[key])
        # If the mineral class is Noise, grab the number of noise pixels
        if mineral == "Noise":
            # If there is some noise
            if counter[key] != 0:
                noise_pixels = counter[key]
                print('noise_pixels', noise_pixels)
            # If there is no noise...
            else:
                noise_pixels = 0

    for key in mineral_dict:
        # print('key', key)
        # Get mineral name
        mineral = mineral_dict[key]
        # print('mineral', mineral)

        # Divide pixels of mineral through the total number of pixels
        pixels_mineral = counter[key]
        pixels_mineral_percentage = pixels_mineral / (total_number_of_pixels - noise_pixels)

        # Store it in the dict
        mineral_vol_percentage[str(mineral)] = round(pixels_mineral_percentage * 100, 2)

    # Print the vol% of each mineral including noise
    print(mineral_vol_percentage)

    # Make string of this info
    mineral_mapping_results = str(mineral_vol_percentage)
    # mineral_mapping_results.replace()
    print('mineral mapping results', mineral_mapping_results)

    return mineral_mapping_results

def get_accuracy_multiclass(pred_arr, original_arr):
    """ Function used to compare the predicted results of the NN with the true results
    :param pred_arr:
    :param original_arr:

    :return: fraction of well-classified samples
    """
    if len(pred_arr) != len(original_arr):
        return False
    pred_arr = pred_arr.cpu().numpy()
    original_arr = original_arr.cpu().numpy()
    final_pred = []

    for i in range(len(pred_arr)):
        final_pred.append(np.argmax(pred_arr[i]))
    final_pred = np.array(final_pred)
    count = 0

    #here we are doing a simple comparison between the predicted_arr and the original_arr to get the final accuracy
    for i in range(len(original_arr)):
        if final_pred[i] == original_arr[i]:
            count +=  1
    return count/len(final_pred)

def show_spectrum_pixel(pixel_name, ds, ds_width, ds_height, list_of_elements):
    """ Function that displays the spectrum of a given pixel coordinate.

    :param pixel_name: str
        Needs to be in the shape of "x15/y15"
    :param ds: np.ndarray
    :param ds_width: int
    :param ds_height: int
    :param list_of_elements: list

    :return:
    """
    # Create a figure
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1, 1, 1)

    # Limit axes to zoom in
    # ax.set_xlim(0, channels_slice_stop - channels_slice_start)
    # ax.set_ylim(0, y_lim)

    # c = df2.loc[pixel_name]
    # print('row is...', c)
    pixel_name_split = pixel_name.split('/')
    print('pixel_name_split', pixel_name_split)
    pixel_name_x = pixel_name_split[0]
    pixel_name_x = int(pixel_name_x.replace('x',''))
    print('pixel_name_x is', pixel_name_x)
    pixel_name_y = pixel_name_split[1]
    pixel_name_y = int(pixel_name_y.replace('y',''))
    print('pixel_name_y is', pixel_name_y)
    print('ds.shape', ds.shape)
    print('ds_width', ds_width)
    print('ds_height', ds_height)

    try:
        spectrum = snv(ds[pixel_name_x, pixel_name_x , :])
    except:
        print('Pixel not in image')
    print('spectrum.shape', spectrum.shape)
    print('spectrum', spectrum)
    ax.plot(list_of_elements, spectrum)
    ax.set_title(str(pixel_name))
    fig.show()

    return

def use_roi_on_loaded_nc(ds, loc_roi_file, input_dim, bands):
    """ This function can be used to reduce a np.hyperimage to a hypercube
    with only the elemental data layers in the 3rd dimension

    :param ds: np.ndarray. Hyperspectral dataset file
    :param loc_roi_file: str. Specifies the name of the file where the wavelengths and elements are located.
    :param input_dim: int. Specifies the "spectral dimension" of the loaded dataset. The spectral dimension is determined based on the training dataset loading.
    :param bands: np.array. Specifies the wavelengths of each band of the spectrometer.
    :return: ds_new, el_list: new dataset drastically reduced in 3rd dimension, el_list: element list
    """

    # Import the stuff
    wavelengths = pd.read_csv(loc_roi_file)

    # Compute min and max wavelength
    min_wavelength = float(min(bands))
    print('min wavelength', min_wavelength)
    max_wavelength = float(max(bands))
    print('max wavelength', max_wavelength)

    # Compute the channel numbers for each element
    roi_elements = list()
    roi_channels = list()
    for index, row in wavelengths.iterrows():
        # Get wavelength
        wavelength = row['Wavelength(nm)']
        # Get element
        element = row['Element']

        if wavelength <= min_wavelength or wavelength >= max_wavelength:
            # Add dummy data
            pass
        else:
            # Add element
            roi_elements.append(element)
            # Look for the index to add in the channels_to_wavelengths_arr

            print('bands', bands)
            print(type(bands))
            print('wavelength', wavelength)
            print(type(wavelength))
            channel_number = np.abs(bands - wavelength).argmin()
            print('channel_number', channel_number)
            roi_channels.append(int(channel_number))

    dict_elements = dict(zip(roi_elements, roi_channels))
    print('roi_elements', roi_elements) # ['Ag', 'Ag*', 'Al', 'Al*', 'Al**', 'Al***', 'Al****', 'Al*****', 'As', 'Au', 'Au*', 'B', 'Ba', 'Ba*', 'Be', 'Be*', 'Bi', 'Bi*', 'B
    print('roi channels', roi_channels) # [4861, 4523, 6990, 3642, 1244, 6919, 2669, 3603, 1173, 1414, 2204, 324, 11466, 6710, 3780, 1624, 3551, 811, 10415, 5999


    # Now i need to make a new dataframe with the elemental data (mean of the channels I took)
    el_list_of_lists = list()
    channel_list_of_lists = list()
    print('len roi elements', len(roi_elements))
    dummy_list_elements = list()
    dummy_list_channels = list()

    for index, el in enumerate(roi_elements):
        # Get the element (without any stars)
        el_clean = el.replace(' ', '')
        el_clean = el_clean.replace('*', '')
        channel = roi_channels[index]

        # If the element is not the last element of the list
        if index != len(roi_elements) - 1:
            next_el_clean = roi_elements[index + 1]
            next_el_clean = next_el_clean.replace(' ', '')
            next_el_clean = next_el_clean.replace('*', '')

            # Check if the element is NOT the same as the next element. Then it needs to be added to the list.
            if el_clean != next_el_clean:
                # print('This element is the same as the next element')
                # Add it to the list of list
                dummy_list_elements.append(el_clean)
                dummy_list_channels.append(channel)
                el_list_of_lists.append(dummy_list_elements)
                channel_list_of_lists.append(dummy_list_channels)
                dummy_list_channels = list()
                dummy_list_elements = list()

            # If the next element in the list is the same element
            else:
                print('This is a new element')
                dummy_list_elements.append(el_clean)
                dummy_list_channels.append(channel)

        # If the element is the last one in the list, it will give an error which I need to avoid
        else:
            print('Last element now')
            # If the element is NOT the same as the PREVIOUS element
            if el_clean != next_el_clean:
                print('new element')
                # Add it to the list of list
                dummy_list_elements.append(el_clean)
                dummy_list_channels.append(channel)
                el_list_of_lists.append(dummy_list_elements)
                channel_list_of_lists.append(dummy_list_channels)
                dummy_list_channels = list()
                dummy_list_elements = list()

            # If the PREVIOUS element in the list is the same element
            else:
                print('previous element is the same')
                print('This is a new element')
                dummy_list_elements.append(el_clean)
                dummy_list_channels.append(channel)
                el_list_of_lists.append(dummy_list_elements)
                channel_list_of_lists.append(dummy_list_channels)

    print('channel_list_of_lists') # [[4861, 4523], [6990, 3642, 1244, 6919, 2669, 3603], [1173], [1414, 2204], [324],
    print(channel_list_of_lists)
    print('el_list_of_lists') # [['Ag', 'Ag'], ['Al', 'Al', 'Al', 'Al', 'Al', 'Al'], ['As'], ['Au', 'Au'], ['B'], ['Ba', 'Ba'], ['Be',
    print(el_list_of_lists)
    print(len(el_list_of_lists))

    # Now loop through the channel_list_of_lists and do the averaging over the different wavelengths and add them to the new ndarray
    # First make a new array with zero's which will be filled up
    ds_new = np.zeros((ds.shape[0], ds.shape[1], input_dim))
    el_list = list()

    for index, channel_list in enumerate(channel_list_of_lists):
        # Compute the mean of the channels using the channel numbers
        el = el_list_of_lists[index][0]
        print(' ')
        print('el is', el) # Ag for example
        print('channel list', channel_list) # [4861, 4523] for example
        print('index is', index) # 0 for example

        # Append element to the el_list
        el_list.append(str(el))

        # If there is only one channel for this element
        if len(channel_list) == 1:
            # Take the right channel
            c = int(channel_list[0])
            # Insert new channel in the right column using the hyperspectral image of ds
            ds_new[:, :, index] = ds[:, :, c]

        elif len(channel_list) == 2:
            # Take the first channel
            ds_el1 = ds[:, :, channel_list[0]]
            # Take the second channel
            ds_el2 = ds[:, :, channel_list[1]]
            # Compute the median and store it in ds_new
            ds_new[:, :, index] = np.median((ds_el1, ds_el2), axis = 0)

        elif len(channel_list) == 3:

            # Take the first channel
            ds_el1 = ds[:, :, channel_list[0]]
            # Take the second channel
            ds_el2 = ds[:, :, channel_list[1]]
            # Take the third channel
            ds_el3 = ds[:, :, channel_list[2]]

            # Compute the median and store it in ds_new
            ds_new[:, :, index] = np.median(
                (ds_el1, ds_el2, ds_el3), axis=0)

        elif len(channel_list) == 4:
            # Take the first channel
            ds_el1 = ds[:, :, channel_list[0]]
            # Take the second channel
            ds_el2 = ds[:, :, channel_list[1]]
            # Take the third channel
            ds_el3 = ds[:, :, channel_list[2]]
            # Take the fourth channel
            ds_el4 = ds[:, :, channel_list[3]]

            # Compute the median and store it in ds_new
            ds_new[:, :, index] = np.median(
                (ds_el1, ds_el2, ds_el3, ds_el4), axis=0)

        elif len(channel_list) == 5:
            # Take the first channel
            ds_el1 = ds[:, :, channel_list[0]]
            # Take the second channel
            ds_el2 = ds[:, :, channel_list[1]]
            # Take the third channel
            ds_el3 = ds[:, :, channel_list[2]]
            # Take the fourth channel
            ds_el4 = ds[:, :, channel_list[3]]
            # Take the fourth channel
            ds_el5 = ds[:, :, channel_list[4]]

            # Compute the median and store it in ds_new
            ds_new[:, :, index] = np.median(
                (ds_el1, ds_el2, ds_el3, ds_el4, ds_el5), axis=0)

        elif len(channel_list) == 6:
            # Take the first channel
            ds_el1 = ds[:, :, channel_list[0]]
            # Take the second channel
            ds_el2 = ds[:, :, channel_list[1]]
            # Take the third channel
            ds_el3 = ds[:, :, channel_list[2]]
            # Take the fourth channel
            ds_el4 = ds[:, :, channel_list[3]]
            # Take the fourth channel
            ds_el5 = ds[:, :, channel_list[4]]
            # Take the fourth channel
            ds_el6 = ds[:, :, channel_list[5]]

            # Compute the median and store it in ds_new
            ds_new[:, :, index] = np.median(
                (ds_el1, ds_el2, ds_el3, ds_el4, ds_el5, ds_el6), axis=0)

        elif len(channel_list) == 7:
            # Take the first channel
            ds_el1 = ds[:, :, channel_list[0]]
            # Take the second channel
            ds_el2 = ds[:, :, channel_list[1]]
            # Take the third channel
            ds_el3 = ds[:, :, channel_list[2]]
            # Take the fourth channel
            ds_el4 = ds[:, :, channel_list[3]]
            # Take the fourth channel
            ds_el5 = ds[:, :, channel_list[4]]
            # Take the fourth channel
            ds_el6 = ds[:, :, channel_list[5]]
            # Take the fourth channel
            ds_el7 = ds[:, :, channel_list[6]]

            # Compute the median and store it in ds_new
            ds_new[:, :, index] = np.median(
                (ds_el1, ds_el2, ds_el3, ds_el4, ds_el5, ds_el6, ds_el7), axis=0)

        elif len(channel_list) == 8:
            # Take the first channel
            ds_el1 = ds[:, :, channel_list[0]]
            # Take the second channel
            ds_el2 = ds[:, :, channel_list[1]]
            # Take the third channel
            ds_el3 = ds[:, :, channel_list[2]]
            # Take the fourth channel
            ds_el4 = ds[:, :, channel_list[3]]
            # Take the fourth channel
            ds_el5 = ds[:, :, channel_list[4]]
            # Take the fourth channel
            ds_el6 = ds[:, :, channel_list[5]]
            # Take the fourth channel
            ds_el7 = ds[:, :, channel_list[6]]
            # Take the fourth channel
            ds_el8 = ds[:, :, channel_list[7]]

            # Compute the median and store it in ds_new
            ds_new[:, :, index] = np.median(
                (ds_el1, ds_el2, ds_el3, ds_el4, ds_el5, ds_el6, ds_el7, ds_el8), axis=0)

        elif len(channel_list) == 9:
            # Take the first channel
            ds_el1 = ds[:, :, channel_list[0]]
            # Take the second channel
            ds_el2 = ds[:, :, channel_list[1]]
            # Take the third channel
            ds_el3 = ds[:, :, channel_list[2]]
            # Take the fourth channel
            ds_el4 = ds[:, :, channel_list[3]]
            # Take the fourth channel
            ds_el5 = ds[:, :, channel_list[4]]
            # Take the fourth channel
            ds_el6 = ds[:, :, channel_list[5]]
            # Take the fourth channel
            ds_el7 = ds[:, :, channel_list[6]]
            # Take the fourth channel
            ds_el8 = ds[:, :, channel_list[7]]
            # Take the fourth channel
            ds_el9 = ds[:, :, channel_list[8]]

            # Compute the median and store it in ds_new
            ds_new[:, :, index] = np.median(
                (ds_el1, ds_el2, ds_el3, ds_el4, ds_el5, ds_el6, ds_el7, ds_el8, ds_el9), axis=0)

        elif len(channel_list) == 10:
            # Take the first channel
            ds_el1 = ds[:, :, channel_list[0]]
            # Take the second channel
            ds_el2 = ds[:, :, channel_list[1]]
            # Take the zhird channel
            ds_el3 = ds[:, :, channel_list[2]]
            # Take the fourth channel
            ds_el4 = ds[:, :, channel_list[3]]
            # Take the fourth channel
            ds_el5 = ds[:, :, channel_list[4]]
            # Take the fourth channel
            ds_el6 = ds[:, :, channel_list[5]]
            # Take the fourth channel
            ds_el7 = ds[:, :, channel_list[6]]
            # Take the fourth channel
            ds_el8 = ds[:, :, channel_list[7]]
            # Take the fourth channel
            ds_el9 = ds[:, :, channel_list[8]]
            # Take the fourth channel
            ds_el10 = ds[:, :, channel_list[9]]

            # Compute the median and store it in ds_new
            ds_new[:, :, index] = np.median((ds_el1, ds_el2, ds_el3, ds_el4, ds_el5, ds_el6, ds_el7, ds_el8, ds_el9, ds_el10), axis = 0)

    return ds_new, el_list

def use_roi_on_libs_training_dataset(df_tr, loc_roi_file, channels_to_wavelengths_arr, use_snv):
    """ Function that reduces the spectral dimension of the training dataset towards a much smaller datacube.
    It takes multiple wavelengths, combines the information (through averaging) and computes one single value per element.

    :param df_tr: training dataset
    :param loc_roi_file: str. Location of the csv files with the wavelengths specified.
    :param channels_to_wavelengths_arr: file that connects both channel numbers and wavelengths (bands)
    :param use_snv: Boolean value indicating the necessity for standard normal variate (SNV) correction

    :return: df_dataset_new_el, el_list, row_names_training_dataset
    """

    el_list = list()

    print('channels_to_wavelengths_arr', channels_to_wavelengths_arr)

    print('df_tr input at beginning of use_roi function')
    print(df_tr.head())
    row_names_training_dataset = df_tr.iloc[:, 3]
    df_class_int = df_tr.iloc[:, 1]
    df_tr = df_tr.iloc[:, 4:]

    print('df_tr.head again')
    print(df_tr.head())
    # Import the stuff
    wavelengths = pd.read_csv(loc_roi_file)

    # Compute min and max wavelength
    min_wavelength = float(min(channels_to_wavelengths_arr))
    max_wavelength = float(max(channels_to_wavelengths_arr))

    # Compute the channel numbers for each element
    roi_elements = list()
    roi_channels = list()
    for index, row in wavelengths.iterrows():
        # Get wavelength
        wavelength = row['Wavelength(nm)']
        # Get element
        element = row['Element']

        if wavelength <= min_wavelength or wavelength >= max_wavelength:
            # Add dummy data
            pass
        else:
            # Add element
            roi_elements.append(element)
            # Look for the index to add in the channels_to_wavelengths_arr
            channel_number = np.abs(channels_to_wavelengths_arr - wavelength).argmin()
            # print(channel_number)
            roi_channels.append(int(channel_number))

    dict_elements = dict(zip(roi_elements, roi_channels))
    print('roi_elements', roi_elements)
    print('roi channels', roi_channels)

    print('df_tr.head()')
    print(df_tr.head())

    # If the data is training data, the only column we need is the class_int row
    df_dataset_new = df_tr.iloc[:, roi_channels]
    # Add another column with the name of the mineral (class_int), preferentially at the beginning
    # df_dataset_new['Class_int'] = df['Class_int']

    print('df_dataset_new at line 604')
    print(df_dataset_new)

    # Now it takes the dataframe and converts it into elemental data
    # Import the wavelength
    wavelengths = pd.read_csv(loc_roi_file)

    print('wavelengths', wavelengths)

    # Compute min and max wavelength
    min_wavelength = float(min(channels_to_wavelengths_arr))
    max_wavelength = float(max(channels_to_wavelengths_arr))

    # Compute the channel numbers for each element
    roi_elements = list()
    roi_channels = list()

    for index, row in wavelengths.iterrows():
        # Get wavelength
        wavelength = row['Wavelength(nm)']
        # Get element
        element = row['Element']

        # If the wavelength does not fit in the spectrometer range, do nothing
        if wavelength <= min_wavelength or wavelength >= max_wavelength:
            # Add dummy data
            pass
        # If the wavelength can be found in the spectrometer range, add it
        else:
            # Add element
            roi_elements.append(element)
            # Look for the index to add in the channels_to_wavelengths_arr
            channel_number = np.abs(channels_to_wavelengths_arr - wavelength).argmin()
            # print(channel_number)
            roi_channels.append(int(channel_number))

    dict_elements = zip(roi_elements, roi_channels)
    # dict_elements = dict(dict_elements)
    print('roi_elements', roi_elements)  # ['Ag', 'Al', 'Al*', 'Al**', 'Al***', 'Al****', etc.
    print('roi_channels', roi_channels)  # [4861, 6990, 3642, 1244, 6919, 2669,

    print(dict_elements)

    # Now it needs to take the df, load it and only copy the necessary columns from the df based on the roi_channels
    # Add the row numbers (aka coordinates) to the new dataframe
    print('df_dataset_new')
    print(type(df_dataset_new))
    print(df_dataset_new)

    # Now i need to make a new dataframe with the elemental data (mean of the channels I took)
    el_list_of_lists = list()
    channel_list_of_lists = list()

    print('len roi elements', len(roi_elements))
    dummy_list_elements = list()
    dummy_list_channels = list()

    for index, el in enumerate(roi_elements):
        # Get the element (without any stars)
        el_clean = el.replace(' ', '')
        el_clean = el_clean.replace('*', '')
        # print('el_clean', el_clean)
        channel = roi_channels[index]

        # If the element is not the last element of the list
        if index != len(roi_elements) - 1:
            next_el_clean = roi_elements[index + 1]
            next_el_clean = next_el_clean.replace(' ', '')
            next_el_clean = next_el_clean.replace('*', '')
            # print('next_el_clean', next_el_clean)

            # Check if the element is NOT the same as the next element. Then it needs to be added to the list.
            if el_clean != next_el_clean:
                # print('This element is the same as the next element')
                # Add it to the list of list
                dummy_list_elements.append(el_clean)
                dummy_list_channels.append(channel)
                el_list_of_lists.append(dummy_list_elements)
                channel_list_of_lists.append(dummy_list_channels)
                dummy_list_channels = list()
                dummy_list_elements = list()

            # If the next element in the list is the same element
            else:
                print('This is a new element')
                dummy_list_elements.append(el_clean)
                dummy_list_channels.append(channel)

        # If the element is the last one in the list, it will give an error which I need to avoid
        else:
            print('Last element now')
            # If the element is NOT the same as the PREVIOUS element
            if el_clean != next_el_clean:
                print('new element')
                # Add it to the list of list
                dummy_list_elements.append(el_clean)
                dummy_list_channels.append(channel)
                el_list_of_lists.append(dummy_list_elements)
                channel_list_of_lists.append(dummy_list_channels)
                dummy_list_channels = list()
                dummy_list_elements = list()

            # If the PREVIOUS element in the list is the same element
            else:
                print('previous element is the same')
                print('This is a new element')
                dummy_list_elements.append(el_clean)
                dummy_list_channels.append(channel)
                el_list_of_lists.append(dummy_list_elements)
                channel_list_of_lists.append(dummy_list_channels)

    print('channel_list_of_lists')
    print(channel_list_of_lists)
    print('el_list_of_lists')
    print(el_list_of_lists)
    print('df_dataset_new.head()')
    print(df_dataset_new.head())

    # Now loop through the channel_list_of_lists and do the averaging over the different wavelengths
    for index, channel_list in enumerate(channel_list_of_lists):
        # Compute the mean of the channels using the channel numbers
        el = el_list_of_lists[index][0]
        print(' ')
        print('el is', el)
        print('channel list', channel_list)
        # print('len channel list', len(channel_list))
        print('index is', index)
        print('df_dataset_new')


        # If there is only one channel for this element
        if len(channel_list) == 1:
            print('one channel now')
            # print(df_dataset_new[channel_list[0]])
            # If it's the first one (Ag), a new df needs to be made
            if index == 0:
                print('channel_list', channel_list)
                df_dataset_new_el = pd.DataFrame(df_dataset_new.loc[:, str(channel_list[0])])
                df_dataset_new_el = df_dataset_new_el.rename(columns={df_dataset_new_el.columns[int(index)]: str(el)})
            # Otherwise, the new column needs to be added
            else:
                df_dataset_new_el = pd.concat([df_dataset_new_el, df_dataset_new[str(channel_list[0])]], axis=1)
                print('df_dataset_new_el')
                print(df_dataset_new_el.head())
                print('index', index)
                print('el', el)
                df_dataset_new_el = df_dataset_new_el.rename(
                    columns={df_dataset_new_el.columns[int(index)]: str(el)})
                # df_dataset_new_el = df_dataset_new_el.rename(columns={df_dataset_new_el.columns[int(index)]: str(el)})

        # If there are more than 1 channel to use
        elif len(channel_list) == 2:
            print('2 channels now')
            # If it's the first one (Ag), a new df needs to be made
            if index == 0:
                print('channel_list', channel_list)
                df_dataset_new_el = pd.DataFrame(df_dataset_new.loc[:, str(channel_list[0])])
                df_dataset_new_el = df_dataset_new_el.rename(columns={df_dataset_new_el.columns[int(index)]: str(el)})

                # print('first element')
                # df_dataset_new_el = df_dataset_new[str(channel_list[0])]
                # print('df_dataset_new_el', df_dataset_new_el.head())

            else:
                df = pd.concat([df_dataset_new[str(channel_list[0])], df_dataset_new[str(channel_list[1])]], axis=1)
                print('df', df.head())
                df_median = pd.DataFrame(data=df.median(axis=1), columns=[el])
                print('df_median', df_median.head())
                df_dataset_new_el = pd.concat([df_dataset_new_el, df_median], axis=1)
                print('df_dataset_new_el', df_dataset_new_el.head())

        elif len(channel_list) == 3:
            print('3 channels now')
            # If it's the first one (Ag), a new df needs to be made
            if index == 0:
                df_dataset_new_el = pd.DataFrame(df_dataset_new.loc[:, str(channel_list[0])])
                df_dataset_new_el = df_dataset_new_el.rename(columns={df_dataset_new_el.columns[int(index)]: str(el)})
            else:
                df = pd.concat([df_dataset_new[str(channel_list[0])], df_dataset_new[str(channel_list[1])],
                                df_dataset_new[str(channel_list[2])]], axis=1)
                print('df', df.head())
                df_median = pd.DataFrame(data=df.median(axis=1), columns=[el])
                print('df_median', df_median.head())
                df_dataset_new_el = pd.concat([df_dataset_new_el, df_median], axis=1)
                print('df_dataset_new_el', df_dataset_new_el.head())

        elif len(channel_list) == 4:
            print('4 channels now')
            # If it's the first one (Ag), a new df needs to be made
            if index == 0:
                df_dataset_new_el = df_dataset_new[str(channel_list[0])]
            else:
                df = pd.concat([df_dataset_new[str(channel_list[0])], df_dataset_new[str(channel_list[1])],
                                df_dataset_new[str(channel_list[2])], df_dataset_new[str(channel_list[3])]], axis=1)
                print('df', df.head())
                df_median = pd.DataFrame(data=df.median(axis=1), columns=[el])
                print('df_median', df_median.head())
                df_dataset_new_el = pd.concat([df_dataset_new_el, df_median], axis=1)
                print('df_dataset_new_el', df_dataset_new_el.head())

        elif len(channel_list) == 5:
            print('5 channels now')
            # If it's the first one (Ag), a new df needs to be made
            if index == 0:
                df_dataset_new_el = df_dataset_new[str(channel_list[0])]
            else:
                df = pd.concat([df_dataset_new[str(channel_list[0])], df_dataset_new[str(channel_list[1])],
                                df_dataset_new[str(channel_list[2])], df_dataset_new[str(channel_list[3])],
                                df_dataset_new[str(channel_list[4])]], axis=1)
                print('df', df.head())
                df_median = pd.DataFrame(data=df.median(axis=1), columns=[el])
                print('df_median', df_median.head())
                df_dataset_new_el = pd.concat([df_dataset_new_el, df_median], axis=1)
                print('df_dataset_new_el', df_dataset_new_el.head())

        elif len(channel_list) == 6:
            print('6 channels now')
            # If it's the first one (Ag), a new df needs to be made
            if index == 0:
                df_dataset_new_el = df_dataset_new[channel_list[0]]
            else:
                df = pd.concat([df_dataset_new[str(channel_list[0])], df_dataset_new[str(channel_list[1])],
                                df_dataset_new[str(channel_list[2])], df_dataset_new[str(channel_list[3])],
                                df_dataset_new[str(channel_list[4])], df_dataset_new[str(channel_list[5])]], axis=1)
                print('df', df.head())
                df_median = pd.DataFrame(data=df.median(axis=1), columns=[el])
                print('df_median', df_median.head())
                df_dataset_new_el = pd.concat([df_dataset_new_el, df_median], axis=1)
                print('df_dataset_new_el', df_dataset_new_el.head())

    # print('df_dataset_new_el')
    # print(df_dataset_new_el.head())

    # Do SNV if necessary
    if use_snv:
        for index, row in df_dataset_new_el.iterrows():
            print('row', row)
            print(type(np.array(row)))
            row_snv = snv(np.array(row))
            print('row_snv', row_snv)

            df_dataset_new_el.loc[index] = row_snv
            # print('df_dataset_new_el', df_dataset_new_el)

    print('df_dataset_new_el after SNV')
    print(df_dataset_new_el.head())

    # Export data
    print('df_class_int')
    print(df_class_int)
    # Add the class_int part in the first column
    df_dataset_new_el.insert(0, "Class_int", df_class_int)

    # Get a list with all the elements
    el_list = list(df_dataset_new_el.columns)
    # Remove the first "Columns" element of the list
    el_list = el_list[1:]
    return df_dataset_new_el, el_list, row_names_training_dataset

def compute_mean_and_stdev_spectrum_from_hyperspectral_image(ds, el_list):
    """ Computes mean spectrum from hyperspectral image.

    :param ds: hyperspectral image
    :param bands: wavelengths of the individual bands
    :return: None
    """
    # Try to compute the mean of the spectrum
    av_spectrum = np.mean(ds, axis=(0, 1))
    av_spectrum_stdev = np.std(ds, axis=(0, 1))
    print('average spectrum is', av_spectrum)
    print(av_spectrum.shape)
    print(av_spectrum)
    plt.figure(figsize=(15,6))
    plt.plot(el_list, av_spectrum)
    # plt.show()
    # plt.plot(ds.bands, av_spectrum_stdev)
    # plt.title('Average spectrum')
    # plt.show()
    # plt.close()

    # Plot standard deviation
    # plt.figure(figsize=(15, 6))
    plt.plot(el_list, av_spectrum_stdev)
    plt.title('Average and stdev spectrum')
    plt.show()
    plt.close()

    return

def train_network_nn(model, optimizer, criterion, train_dataset_loader, num_epochs, train_losses, validation_losses, validation_dataset_loader,
                     use_lr_scheduler, learning_rate, step_lr, gamma_lr, batch_size, dropout_fc):
    """ This function trains the neural network given the parameters foreseen in the arguments of the function.

    :param model: neural network (in its ground state)
    :param optimizer: pre-defined optimizer.
    :param criterion: nn.CrossEntropyLoss()
    :param train_dataset_loader: DataLoader for training dataset
    :param num_epochs: str. Number of iterations necessary
    :param train_losses: a bunch of zeroes that need to be filled in later during the training
    :param validation_losses: a bunch of zeroes that need to be filled in later during the training
    :param validation_dataset_loader: DataLoader for validation dataset.
    :param use_lr_scheduler: Boolean value expressing the need for learning rate scheduler.
    :param learning_rate: float value that indicates the learning rate for the neural network
    :param step_lr: parameter for learning rate adjustment. Indicates number of steps.
    :param gamma_lr: parameter for learning rate adjustment. Indicates fraction.
    :param batch_size: normally it is 64, 128 or 256. It is 128 in this case.
    :param dropout_fc: parameter between 0 and 1 describing the dropout in the fully connected layers

    :return: trained model
    """

    model = model.to(device)
    model = model.to(torch.device(device)) # Try putting it on the GPU

    # Start to train network

    if use_lr_scheduler:
        scheduler = StepLR(optimizer, step_size=step_lr, gamma=gamma_lr)

    # Store the stuff of Tensorboard
    log_dir = r"C:\Users\simon\PycharmProjects\automated-mineralogy\libs\Mineralmaps\logs_tensorboard"  # Directory to store the log files
    #os.chdir(log_dir)
    # Get the start datetime
    st = datetime.datetime.now()
    print('start time is ', st)
    timestamp = st.strftime("%Y%m%d-%H%M%S")
    experiment_name = str('lr') + str(learning_rate) \
                      + str('batchsize') + str(batch_size) \
                      + str('dropout') + str(dropout_fc) \
                      + str('step_lr') + str(step_lr) \
                      + str('gamma_lr') + str(gamma_lr) \
                      + str('drouput_fc') + str(dropout_fc)

    experiment_name = f"{experiment_name}_{timestamp}"

    for epoch in range(num_epochs):
        final_output = list()

        # for i, data in enumerate(X_train):
        for inputs, labels in train_dataset_loader:
            # Clear out the gradients from the last step loss.backward()
            optimizer.zero_grad()

            # Forward feed
            output_train = model(inputs)

            # Compute the loss of the training cycle
            loss_train = criterion(output_train, labels)

            # print('loss_train', loss_train)

            # Backward propagation: calculate gradients
            loss_train.backward()

            # Update the weights
            optimizer.step()

        for inputs_validation, labels_test in validation_dataset_loader:
            # Calculate output
            validation_test = model(inputs_validation)

            # Calculate the loss of the test cycle
            loss_validation = criterion(validation_test, labels_test)

        # Compute training statistics
        train_losses[epoch] = loss_train.item()
        validation_losses[epoch] = loss_validation.item()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss_train.item():.4f}, Validation Loss: {loss_validation.item():.4f}")
        if use_lr_scheduler == True:
            # Change the learning rate
            scheduler.step()

        # Create the log directory
        log_dir = r"C:\Users\simon\PycharmProjects\automated-mineralogy\libs\Mineralmaps\logs_tensorboard" + str('/') + str(experiment_name)
        # log_dir = f"logs/{experiment_name}"
        writer = SummaryWriter(log_dir)
        # Type this command to launch the tensorboard GUI:
        #               tensorboard --logdir="C:\Users\simon\PycharmProjects\automated-mineralogy\libs\Mineralmaps\logs_tensorboard"

        # Log scalar values
        writer.add_scalar('Loss/train', loss_train, epoch)
        writer.add_scalar('Loss/validation', loss_validation, epoch)
        new_learning_rate = learning_rate * math.pow(gamma_lr, math.floor((1 + epoch) / step_lr))
        writer.add_scalar('Learning rate', new_learning_rate, epoch)
        writer.add_scalar('Batch size', batch_size, epoch)

    # Close writer when ready
    writer.close()

    return model
