
"""
This script is for when you want to make a training dataset from a given image and store it into a csv file
Input data is from a map (nc file)

ATTENTION: manual preprocessing is still required unfortunately. Some pixels are just not really galena.

"""
"""
# Silicon is done on NB32_1, with 9000 max intensity
# Copper is done on NB35_1 with 3000 max intensity
# Ag is done on NB32_5 with 15000 max intensity
"""

# Import dependencies
import os
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def find_closest_value_index(lst, target):
    closest_index = None
    closest_difference = float('inf')

    for i, value in enumerate(lst):
        difference = abs(value - target)
        if difference < closest_difference:
            closest_difference = difference
            closest_index = i

    return closest_index

def find_coordinates_using_threshold(element, sample, wavelength, nb_of_coordinates, min_intensity):
    os.chdir(r"C:\Users\simon\PycharmProjects\automated-mineralogy\libs\Datacubes\Input")
    # Load the nc file
    print('sample', sample)
    ds = xr.open_dataset(sample)

    # Take the bands
    bands = ds.bands.values
    print('bands', bands)

    # Make an empty list for storing the coords
    coords = list()

    # Get column of the element
    column_number = find_closest_value_index(bands, wavelength)
    print('column_number', column_number)

    # Iterate through the dataset
    width = ds.dims.mapping['x']
    height = ds.dims.mapping['y']
    for i in range(width):
        for j in range(height):
            spectrum = ds.mapping.isel(x=i, y=j)
            intensity = float(spectrum[column_number].values)
            if intensity > min_intensity:
                coord = str(element) + str('/x') + str(j) + str('/y') + str(i) + str('/') + str(sample_name[:-3]+str('/')+str(intensity))
                coords.append(coord)

    print('coords', coords)
    print('len(coords)', len(coords))

    return coords


# The training dataset can be made with this code
show_datacube = True
sample_list_with_known_coordinates = {
    "NB35_1uint.nc": ["crystals", "dispersed,Wil"],
    "NB32_3_2023-02-1517-25-31.nc": ["crystals"],
    "NB32_6_2023-02-1514-30-03.nc": ["crystals"],
    "Ivan_BRD-27-22-13_B_only16376.nc": ["crystals"],
    "Santoro1342023-06-26_12-03-18_only_16376.nc": ["crystals"]
}

# Load data
os.chdir("../Datacubes/Input")

# Create dictionary for storing the spectral data
mineral_db = dict()

for sample_name in sample_list_with_known_coordinates:
    # Load dataset
    ds = xr.open_dataset(sample_name)
    ds.load()
    print(ds)
    print('ds.bands', ds.bands)
    print(ds.dims['bands'])

    print('sample', sample_name)
    values_dict = sample_list_with_known_coordinates[sample_name]
    print('values_dict', values_dict)

    for value_dict in values_dict:
        print('value_dict', value_dict)
        print('type value_dict', type(value_dict))
        if value_dict == "crystals":
            if sample_name == "NB35_1uint.nc":
                print('option 1')

                minerals_locations = {'Dol': str("[30, 160, 36, 140]"), # hor1, vert1, hor2, vert2
                                      'Cc': str("[50, 120, 56, 100]"),
                                      'Noise': str("[0, 20, 6, 0]"),
                                        'Sphal': str("[65, 75, 75, 80]")  #50x Sphal
                                      } #

            # elif sample_name == "NB32_3_2023-02-1517-25-31.nc":
            #     # Store the spectra for each pixel in a dataframe with a labelled column
            #     minerals_locations = {'Gal': str("[45, 240, 55, 250];[25,21,30,25]"),  # hor1, vert1, hor2, vert2 #120
            #                           "Sphal": str("[80, 40, 84, 50]"), #40x Sphal
            #                           }

            elif sample_name == "NB32_3_2023-02-1517-25-31.nc":
                print('option 2')
                # Store the spectra for each pixel in a dataframe with a labelled column
                minerals_locations = {'Gal': str("[50, 241, 57, 251];[44, 245, 51, 250];[25,21,30,25]"),  # hor1, vert1, hor2, vert2 #120
                                      "Sphal": str("[41, 71, 45, 81]"), #40x Sphal
                                      }

            elif sample_name == "NB32_6_2023-02-1514-30-03.nc":
                print('option 3')
                # Store the spectra for each pixel in a dataframe with a labelled column
                minerals_locations = {"Sphal": str("[40, 140, 46, 145]"),  # 30x Sphal
                                      }

            # elif sample_name == "Ivan_BRD-27-22-13_A_only16376.nc":
            #     print('option 4')
            #     minerals_locations = {"Bar": str("[44, 65, 54, 77]")}
            #                          # "T+B": str("[48, 18, 54, 28]")}

            elif sample_name == "Ivan_BRD-27-22-13_B_only16376.nc":
                print('option 5')
                minerals_locations = {"Bar": str("[80, 121, 100, 127]")}
                                     # "T+B": str("[48, 18, 54, 28]")}

            elif sample_name == "Santoro1342023-06-26_12-03-18_only_16376.nc":
                print('option 6')
                minerals_locations = {"Pyr": str("[110, 10, 122, 20]")}

            # Make the image
            fig = plt.figure(figsize=(12, 12))
            ax1 = fig.add_subplot(111)
            map1 = ds.mapping.isel(bands=[6715]) #206 for Zn, channel 732 for Pb, 2881 for Sb
            map1.plot(ax=ax1, cmap=plt.cm.Greys)

            print('minerals_locations', minerals_locations)

            # Show the areas which have been selected
            for key in minerals_locations:
                print('key is', key)
                loc = minerals_locations[key]
                loc = loc.split(';')
                print('loc is', loc)

                for i in loc:
                    print('i is', i)
                    i = i.replace('[', "")
                    i = i.replace(']', "")
                    i = i.split(',')
                    hor1 = int(i[0])
                    print('hor1', hor1)
                    vert1 = int(i[1])
                    print('vert1', vert1)
                    hor2 = int(i[2])
                    print('hor2', hor2)
                    vert2 = int(i[3])
                    print('vert2', vert2)

                    # Compute width and height of rectangle
                    w = abs(hor1 - hor2)
                    print('w is', w)
                    h = abs(vert1 - vert2)
                    print('h is', h)

                    x_anch = min(hor1, hor2)
                    print('x_anch', x_anch)
                    y_anch = min(vert1, vert2)
                    print('y_anch', y_anch)

                    if key == 'Dol':
                        color = 'blue'
                    elif key == 'Gal':
                        color = 'grey'
                    elif key == 'Sphal':
                        color = 'chocolate'
                    elif key == 'Noise':
                        color = 'lightgrey'
                    elif key == "Cc":
                        color = 'cyan'
                    else:
                        color = 'lime'

                    ax1.add_patch(Rectangle((x_anch, y_anch),
                                           w,
                                           h,
                                           fc='none',
                                           ec=color,
                                           lw=2))
                    print('rectangle drawn')

                    for i in range(w):
                        for j in range(h):
                            spectrum = ds.mapping.isel(x=y_anch+j,y=x_anch+i)
                            list_spectrum = spectrum.values
                            name = str(key) + str('x') + str(y_anch+j) + str('/y') + str(x_anch+i) + str(sample_name[:-3])
                            print(name)
                            if key == 'Sphal':
                                mineral_class_int = str('3')
                            elif key == "Dol":
                                mineral_class_int = str('0')
                            elif key == "Cc":
                                mineral_class_int = str('1')
                            elif key == "Noise":
                                mineral_class_int = str('2')
                            elif key == "Gal":
                                mineral_class_int = str('4')
                            elif key == "Wil":
                                mineral_class_int = str('5')
                            elif key == "Bar":
                                mineral_class_int = str('6')
                            elif key == "Pyr":
                                mineral_class_int = str('7')
                            else:
                                mineral_class_int = str('?')

                            list_spectrum = list_spectrum.tolist()
                            list_spectrum.insert(0, name)
                            list_spectrum.insert(0, key)
                            list_spectrum.insert(0, mineral_class_int)

                            mineral_db[name] = list_spectrum

                fig.show()
        else:

            print(' ********************************** DISPERSED NOW **********************************')

            # Take the name of the mineral
            value_dict_split = value_dict.split(',')
            mineral_list_dispersed_name = value_dict_split[1]
            print('Mineral_list_dispersed_name', mineral_list_dispersed_name)

            if mineral_list_dispersed_name == "Wil":
                wavelength = 251.61
                nb_of_coordinates = 120
                min_intensity = 11000

            mineral_list_dispersed_coords = find_coordinates_using_threshold(mineral_list_dispersed_name,
                                                                             sample_name,
                                                                             wavelength,
                                                                             nb_of_coordinates,
                                                                             min_intensity)
            print('mineral_list_dispersed_coords', mineral_list_dispersed_coords)

            # Load the right sample
            ds = xr.open_dataset(sample_name)
            ds.load()

            for index, coordlist in enumerate(mineral_list_dispersed_coords[:120]):
                print('index', index)
                print('coordlist', coordlist)
                coordslist_split = coordlist.split('/')
                print('coordslist_split', coordslist_split)

                x = coordslist_split[1]
                x = x.replace('x', '')
                y = coordslist_split[2]
                y = y.replace('y', '')

                print('x is', x)
                print('y is', y)
                spectrum = ds.mapping.isel(x=int(y), y=int(x))
                list_spectrum = spectrum.values

                name = str(mineral_list_dispersed_name) + str('x') + str(y) + str('/y') + str(x) + str(sample_name[:-3])
                print(name)

                if mineral_list_dispersed_name == 'Sphal':
                    mineral_class_int = str('3')
                elif mineral_list_dispersed_name == "Dol":
                    mineral_class_int = str('0')
                elif mineral_list_dispersed_name == "Cc":
                    mineral_class_int = str('1')
                elif mineral_list_dispersed_name == "Noise":
                    mineral_class_int = str('2')
                elif mineral_list_dispersed_name == "Gal":
                    mineral_class_int = str('4')
                elif mineral_list_dispersed_name == "Wil":
                    mineral_class_int = str('5')
                elif mineral_list_dispersed_name == "Bar":
                    mineral_class_int = str('6')
                elif mineral_list_dispersed_name == "Pyr":
                    mineral_class_int = str('7')
                else:
                    mineral_class_int = str('?')

                list_spectrum = list_spectrum.tolist()
                list_spectrum.insert(0, name)
                list_spectrum.insert(0, mineral_list_dispersed_name)
                list_spectrum.insert(0, mineral_class_int)
                mineral_db[name] = list_spectrum


"""
Prepare the dataframe for exporting
"""

print('Now exporting dict to a df')
# Export dict to panda dataframe, row by row
df = pd.DataFrame.from_dict(mineral_db, orient='index')  # , index = wavelengths_arr)
print(df)

header_list = ['Class_int', 'Class', "Coordinate"]
for i in range(16376):
    header_list.append(str(i))

# Give the three first columns a name
df.columns = header_list

# Sort on Class_int
df = df.sort_values('Class_int')

# # Export to Excel
# df.to_excel("AND1_annotated_dataset_LIBS.xlsx")

# Change directory
os.chdir(r"C:\Users\simon\PycharmProjects\automated-mineralogy\libs\Mineralmaps")

# Export to csv (which are faster to write and read
# name_csv = str("training_database_5_minerals_01June2023.csv")
name_csv = str("training_database_8_minerals_03July2023.csv")
df.to_csv(name_csv, sep=",")




