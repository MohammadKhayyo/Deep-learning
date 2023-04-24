# Get the list of all files and directories
import os

path = r"./pieces"
dir_list = os.listdir(path)

# pair_order_list = combination_two_list(dir_list, random.sample(dir_list, 100))
#
with open('pieces_to_pictures_pairs_names.txt', 'w') as f:
    for piece in dir_list:
        picture_filename = piece.split('_P')[0] + '.jpg'
        f.write(piece)
        f.write(' ')
        f.write(picture_filename)
        f.write('\n')
