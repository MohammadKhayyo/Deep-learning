from PIL import Image
import cv2
import os

path = r'pieces'


def get_max_dimensions(path):
    return max(Image.open(path + '/' + f, 'r').size for f in os.listdir(path))


# new_size = get_max_dimensions(path)
new_size = (1024, 576)
print(new_size)
for piece in os.listdir(path):
    img = cv2.imread(path + '/' + piece)
    # rotate image to landscape mode
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    old_im = Image.open(path + '/' + piece)
    old_size = old_im.size
    new_im = Image.new("RGB", new_size)
    box = tuple((n - o) // 2 for n, o in zip(new_size, old_size))
    new_im.paste(old_im, box)
    # cv2.imwrite(r'pieces_fixed/' + piece, img)
    new_im.save(r'pieces_fixed/' + piece)

# old_im = Image.open('someimage.jpg')
# old_size = old_im.size
#
# new_im = Image.new("RGB", new_size)   ## luckily, this is already black!
# box = tuple((n - o) // 2 for n, o in zip(new_size, old_size))
# new_im.paste(old_im, box)
#
# new_im.show()
# new_im.save('someimage.jpg')
