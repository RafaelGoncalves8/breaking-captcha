import imutils
import cv2

def trim(image, thres):
    """
    A helper function to delete blank rows of image.
    image: image to trim
    thres: threshold to be considered blank
    """
    return image[np.all(image >= thres, axis=1)]

def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    image: image to resize
    width: desired width in pixels
    height: desired height in pixels
    return: the resized image
    """

    WHITE = [int(image[0,0]),int(image[0,0]),int(image[0,0])]

    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_CONSTANT,value=WHITE)
    image = cv2.resize(image, (width, height))

    # return the pre-processed image
    return image
