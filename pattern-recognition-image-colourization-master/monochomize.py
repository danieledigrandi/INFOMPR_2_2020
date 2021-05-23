import os
import numpy as np
import cv2
from skimage import color, io
from matplotlib import pyplot as plt
from PIL import Image
from typing import Callable, List


def convertPILLOW(fileName: str):
    """
    Result similar to OPENCV; 0-255
    """
    return Image.open(fileName).convert('L')


def convertSKIMAGE(filename: str):
    """
    Result 0.0-1.0
    """
    return color.rgb2gray(io.imread(filename))


def convertOPENCV(filename: str):
    """
    Result similar to PILLOW; 0-255
    Already in Dataset.py
    """
    return cv2.imread(filename, 0)


def convertOPENCV_mean(filename: str):
    """
    The mean of each pixel
    """
    img = cv2.imread(filename)
    ans = np.array([[int(round(np.mean(b))) for b in a] for a in img])
    return ans


def convertLuminanceCorrected(filename: str):
    """ https://e2eml.school/convert_rgb_to_grayscale.html """
    img = cv2.imread(filename)
    ans = np.array([[convertPixel(b) for b in a] for a in img])
    return ans


def convertGammaLuminanceCorrected(filename: str):
    """ https://e2eml.school/convert_rgb_to_grayscale.html """
    img = cv2.imread(filename)
    ans = np.array([[convertPixel2(b) for b in a] for a in img])
    return ans


def convertPILLOW_np(inp: np.ndarray) -> np.ndarray:
    """
    Result similar to OPENCV; 0-255
    """
    return np.array(Image.fromarray(inp).convert('L'))


def convertSKIMAGE_np(inp: np.ndarray) -> np.ndarray:
    """
    Result 0.0-1.0
    """
    return (color.rgb2gray(inp) * 255).astype(np.uint8)


def convert_opencv_gray(inp: np.ndarray) -> np.ndarray:
    """
    Basic openCV gray conversion
    """
    return cv2.cvtColor(inp, cv2.COLOR_RGB2GRAY)


def convert_opencv_luminance(inp: np.ndarray) -> np.ndarray:
    """
    OpenCV luminance conversion in Lab mapping
    """
    lab_img: np.ndarray = cv2.cvtColor(inp, cv2.COLOR_BGR2Lab)
    return np.reshape(lab_img[:, :, 0], (lab_img.shape[0], lab_img.shape[1], 1))


def convert_mean_np(inp: np.ndarray) -> np.ndarray:
    """
    The mean of each pixel
    """
    ans = np.array([[int(round(np.mean(b))) for b in a] for a in inp])
    return ans


def convertLuminanceCorrected_np(inp: np.ndarray) -> np.ndarray:
    """ https://e2eml.school/convert_rgb_to_grayscale.html """
    ans = np.array([[convertPixel(b) for b in a] for a in inp])
    return ans


def convertGammaLuminanceCorrected_np(inp: np.ndarray) -> np.ndarray:
    """ https://e2eml.school/convert_rgb_to_grayscale.html """
    ans = np.array([[convertPixel2(b) for b in a] for a in inp])
    return ans


def convertPixel(inp):
    """ https://e2eml.school/convert_rgb_to_grayscale.html """
    return int(0.2126 * inp[0] + 0.7152 * inp[1] + 0.0722 * inp[2])


def convertPixel2(inp):
    """ https://e2eml.school/convert_rgb_to_grayscale.html """
    return int(0.299 * inp[0] + 0.587 * inp[1] + 0.114 * inp[2])


def showImage(inp):
    plt.imshow(inp, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()


def writeImageInt(img, name: str):
    """
    img: np.array with integers!
    name: location + filename + extension (.jpg, .png) No idea how much compression is used for jpg
    """
    cv2.imwrite(name, img)


def writeImageFloat(img, name: str):
    """
    img: np.array with floats!
    name: location + filename + extension (.jpg, .png) No idea how much compression is used for jpg
    """
    io.imsave(name, img)


def convertAndWrite(inputLoc: str = 'fromColour', outputLoc: str = 'monochrome', ext: str = ".jpg",
                    printFirst: bool = False):
    """
    converts all images in 'inputloc' folder to monochrome/grayscale,
    places them in 'outputloc'
    """
    firstSeen = False
    for filename in os.listdir(os.path.join(os.getcwd(), inputLoc)):
        temp = io.imread(f"{inputLoc}/{filename}")
        print(f"Type scipy {temp}")
        tempArr = np.array(temp)
        print(tempArr)
        outp = color.rgb2gray(tempArr)
        print(outp)
        print(color.rgb2gray(np.array([0, 0, 0])))

        filename_ = filename[:-4]
        print(filename_)

        # Float converted images
        ski_ = convertSKIMAGE(os.path.join(os.getcwd(), f"{inputLoc}/{filename}"))
        writeImageFloat(ski_, outputLoc + f"/{filename_}_skimage{ext}")

        # Int converted images
        pil_ = np.array(convertPILLOW(os.path.join(os.getcwd(), f"{inputLoc}/{filename}")))
        opencv_ = convertOPENCV(os.path.join(os.getcwd(), f"{inputLoc}/{filename}"))
        mean_ = convertOPENCV_mean(os.path.join(os.getcwd(), f"{inputLoc}/{filename}"))
        lumi_ = convertLuminanceCorrected(os.path.join(os.getcwd(), f"{inputLoc}/{filename}"))
        gam_ = convertGammaLuminanceCorrected(os.path.join(os.getcwd(), f"{inputLoc}/{filename}"))
        writeImageInt(pil_, f"{outputLoc}/{filename_}_pil{ext}")
        writeImageInt(opencv_, f"{outputLoc}/{filename_}_opencv{ext}")
        writeImageInt(mean_, f"{outputLoc}/{filename_}_mean{ext}")
        writeImageInt(lumi_, f"{outputLoc}/{filename_}_luminance{ext}")
        writeImageInt(gam_, f"{outputLoc}/{filename_}_gamma{ext}")
        if printFirst and not firstSeen:
            firstSeen = True
            print("Pil:")
            print(pil_)
            print("Opencv")
            print(opencv_)


monochomize_function_calls: List[Callable[[np.ndarray], np.ndarray]] = [
    convertPILLOW_np,
    convertSKIMAGE_np,
    convertLuminanceCorrected_np,
    convertGammaLuminanceCorrected_np,
    convert_mean_np,
    convert_opencv_gray,
    convert_opencv_luminance
]

monochomize_function_names: List[str] = [
    "pillow",
    "skimage",
    "lum_corr",
    "gamma_lum_corr",
    "mean",
    "opencv_gray",
    "opencv_lum"
]

assert len(monochomize_function_calls) == len(monochomize_function_names)

monochomize_methods = len(monochomize_function_names)

if __name__ == "__main__":
    convertAndWrite(printFirst=False)
