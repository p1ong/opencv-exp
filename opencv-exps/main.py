import cv2
import numpy as np
import matplotlib.pyplot as plt

def setupTest(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Testing, {name}')  # Press ⌘F8 to toggle the breakpoint.
    cb_img = cv2.imread("images/checkerboard_color.png")
    coke_img = cv2.imread("images/coca-cola-logo.png")

    # Use matplotlib imshow()
    plt.imshow(cb_img)
    plt.title("mathplotlib imshow")
    plt.show()

    # User OpenCV imshow(), display until key is pressed
    window1 = cv2.namedWindow("Window 1")
    cv2.imshow(window1, coke_img)
    cv2.waitKey(0)                      # Display window until key is pressed (default)
    cv2.destroyWindow(window1)

def colourChannelExperiments():
    # Split the image into Blue, Green and Red components
    img_NZ_bgr = cv2.imread("images/New_Zealand_Lake.jpg",cv2.IMREAD_COLOR)
    b,g,r = cv2.split(img_NZ_bgr)

    # Display the channels
    plt.figure(figsize=[20,5])
    plt.subplot(141)
    plt.imshow(r,cmap='gray')
    plt.title("Red Channel")

    plt.subplot(142)
    plt.imshow(g,cmap='gray')
    plt.title("Green Channel")

    plt.subplot(143)
    plt.imshow(b,cmap='gray')
    plt.title("Blue Channel")

    #Merge the individual channels
    imgMerged = cv2.merge((b,g,r))
    plt.subplot(144)
    plt.imshow(imgMerged[:,:,::-1])
    plt.title("Merged Channel")
    plt.show()

def imageManipulation():
    # Read image as gray scale
#    cb_img = cv2.imread("images/checkerboard_18x18.png",0)
    # Set thecolour map for proper gray scale rendering
#    plt.imshow(cb_img, cmap='gray')
#    print(cb_img)
#    plt.show()

    # Example of numpy slicing to set multiple pixels
#    cb_img[2:4,2:4] = 200
#    plt.imshow(cb_img, cmap='gray')
#    plt.show()

    img_NZ_bgr = cv2.imread("images/New_Zealand_Boat.jpg", cv2.IMREAD_COLOR)
    # Reorder from BGR to RGB
    img_NZ_rgb = img_NZ_bgr[:,:,::-1]
 #   plt.imshow(img_NZ_rgb)
 #   plt.show()

    # Cropping using numpy array slicing
    cropped_region = img_NZ_rgb[200:400,300:600]
    plt.figure(figsize=[20, 3])
    plt.subplot(142)
    plt.imshow(cropped_region)
    plt.title("Cropped Image")

    resized_cropped = cv2.resize(cropped_region,None,fx=2,fy=2)
    plt.subplot(143)
    plt.imshow(resized_cropped)
    plt.title("Resized Image")
    plt.show()

    # Examples of flipping
    img_horz = cv2.flip(img_NZ_rgb,1)
    img_vert = cv2.flip(img_NZ_rgb, 0)
    img_both = cv2.flip(img_NZ_rgb,-1)

    plt.figure(figsize=[18,5])
    plt.subplot(141)
    plt.imshow(img_horz)
    plt.title("Horizontal Flip")
    plt.subplot(142)
    plt.imshow(img_vert)
    plt.title("Vertical Flip")
    plt.subplot(143)
    plt.imshow(img_both)
    plt.title("Both Flipped")
    plt.subplot(144)
    plt.imshow(img_NZ_rgb)
    plt.title("Original")
    plt.show()

def imageAnnotation():
    image = cv2.imread("images/Apollo_11_Launch.jpg", cv2.IMREAD_COLOR)

    # Not that the colour is a BGR and RGB, so red is the third value not the first
    cv2.circle(image,(900,500), 100, (0,0,255), thickness=5,lineType=cv2.LINE_AA )
    text = "Apollo 11 Saturn V Launch 16th July 1969"
    fontScale = 2.3
    fontFace = cv2.FONT_HERSHEY_PLAIN
    fontColour = (0,255,0)
    fontThickness = 2
    cv2.putText(image,text, (200,700), fontFace, fontScale, fontColour, fontThickness,lineType=cv2.LINE_AA)

    plt.imshow(image[:,:,::-1])
    plt.show()

def imageEnhancement():
#    img_bgr = cv2.imread("images/New_Zealand_Coast.jpg",cv2.IMREAD_COLOR)
#    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Adjust the image brightness using numpy matrix
#    matrix = np.ones(img_rgb.shape, dtype="uint8") * 50

#   img_rgb_brighter = cv2.add(img_rgb, matrix)
#    img_rgb_darker = cv2.subtract(img_rgb, matrix)

#    plt.figure(figsize=[18,5])
#    plt.subplot(131);plt.imshow(img_rgb_darker);plt.title("Darker")
#    plt.subplot(132);plt.imshow(img_rgb);plt.title("Orignal")
#    plt.subplot(133);plt.imshow(img_rgb_brighter);plt.title("Brigher")

#    plt.show()

    # These functions are used to extract features out of typically black and white images

    img_read = cv2.imread("images/Piano_Sheet_Music.png", cv2.IMREAD_GRAYSCALE)
    # Perform global thresholding, not that the threshold function returns two values
    retval, img_thresh_gbl_1 = cv2.threshold(img_read, 50, 255, cv2.THRESH_BINARY)
    retval, img_thresh_gbl_2 = cv2.threshold(img_read, 130,255, cv2.THRESH_BINARY)
    img_thresh_adp = cv2.adaptiveThreshold(img_read, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)

    plt.figure(figsize=[18,15])
    plt.subplot(221);plt.imshow(img_read, cmap="gray");plt.title("Original")
    plt.subplot(222);plt.imshow(img_thresh_gbl_1, cmap="gray");plt.title("Threshold (global:50)")
    plt.subplot(223);plt.imshow(img_thresh_gbl_2, cmap="gray");plt.title("Threshold (global:130")
    plt.subplot(224);plt.imshow(img_thresh_adp, cmap="gray");plt.title("Threshold (adaptive)")

    plt.show()

def camaraAccess():



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
#    print_hi('OpenCV experiments')
#    colourChannelExperiments()
#    imageManipulation()
#    imageAnnotation()
#    imageEnhancement()
    camaraAccess()