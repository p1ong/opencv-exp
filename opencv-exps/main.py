import cv2
import matplotlib.pyplot as plt

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('OpenCV experiments')