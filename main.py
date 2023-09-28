# importing required libraries
import numpy as np
import cv2
import matplotlib.pylab as plt

# Function to create a mask for the region of interest in the image
def region_of_interest(img, vertices):

    mask = np.zeros_like(img) 
    
    # Filling the specified region (defined by 'vertices') in the mask with white color (255)
    cv2.fillPoly(mask, vertices, 255) 
    
    # Applying the mask to the input image using a bitwise AND operation
    masked_image = cv2.bitwise_and(img, mask)
    
    return masked_image


# Function to draw lines on a blank image and overlay them on the original image
def draw_the_lines(img, lines):

    img = np.copy(img) 
    
    # Creating a blank image with the same dimensions as the input image
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # Iterating through the lines detected in the image
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Draw a green line on the blank image
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)

    # Overlaying the lines on the original image with a specified weight
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


# Function to process the input image
def process(image):

    height = image.shape[0]
    width = image.shape[1]

    # Defining the vertices of the region of interest as a triangle
    region_of_interest_vertices = [ (0, height), (width/2, height/2), (width, height)]

    # Converting the input image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Applying the Canny edge detection algorithm to the grayscale image
    canny_image = cv2.Canny(gray_image, 100, 120)

    # Cropping the Canny edge-detected image using the defined region of interest
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

    # Detecting lines in the cropped image using the Hough Line Transform
    lines = cv2.HoughLinesP(cropped_image,  # input image in which lines will be detected
                            rho=2,  # resolution of the accumulator in pixels
                            theta=np.pi/180,  # angle resolution of the accumulator in radians
                            threshold=50,  # threshold parameter
                            lines=np.array([]),  # numpy array to store the detected line segments
                            minLineLength=40,  # minimum length of a line segment
                            maxLineGap=100  # maximum allowed gap between line segments
                            ) 

    # Drawing the detected lines on the original image
    image_with_lines = draw_the_lines(image, lines)

    return image_with_lines


# Opening a video file for reading
cap = cv2.VideoCapture('video.mp4')  # video obtained from kaggle 

while cap.isOpened():
    ret, frame = cap.read()

    # Current frame processing using the 'process' function
    frame = process(frame)

    # Displaying the processed frame
    cv2.imshow('frame', frame)

    # Loop break when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Releasing the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
