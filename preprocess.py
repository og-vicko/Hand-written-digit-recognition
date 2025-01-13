# Importing Dependencies
import pandas as pd
import numpy as np
import re, os
import torch
import cv2
import glob
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from skimage import morphology
from skimage.metrics import structural_similarity as ssim
from collections import Counter
from cnn_model import CNN

model = CNN()
# model_save_path = r'C:\Users\Victor\Desktop\vicko\MSC_AI\Module projects\ML&_Computer_Vision\num_recg_model.pth'
# model.load_state_dict(torch.load(model_save_path))
# model.eval() 

with open ('num_recg_model.pth', 'rb') as f:
    state_dict = torch.load("num_recg_model.pth")
    model.load_state_dict(state_dict)
    model.eval()

def detect_background(image):
    """Detects if the image background is predominantly white or black.

    Args:
        image: The input image (NumPy array).

    Returns:
        "white" if the background is white, "black" otherwise.
    """

    avg_intensity = np.mean(image)

    if avg_intensity > 127:  # Threshold for considering background white
        return "white"
    else:
        return "black"


def remove_background_noise(img, min_size):
  kernel = np.ones((1, 50), np.uint8)  # Kernel size (1, 50) for horizontal line detection
  horizontal_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

  # Subtract the detected lines from the original image
  processed_img = cv2.subtract(img, horizontal_lines)
  # cv2_imshow(processed_img)
  print('\n-------------------------Remove Smaller Noise----------------------')

  processed_img = morphology.remove_small_objects(processed_img.astype(bool), min_size=min_size)#min_size=min connected pixels

  # Convert back to uint8 for display (if needed)
  processed_img = processed_img.astype(np.uint8) * 255

  return processed_img


def is_image_or_video(data):
    """Checks if the data is an image or video based on its type and shape.

    Args:
        data: The input data (file path, NumPy array, etc.).

    Returns:
        "image", "video", or None if the format is not recognized.
    """
    if isinstance(data, str):  # If it's a file path, check extension
        ext = os.path.splitext(data)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return "image"
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
            return "video"
        else:
            return None
    elif isinstance(data, np.ndarray):  # check if it's a NumPy array (likely an image)
        if data.ndim == 2 or (data.ndim == 3 and data.shape[2] in [3, 4]): # 2D OR 3D images with 3/4 color channels
            return "image"
        else:
            return None  # Handle other array shapes as needed
    elif isinstance(data, Image.Image):  # Check if it's a PIL Image
        return "image"        
    else:
        return None  # Handle unsupported data types if needed
    

def make_predict(model, image_tensors):
  predicted_classes = []
  number_string = ""

  for digit_tensor in image_tensors: # Use image_tensors directly
    digit = digit_tensor.unsqueeze(0)

    with torch.no_grad():  # This prevents unnecessary calculations for gradients
        output = model(digit)

    # Get the predicted class (index of the highest probability)
    predicted_class = torch.argmax(output, 1)
    predicted_classes.append(predicted_class.item())

  number_string = "".join(str(num) for num in predicted_classes)

  if number_string:
    print(f"Predicted Number: {number_string}")


def preprocess_image(image):
  print('###################################### PREPROCESSING IMAGE #######################################')
  transform = transforms.Compose([
      transforms.Grayscale(num_output_channels=1),  # Ensure the image is in grayscale
      transforms.Resize((28, 28)),  # Resize to match input size
      transforms.ToTensor(),  # Convert image to tensor
      transforms.Normalize((0.5,), (0.5,))  # Normalize to [0, 1] range
  ])

  # Apply the transformations
  img_pil = Image.fromarray(image)  # Convert numpy array to PIL Image
  input_image = transform(img_pil)

  # Add batch dimension (since model expects [batch_size, channels, height, width])
  input_image = input_image.unsqueeze(0)
  return input_image


def preprocess_digit_sequence_image(image, mnist_image=False, min_size=4000):
  print('###################################### PREPROCESSING IMAGE WITH DIGIT SEQUENCE #######################################')
  print(image.shape)

  gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  blurred_img = cv2.GaussianBlur(gray_img,(5,5),0)

  print('\n---------------------Remove BackGround Noise----------------------')
  thresh = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,

                                          cv2.THRESH_BINARY_INV, 51, 25)
  thresh = cv2.GaussianBlur(thresh,(25,25),0)
  # cv2_imshow(reduce_size(thresh))

  print('\n----------------------Further Noise Removal-----------------------')
  processed_img = remove_background_noise(thresh, min_size)
  # cv2_imshow(processed_img)

  # Find contours
  contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Filter out small contours based on area
  min_contour_area = 500  # Set appropriate threshold
  filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

  print(f"\nNumber of filtered contours: {len(filtered_contours)}")

  # Draw bounding boxes around each detected contour (digit)
  digit_bounding_boxes = []
  cont_count = 0 # conturs areas

  for contour in filtered_contours:
      x, y, w, h = cv2.boundingRect(contour)
      aspect_ratio = w / float(h)
      if 0.4 < aspect_ratio < 1.0 and w > 40 and h > 50:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cont_count+=1
        digit_bounding_boxes.append((x, y, w, h))
  print(f"Numbers detected: {cont_count}")

  # cv2_imshow(reduce_size(image))  # Show the image with bounding boxes

  # The bounding boxes may not be well ordered due to the conturs so we have to sort by x-coordinate --> left to right
  digit_bounding_boxes.sort(key=lambda box: box[0])

  # Extract and resize digits
  digit_images = []
  for (x, y, w, h) in digit_bounding_boxes:
      # digit_crop = thresh[y:y+h, x:x+w]
      digit_crop = processed_img[y:y+h, x:x+w]
      digit_resized = cv2.resize(digit_crop, (28, 28))
      digit_images.append(digit_resized)
      # cv2_imshow(digit_resized)

  # Normalize and convert to tensor
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
  digit_tensors = [transform(digit) for digit in digit_images]

  return digit_tensors, digit_images




def preprocess_image_with_balck_background(image, mnist_image=False):
  print('###################################### PREPROCESSING IMAGE WITH BLACK BACKGROUND #######################################')
  gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  blurred_img = cv2.GaussianBlur(gray_img,(5,5),0)

  print('\n---------------------Remove BackGround Noise----------------------')
  _, processed_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  # cv2_imshow(processed_img)

  # Find contours
  contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Filter out small contours based on area
  min_contour_area = 500  # Set appropriate threshold
  filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

  print(f"\nNumber of filtered contours: {len(filtered_contours)}")

  # Draw bounding boxes around each detected contour (digit)
  digit_bounding_boxes = []
  cont_count = 0 # conturs areas

  for contour in filtered_contours:
      x, y, w, h = cv2.boundingRect(contour)
      aspect_ratio = w / float(h)
      cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
      cont_count+=1
      digit_bounding_boxes.append((x, y, w, h))
  print(f"Numbers detected: {cont_count}")

  # cv2_imshow(reduce_size(image))  # Show the image with bounding boxes

  # The bounding boxes may not be well ordered due to the conturs so we have to sort by x-coordinate --> left to right
  digit_bounding_boxes.sort(key=lambda box: box[0])

  # Extract and resize digits
  digit_images = []
  for (x, y, w, h) in digit_bounding_boxes:
      # digit_crop = thresh[y:y+h, x:x+w]
      digit_crop = gray_img[y:y+h, x:x+w]
      digit_resized = cv2.resize(digit_crop, (28, 28))
      digit_images.append(digit_resized)
      # cv2_imshow(digit_resized)

  # Normalize and convert to tensor
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
  digit_tensors = [transform(digit) for digit in digit_images]

  return digit_tensors, digit_images


# Function to calculate similarity
def is_similar(frame1, frame2, threshold=0.9):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score >= threshold


def preprocess_video(video_path):
  cap = cv2.VideoCapture(video_path)

  frame_count = 0
  similarity_threshold = 0.9  # Similarity score threshold
  consecutive_frames = 3  # Number of consecutive frames to check similarity
  frame_buffer = []  # Buffer to store frames for comparison
  selected_frames = []
  selected_frames_count = 0

  print('********************************* EXTRACTING STABLE FRAMES **************************************')

  while cap.isOpened():
      success, frame = cap.read()
      if not success: # check if a frame was returned
          break
      # cv2_imshow(frame)
      frame_count += 1

      frame_buffer.append(frame) # Add frame to frame_buffer list

      # Check if buffer has enough frames
      if len(frame_buffer) >= consecutive_frames:
          stable = True

          # Compare all frames to the first frame in the buffer
          for i in range(1, len(frame_buffer)):
              if not is_similar(frame_buffer[0], frame_buffer[i], threshold=similarity_threshold):
                  stable = False
                  break

          if stable: # If the frames in the buffer are similar select it
              print(f"\nFrame {frame_count} is stable.")
              selected_frames.append(frame_buffer[-1])
              selected_frames_count+=1

          frame_buffer.clear()  # Clear the buffer for the next set of frames
  print(f"\nTotal frames in the video: {frame_count}")
  print(f'Selected Frames: {selected_frames_count}')

  cap.release()

  return selected_frames


def data_pipeline(image, mnist=False, video_path=None):
  """Processes an image or video for digit recognition using a trained CNN model.

  This function serves as the main pipeline for digit recognition. It takes an image
  or a video as input and performs the necessary preprocessing steps before
  feeding the data to the trained CNN model for prediction. It handles different
  input types (MNIST images, real-world images, and videos) and applies appropriate
  preprocessing techniques based on the input type.

  Args:
    image: The input image (NumPy array) or video path (string).
    mnist: A boolean indicating whether the input is an MNIST image.
            Defaults to False.

  Returns:
    None. It prints the predicted digit(s).

  Process:
    1. MNIST Image Handling: If `mnist` is True, it assumes the input is an MNIST image and
        applies the `preprocess_image` function to prepare it for the model. Then, it makes
        a prediction using the model and prints the result.
    2. Video Handling: If the input is a video, it first extracts stable frames using
        the `preprocess_video` function. For each stable frame, it applies the
        `preprocess_digit_sequence_image` function to extract and preprocess digits,
        and then uses the `make_predict` function to predict and print the digits.
    3. Real-world Image Handling: If the input is a real-world image, it first determines
        the background color (black or white). Based on the background color, it applies
        either `preprocess_image_with_balck_background` or `preprocess_digit_sequence_image`
        to preprocess the image. Finally, it uses the `make_predict` function to predict
        and print the digits.
  """
  # print(image, ':::::::::::::::::::::::::::::::::::::::::::::::')
  ############################################## MNIST #####################################################
  if mnist:
    processed_digit = preprocess_image(image)
    with torch.no_grad():  # This prevents unnecessary calculations for gradients
      output = model(processed_digit)

    predicted_class = torch.argmax(output, 1)
    print(f"Predicted Class: {predicted_class.item()}")

  ############################################## REAL WORLD VIDEO ################################################
  if is_image_or_video(image) == 'video' and not mnist:
    frames = preprocess_video(video_path)
    for frame in frames:
      image_tensors, detected_number = preprocess_digit_sequence_image(frame, min_size=50)
      make_predict(model, image_tensors)

  ############################################## REAL WORLD IMAGES ################################################
  if is_image_or_video(image) == 'image' and not mnist:
    print("got here 1::::::::::::::::::::::::::::::::::::::::")

    if detect_background(image) == 'black': # if image has a black background
      image_tensors, detected_number = preprocess_image_with_balck_background(image)
      make_predict(model, image_tensors)

    else:
      print("got here 2::::::::::::::::::::::::::::::::::::::::")
      image_tensors, detected_number = preprocess_digit_sequence_image(image)
      make_predict(model, image_tensors)
