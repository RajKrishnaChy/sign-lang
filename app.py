from function import *
from keras.models import model_from_json
import cv2
import numpy as np

# Load the model
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

# Replace this with the path to your notepad file
notepad_file_path = "C:/Users/dell/OneDrive/Desktop/output.txt"

colors = []
for i in range(0, 20):
    colors.append((245, 117, 16))
print(len(colors))

# 1. New detection variables
sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8

# Replace the image path with the path to your image
image_path ="C:/Users/dell/Downloads/0.png" 
frame = cv2.imread(image_path)

# Set mediapipe model
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    # Make detections
    cropframe = frame[40:400, 0:300]
    frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
    image, results = mediapipe_detection(cropframe, hands)

    # 2. Prediction logic
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)

    # Ensure that the sequence length reaches 15
    while len(sequence) < 15:
        # Make detections
        cropframe = frame[40:400, 0:300]
        image, results = mediapipe_detection(cropframe, hands)

        # Append keypoints to the sequence
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

    # Trim the sequence to maintain a fixed length of 15
    sequence = sequence[-15:]

    try:
        if len(sequence) == 15:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predicted_label = actions[np.argmax(res)]

            # Convert the output alphabet label into text
            alphabet_text = predicted_label if predicted_label.isalpha() else ""

            print(alphabet_text)
            predictions.append(np.argmax(res))

            # 3. Viz logic
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                            accuracy.append(str(res[np.argmax(res)] * 100))
                    else:
                        sentence.append(actions[np.argmax(res)])
                        accuracy.append(str(res[np.argmax(res)] * 100))

            if len(sentence) > 1:
                sentence = sentence[-1:]
                accuracy = accuracy[-1:]

            # Write the output alphabet to the notepad file
            with open(notepad_file_path, 'w') as notepad_file:
                notepad_file.write(alphabet_text)

    except Exception as e:
        # print(e)
        pass

    cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
    cv2.putText(frame, "Output: -" + ' '.join(sentence) + ''.join(accuracy), (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show to screen
    cv2.imshow('OpenCV Feed', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
