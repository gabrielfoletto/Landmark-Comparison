import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(image):
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        else:
            return None

def visualize_landmarks(image, landmarks):
    if landmarks is None:
        return None

    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION | mp_face_mesh.FACEMESH_CONTOURS | mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
    )

    return annotated_image

def compare_landmarks(landmarks1, landmarks2, threshold=0.2):
    if landmarks1 is None or landmarks2 is None:
        return 0

    distances = []
    for landmark1, landmark2 in zip(landmarks1.landmark, landmarks2.landmark):
        dx = landmark1.x - landmark2.x
        dy = landmark1.y - landmark2.y
        distance = (dx ** 2 + dy ** 2) ** 0.5
        distances.append(distance)

    avg_distance = sum(distances) / len(distances)

    similarity = 1 - avg_distance / threshold
    return similarity

def main():
    IMAGE_FILE1 = 'imagetest5.jpg'  # Path to the first image
    IMAGE_FILE2 = 'imagetest9.jpg'  # Path to the second image

    # Load images
    image1 = cv2.imread(IMAGE_FILE1)
    image2 = cv2.imread(IMAGE_FILE2)

    # Extract landmarks
    landmarks1 = extract_landmarks(image1)
    landmarks2 = extract_landmarks(image2)

    # Visualize landmarks on images for inspection
    annotated_image1 = visualize_landmarks(image1, landmarks1)
    annotated_image2 = visualize_landmarks(image2, landmarks2)

    # Display annotated images (for debugging purposes)
    cv2.imshow('Annotated Image 1', annotated_image1)
    cv2.imshow('Annotated Image 2', annotated_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Compare landmarks and calculate similarity
    similarity = compare_landmarks(landmarks1, landmarks2)
    print(f"The similarity between the landmarks is {similarity:.2%}")

    if similarity < 0:
        print("Warning: Negative similarity value. Check landmark matching and threshold.")

if __name__ == '__main__':
    main()
