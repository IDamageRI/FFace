import os
import cv2
from Fface_recognition import load_face_descriptors, compare_faces, get_face_landmarks

def process_video(video_path, face_descriptors, faces):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:  # Обработка каждого 30-го кадра
            try:
                min_dist, closest_face, is_match = compare_faces(frame, face_descriptors, faces)
                if is_match:
                    print(f'Полученное лицо совпадает с лицом на фото: {closest_face}')
                else:
                    print('На фотографиях разные люди!')
            except Exception as e:
                print(f"Error processing frame: {e}")

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_webcam(face_descriptors, faces):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            min_dist, closest_face, is_match = compare_faces(frame, face_descriptors, faces)
            closest_face_img_path = os.path.join('face_bd', closest_face)
            closest_face_img, _ = get_face_landmarks(closest_face_img_path)

            if is_match:
                print(f'Полученное лицо совпадает с лицом на фото: {closest_face}')
            else:
                print('На фотографиях разные люди!')

            img, landmarks = get_face_landmarks(frame)

            for landmarks_set in landmarks:
                for (x, y) in landmarks_set:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Resize the closest face image to match the frame size
            closest_face_img = cv2.resize(closest_face_img, (frame.shape[1], frame.shape[0]))

            # Combine frames side by side
            combined_frame = cv2.hconcat([frame, closest_face_img])
            cv2.imshow('Webcam and Most Similar Face', combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"Error processing frame: {e}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    base_path = 'face_bd'
    face_descriptors, faces = load_face_descriptors(base_path)
    print(f"Loaded {len(faces)} faces from the directory {base_path}.")

    # Uncomment the line below to process a video file
    process_video('path_to_video.avi', face_descriptors, faces)

    # Uncomment the line below to process webcam feed
    process_webcam(face_descriptors, faces)
