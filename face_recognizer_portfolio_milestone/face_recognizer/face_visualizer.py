from typing import List, Tuple
import face_recognition
from PIL import Image, ImageDraw


class FaceVisualizer:
    """Draws bounding boxes and facial features on images."""

    @staticmethod
    def draw_faces(
        image_path: str, faces: List[Tuple[str, Tuple[int, int, int, int]]]
    ) -> None:
        image = face_recognition.load_image_file(image_path)
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        for name, (top, right, bottom, left) in faces:
            draw.rectangle([left, top, right, bottom], outline="red", width=3)
            draw.text((left, top - 10), name, fill="red")

        pil_image.show()

    @staticmethod
    def draw_landmarks(image_path: str, landmarks) -> None:
        image = face_recognition.load_image_file(image_path)
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image, "RGBA")

        for face_landmarks in landmarks:
            for feature, points in face_landmarks.items():
                if "eyebrow" in feature or "lip" in feature:
                    draw.line(points, fill=(128, 0, 128, 150), width=2)

        pil_image.show()
