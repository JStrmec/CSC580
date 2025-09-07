import os
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import face_recognition
from PIL import Image, ImageDraw

RESOURCES_DIR = os.getenv("RESOURCES_DIR", "face_recognizer/resources")


class FaceDatabase:
    """Manages loading and storing known faces with their encodings."""

    def __init__(self, resource_dir: str = None):
        self.resource_dir = resource_dir or RESOURCES_DIR
        self.known_encodings: List = []
        self.known_names: List[str] = []

    def add_person(self, name: str, image_path: str) -> None:
        """Add a known person from an image."""
        full_path = os.path.join(self.resource_dir, image_path)
        image = face_recognition.load_image_file(full_path)
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            print(f"[WARN] No face found for {name} in {image_path}")
            return
        self.known_encodings.append(encodings[0])
        self.known_names.append(name)

    def load_people(self, people: Dict[str, str]) -> None:
        """Load multiple people. people = {name: filename}"""
        for name, path in people.items():
            self.add_person(name, path)


import face_recognition
from PIL import Image, ImageDraw


class FaceRecognizer:
    """Recognizes unknown faces using a known database."""

    def __init__(self, database=None, tolerance: float = 0.6):
        self.database = database
        self.tolerance = tolerance

    def recognize_faces(
        self, image_path: str
    ):
        """Return a list of recognized faces with bounding boxes."""
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=2)
        unknown_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)

        results = []
        if not self.database:
            return results

        for encoding, location in zip(unknown_encodings, face_locations):
            matches = face_recognition.compare_faces(self.database.known_encodings, encoding, self.tolerance)
            name = "Unknown"
            if True in matches:
                idx = matches.index(True)
                name = self.database.known_names[idx]
            results.append((name, location))
        return results

    def is_person_in_group(
        self, known_path: str, group_path: str, output_path: str = "comparison_result.jpg"
    ) -> bool:
        """Check if the person in known_path exists in group_path.
        Saves an output image showing both images side by side with annotations."""
        # Encode known person
        known_image = face_recognition.load_image_file(known_path)
        known_encodings = face_recognition.face_encodings(known_image)
        if not known_encodings:
            raise ValueError("No face found in known image.")
        known_encoding = known_encodings[0]

        # Encode group faces
        group_image = face_recognition.load_image_file(group_path)
        group_face_locations = face_recognition.face_locations(group_image)
        group_encodings = face_recognition.face_encodings(group_image, group_face_locations)

        # Compare
        results = face_recognition.compare_faces(group_encodings, known_encoding, tolerance=self.tolerance)
        match_found = True in results if results else False

        # Draw bounding boxes
        pil_group = Image.fromarray(group_image)
        draw = ImageDraw.Draw(pil_group)
        for (top, right, bottom, left), is_match in zip(group_face_locations, results):
            color = "green" if is_match else "red"
            draw.rectangle([left, top, right, bottom], outline=color, width=3)

        # Label known image
        pil_known = Image.fromarray(known_image)
        draw_known = ImageDraw.Draw(pil_known)
        draw_known.text((10, 10), "Known Person", fill="blue")

        # Combine both images side by side
        total_width = pil_known.width + pil_group.width
        max_height = max(pil_known.height, pil_group.height)
        combined = Image.new("RGB", (total_width, max_height), (255, 255, 255))
        combined.paste(pil_known, (0, 0))
        combined.paste(pil_group, (pil_known.width, 0))

        # Save output
        combined.save(output_path)
        print(f"Result saved to {output_path}")

        return match_found


class FaceSimilarityFinder:
    """Finds the most similar face to a reference image in a directory."""

    def __init__(self, tolerance: float = 1.0):
        self.tolerance = tolerance
        self.best_face_distance: float = tolerance
        self.best_face_image: Optional[Image.Image] = None
        self.best_match_path: Optional[Path] = None

    def find_most_similar(
        self, known_image_path: str, search_dir: str, image_ext: str = "*.png"
    ) -> Tuple[Optional[Image.Image], float, Optional[Path]]:
        known_image = face_recognition.load_image_file(known_image_path)
        known_encodings = face_recognition.face_encodings(known_image)
        if not known_encodings:
            raise ValueError(f"No face found in reference image {known_image_path}")
        known_encoding = known_encodings[0]

        self.best_face_distance = self.tolerance
        self.best_face_image = None
        self.best_match_path = None

        for image_path in Path(search_dir).glob(image_ext):
            unknown_image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(unknown_image)
            if not encodings:
                continue

            face_distance = face_recognition.face_distance(encodings, known_encoding)[0]
            if face_distance < self.best_face_distance:
                self.best_face_distance = face_distance
                self.best_face_image = Image.fromarray(unknown_image)
                self.best_match_path = image_path

        return self.best_face_image, self.best_face_distance, self.best_match_path

    def show_best_match(self) -> None:
        if self.best_face_image:
            self.best_face_image.show()
        else:
            print("No matching face found within tolerance.")


class FaceAnalyzer:
    """Analyzes facial features such as landmarks."""

    @staticmethod
    def extract_landmarks(image_path: str):
        image = face_recognition.load_image_file(image_path)
        return face_recognition.face_landmarks(image)


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


if __name__ == "__main__":
    # Build database
    db = FaceDatabase()
    db.load_people(
        {
            "Person 1": "person_1.jpg",
            "Person 2": "person_2.jpg",
            "Person 3": "person_3.jpg",
        }
    )

    # Recognize in new image
    recognizer = FaceRecognizer(db)
    unknown_image = RESOURCES_DIR + "/unknown_7.jpg"
    people_image = RESOURCES_DIR + "/people.jpg"
    test_image = RESOURCES_DIR + "/test_face.jpg"
    faces_found = recognizer.recognize_faces(unknown_image)
    print(faces_found)

    # Visualize bounding boxes
    FaceVisualizer.draw_faces(unknown_image, faces_found)

    # Landmarks
    landmarks = FaceAnalyzer.extract_landmarks(people_image)
    FaceVisualizer.draw_landmarks(people_image, landmarks)

    # Similarity search
    finder = FaceSimilarityFinder()
    peoples_image_dir = RESOURCES_DIR + "/people"
    best_img, distance, path = finder.find_most_similar(
        known_image_path=test_image, search_dir=peoples_image_dir
    )

    if best_img:
        print(f"Best match found: {path} with distance {distance:.4f}")
        finder.show_best_match()
    else:
        print("No similar face found.")

    # Paths
    known_path = RESOURCES_DIR + "/expected_person.jpg"  # single face
    group_path = RESOURCES_DIR + "/expected_person_not_in_group.jpg"  # group of faces

    ecognizer = FaceRecognizer()
    found = recognizer.is_person_in_group(
        known_path=known_path,
        group_path=group_path,
        output_path="person_vs_group.jpg"
    )

    if found:
        print("✅ Person IS in the group.")
    else:
        print("❌ Person is NOT in the group.")
