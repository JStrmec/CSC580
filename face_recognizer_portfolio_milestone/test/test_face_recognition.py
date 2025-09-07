import unittest
import numpy as np
from PIL import Image
from face_recognizer.main import (
    FaceDatabase,
    FaceRecognizer,
    FaceSimilarityFinder,
    FaceAnalyzer,
)


class TestFaceRecognitionSystem(unittest.TestCase):
    def setUp(self):
        # Create a dummy black square image (no face)
        self.test_image_path = "test_blank.jpg"
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        img.save(self.test_image_path)

    def test_database_add_person_no_face(self):
        db = FaceDatabase(resource_dir=".")
        db.add_person("Nobody", self.test_image_path)
        self.assertEqual(len(db.known_encodings), 0)

    def test_recognizer_no_face(self):
        db = FaceDatabase(resource_dir=".")
        recognizer = FaceRecognizer(db)
        results = recognizer.recognize_faces(self.test_image_path)
        self.assertEqual(results, [])

    def test_similarity_no_face(self):
        finder = FaceSimilarityFinder()
        with self.assertRaises(ValueError):
            finder.find_most_similar(self.test_image_path, ".")

    def test_analyzer_no_face(self):
        landmarks = FaceAnalyzer.extract_landmarks(self.test_image_path)
        self.assertEqual(landmarks, [])
