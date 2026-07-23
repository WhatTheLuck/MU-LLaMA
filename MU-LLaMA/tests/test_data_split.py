import unittest

from util.data_split import split_manifest, split_record_indices


class DataSplitTest(unittest.TestCase):
    def test_audio_grouped_test_split_has_no_leakage(self):
        records = [
            {"audio_id": f"audio_{audio}", "question_id": f"{audio}_{question}"}
            for audio in range(20)
            for question in range(3)
        ]
        split = split_record_indices(
            records, validation_fraction=0.1, test_fraction=0.1,
            seed=42, split_unit="audio",
        )
        self.assertEqual(
            sorted(split["train"] + split["validation"] + split["test"]),
            list(range(len(records))),
        )
        manifest = split_manifest(
            records, split, "audio", 42, validation_fraction=0.1,
            test_fraction=0.1,
        )
        self.assertTrue(all(value == 0 for value in manifest["group_overlap"].values()))
        self.assertGreater(manifest["splits"]["test"]["samples"], 0)

    def test_external_test_manifest_checks_audio_overlap(self):
        records = [{"audio_name": f"train_{index}.wav"} for index in range(10)]
        split = split_record_indices(records, 0.2, 0.0, 42, "audio")
        external = [{"audio_name": f"test_{index}.wav"} for index in range(4)]
        manifest = split_manifest(
            records, split, "audio", 42, 0.2, 0.0,
            external_test_records=external,
        )
        self.assertEqual(manifest["test_source"], "external")
        self.assertEqual(manifest["splits"]["test"]["samples"], 4)
        self.assertTrue(all(value == 0 for value in manifest["group_overlap"].values()))

    def test_split_is_deterministic_and_validates_fractions(self):
        records = [{"audio_id": f"audio_{index}"} for index in range(10)]
        first = split_record_indices(records, 0.2, 0.2, 7, "audio")
        second = split_record_indices(records, 0.2, 0.2, 7, "audio")
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, "below 1"):
            split_record_indices(records, 0.6, 0.4, 7, "audio")


if __name__ == "__main__":
    unittest.main()
