from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mini_vlm.utils.model_loading import latest_epoch_adapter_path


class ModelLoadingTest(unittest.TestCase):
    def test_latest_epoch_adapter_path_uses_highest_epoch_number(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "visual_adapter_epoch_2.pt").write_bytes(b"")
            (root / "visual_adapter_epoch_10.pt").write_bytes(b"")
            (root / "visual_adapter_epoch_1.pt").write_bytes(b"")

            self.assertEqual(latest_epoch_adapter_path(root).name, "visual_adapter_epoch_10.pt")


if __name__ == "__main__":
    unittest.main()
