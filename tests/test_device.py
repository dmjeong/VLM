from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch가 설치된 환경에서만 device 선택 테스트를 실행합니다.")
class DeviceSelectionTest(unittest.TestCase):
    def test_explicit_cpu_device_is_respected(self) -> None:
        from mini_vlm.utils.device import select_torch_device

        self.assertEqual(str(select_torch_device("cpu")), "cpu")


if __name__ == "__main__":
    unittest.main()
