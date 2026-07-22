import tempfile
import unittest
from pathlib import Path

import numpy as np

from flowmimic.src.data.openpose import load_aist_openpose


class OpenPoseFpsCacheTest(unittest.TestCase):
    def test_cache_is_scoped_by_resampling_timebase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            openpose_dir = root / "openpose"
            cache_dir = root / "cache"
            openpose_dir.mkdir()
            pkl_path = root / "gBR_sBM_cAll_d01_mBR0_ch01.pkl"
            source_path = openpose_dir / "gBR_sBM_c01_d01_mBR0_ch01.npy"

            data = np.zeros((10, 25, 3), dtype=np.float32)
            data[..., 2] = 1.0
            data[:, :, 0] = np.arange(10, dtype=np.float32)[:, None]
            np.save(source_path, data)

            at_30, _ = load_aist_openpose(
                str(pkl_path),
                str(openpose_dir),
                src_fps=60,
                target_fps=30,
                cache_root=str(cache_dir),
                write_cache=True,
                camera="01",
            )
            at_60, _ = load_aist_openpose(
                str(pkl_path),
                str(openpose_dir),
                src_fps=60,
                target_fps=60,
                cache_root=str(cache_dir),
                write_cache=True,
                camera="01",
            )

            self.assertEqual(at_30.shape[0], 5)
            self.assertEqual(at_60.shape[0], 10)
            cache_names = sorted(path.name for path in cache_dir.rglob("*.npz"))
            self.assertEqual(len(cache_names), 2)
            self.assertTrue(any("60to30fps" in name for name in cache_names))
            self.assertTrue(any("60to60fps" in name for name in cache_names))


if __name__ == "__main__":
    unittest.main()
