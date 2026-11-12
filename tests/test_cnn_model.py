import unittest
try:
    import torch
    from src.cnn_model import StormCNN
    HAVE = True
except Exception:
    HAVE = False


@unittest.skipUnless(HAVE, "PyTorch not installed (CNN code could not be executed where this project was written)")
class T(unittest.TestCase):
    def test_shapes_and_roundtrip(self):
        m = StormCNN(48).eval(); x = torch.randn(7, 48, 1)
        self.assertEqual(tuple(m(x).shape), (7,))
        import tempfile, os, shutil
        tmp_dir = tempfile.mkdtemp()
        try:
            p = os.path.join(tmp_dir, "m.pt")
            torch.save({"state_dict": m.state_dict(), "window": 48}, p)
            m2 = StormCNN(48)
            m2.load_state_dict(torch.load(p)["state_dict"])
            m2.eval()
            self.assertTrue(torch.allclose(m(x), m2(x)))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
