from unittest import TestCase


class TestProcessIO(TestCase):
    def test_process(self):
        from .processing import ImageProcess, ImageIO
        from .common import Image
        import numpy as np

        startImage = Image(src=np.array([[0,1,0], [1,1,1], [0,1,0]]))

        proc:ImageProcess = ImageIO().initProcess("Test Process")

        def not_image(_image):
            return ImageIO(image=np.logical_not(_image))

        proc.addFunction(not_image)

        proc.run(ImageIO(image=startImage))

        assert np.array_equalproc.resultImage
