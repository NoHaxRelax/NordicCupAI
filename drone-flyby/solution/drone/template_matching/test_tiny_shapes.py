import json,tempfile,unittest
from pathlib import Path
import cv2,numpy as np
from .detector import sha,iou
from .tiny_shapes import TinyShapeDetector


class TinyShapeTests(unittest.TestCase):
    def test_fixed_shape_survives_brightness_and_rotation(self):
        with tempfile.TemporaryDirectory() as d:
            bank=Path(d);lab=np.full((32,26,3),(205,135,138),np.uint8)
            poly=np.array([[8,7],[13,7],[13,11],[18,11],[18,23],[7,23],[7,13],[8,13]],np.int32)
            cv2.fillPoly(lab,[poly],(190,117,151));image=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
            cv2.imwrite(str(bank/'sprite.png'),image)
            (bank/'manifest.json').write_text(json.dumps({'templates':[dict(id='shape',file='sprite.png',sha256=sha(bank/'sprite.png'),**{'class':'small_launcher'})]}))
            model=TinyShapeDetector(bank)
            self.assertEqual(model.detect(np.full((160,200,3),128,np.uint8)),[])
            for dark in [0,40]:
                for turns in [0,1,2,3]:
                    tile=np.rot90(lab,turns).copy();tile[:,:,0]-=dark
                    scene=np.full((160,200,3),(205-dark,135,138),np.uint8);h,w=tile.shape[:2];scene[60:60+h,80:80+w]=tile
                    rows=model.detect(cv2.cvtColor(scene,cv2.COLOR_LAB2BGR))
                    self.assertEqual(len(rows),1)
                    self.assertGreater(iou(rows[0]['bbox'],[80,60,80+w,60+h]),.5)
                    self.assertEqual(rows,model.detect(cv2.cvtColor(scene,cv2.COLOR_LAB2BGR)))


if __name__=='__main__':unittest.main()
