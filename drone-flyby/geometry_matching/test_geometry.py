import unittest
import numpy as np
from experiment import Camera,triangulate,rigid

class GeometryTests(unittest.TestCase):
    def test_projection_inverse_at_varied_heights(self):
        camera=Camera(3340,20,.7,13.8888889)
        points=np.array([[-100,500,0],[40,620,20],[130,580,70.]])
        uv=camera.project(points,25)
        recovered=camera.unproject(uv,25,points[:,2])
        np.testing.assert_allclose(recovered,points,atol=1e-9)
    def test_multiview_triangulation(self):
        camera=Camera(); point=np.array([100.,300.,35.]);frames=[1,5,10,15]
        uv=[camera.project(point,t)[0] for t in frames]
        np.testing.assert_allclose(triangulate(camera,uv,frames),point,atol=1e-8)
    def test_yaw_translation_without_scale(self):
        a=np.array([[0,0],[4,0],[2,3],[1,6.]])
        theta=.7;r=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        b=a@r.T+[15,800]
        fit,t=rigid(a,b)
        np.testing.assert_allclose(fit,r,atol=1e-10)
        np.testing.assert_allclose(t,[15,800],atol=1e-10)
    def test_forward_flight_moves_ground_down(self):
        cam=Camera();p=np.array([0,300,0.])
        self.assertGreater(cam.project(p,1)[0,1],cam.project(p,0)[0,1])
        nadir=cam.project([0,0,0],0)[0]
        self.assertAlmostEqual(nadir[1],1080+cam.f*np.tan(np.radians(cam.pitch)))

if __name__=='__main__':unittest.main()
