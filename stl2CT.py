import pymesh
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import pyrender
from mpl_toolkits.mplot3d import Axes3D

path = '/home/srk1995/pub/db/kaist_vessel/'
CT = pymesh.load_mesh(path + 'hepatic_artery_200417.stl')
CT_tri = trimesh.load(path + 'hepatic_artery_200417.stl')

mesh = pyrender.Mesh.from_trimesh(CT_tri)
scene = pyrender.Scene()
scene.add(mesh)
pyrender.Viewer(scene)

scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0])
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_trisurf(CT.vertices[:, 0] * 1e-4, CT.vertices[:, 1] * 1e-4, CT.vertices[:, 2] * 1e-4, triangles=CT.faces, cmap=plt.cm.Spectral)
ax.set_zlim(-1, 1)
plt.xlabel('x-axis', labelpad=10)
plt.ylabel('y-axis', labelpad=10)

plt.show()

plt.triplot(CT.vertices[0] * 1e-4, CT.vertices[1] * 1e-4, CT.faces)