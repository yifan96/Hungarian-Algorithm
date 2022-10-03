
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rosbag
from matplotlib.patches import Circle, PathPatch
from matplotlib.ticker import FormatStrFormatter
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
import mpl_toolkits.mplot3d.art3d as art3d

def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid



bag = rosbag.Bag('/home/yifan/Articles/2022IROS/3agent_multitask.bag')
traj1 = []
traj2 = []
traj3 = []
traj4 = []

for topic, msg , t in bag.read_messages(topics='/hummingbird1/ground_truth/position'):
    traj1.append([msg.point.x,msg.point.y,msg.point.z])
for topic, msg , t in bag.read_messages(topics='/hummingbird2/ground_truth/position'):
    traj2.append([msg.point.x,msg.point.y,msg.point.z])
for topic, msg , t in bag.read_messages(topics='/hummingbird3/ground_truth/position'):
    traj3.append([msg.point.x,msg.point.y,msg.point.z])
#for topic, msg , t in bag.read_messages(topics='/hummingbird4/ground_truth/position'):
    #traj4.append([msg.point.x,msg.point.y,msg.point.z])
bag.close()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')


ax.set_xlabel('x-axis(m)')
ax.set_ylabel('y-axis(m)')
ax.set_zlabel('z-axis(m)')
ax.set_zticks([0.0, 1.0, 2.0, 3.0])
ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# first trial

ax.scatter([-0.11, -0.49, 1.62, -1.09, -0.77], [4.66, 3.62, 3.31, 2.54, 5.34], [1.38, 0.95, 0.87, 1.33, 1.37], c='r', marker='*', s=50)
# second trial
# ax.scatter([3.8, -3.1, -1.3, -6.5],[-1.1, -0.37, 2.8, 2.0],[0,0,0,0], c='k', marker='o')
# ax.scatter([7.7, 1.23, 1.48, -3.4],[-5.5, -1.0 , 6.22, 7.3],[1.5, 2.3, 1.0 ,0.8], c='r', marker='*', s=50)

ax.legend(loc="lower right", prop={'size': 13})
plt.gca().set_box_aspect((18, 18, 3.5))
plt.show()
