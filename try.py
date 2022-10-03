
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rosbag
from matplotlib.patches import Circle, PathPatch
from matplotlib.ticker import FormatStrFormatter
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time

# get the start time
st =  time.process_time()

# bag = rosbag.Bag('/home/yifan/05-16-2agent2task2.bag')
# traj1 = []
# traj2 = []
# traj3 = []
#
# for topic, msg , t in bag.read_messages(topics='/pixy/vicon/demo_crazyflie4/demo_crazyflie4/odom'):
#     traj1.append([msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z])
# for topic, msg , t in bag.read_messages(topics='/pixy/vicon/demo_crazyflie8/demo_crazyflie8/odom'):
#     traj2.append([msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z])
# # for topic, msg , t in bag.read_messages(topics='/pixy/vicon/demo_crazyflie3/demo_crazyflie3/odom'):
# #     traj3.append([msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z])
# bag.close()



# bag = rosbag.Bag('/home/yifan/05-16-2agent5task2.bag')
# traj1 = []
# traj2 = []
# traj3 = []
#
# for topic, msg , t in bag.read_messages(topics='/pixy/vicon/demo_crazyflie4/demo_crazyflie4/odom'):
#     traj1.append([msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z])
# for topic, msg , t in bag.read_messages(topics='/pixy/vicon/demo_crazyflie8/demo_crazyflie8/odom'):
#     traj2.append([msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z])
# # for topic, msg , t in bag.read_messages(topics='/pixy/vicon/demo_crazyflie3/demo_crazyflie3/odom'):
# #     traj3.append([msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z])
# bag.close()

# bag = rosbag.Bag('/home/yifan/05-16-2agent5task2.bag')
bag = rosbag.Bag('/home/yifan/05-16-3agent6task3.bag')
traj1 = []
traj2 = []
traj3 = []

for topic, msg , t in bag.read_messages(topics='/pixy/vicon/demo_crazyflie3/demo_crazyflie3/odom'):
    traj1.append([msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z])
for topic, msg , t in bag.read_messages(topics='/pixy/vicon/demo_crazyflie6/demo_crazyflie6/odom'):
    traj2.append([msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z])
for topic, msg , t in bag.read_messages(topics='/pixy/vicon/demo_crazyflie8/demo_crazyflie8/odom'):
    traj3.append([msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z])
bag.close()


fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

# print(traj_gt1[:,0])
#
# print(np.array(traj1)[:,0])




# first trial
#ax.scatter([6.78, -1.16, 2.24], [-2.74, -2.23, 6.79], [0,0,0], c='k', marker='o')
#ax.scatter([-0.5731, -1.1492, -2.46, 2.4141, 0.5223, 1.60, 1.20], [3.8158, 2.4853, 4.4835, 4.3213, 4.53, 3.37, 5.11], [0.8, 0.8, 1.3, 0.7, 0.9, 1.2, 0.9], c='r', marker='*', s=50)
# second trial
# ax.scatter([3.8, -3.1, -1.3, -6.5],[-1.1, -0.37, 2.8, 2.0],[0,0,0,0], c='k', marker='o')
# ax.scatter([7.7, 1.23, 1.48, -3.4],[-5.5, -1.0 , 6.22, 7.3],[1.5, 2.3, 1.0 ,0.8], c='r', marker='*', s=50)

# 2agents2tasks
# ax.scatter([1.2, 1.8],[3.0, 3.0],[0.0, 0.0], c='k', marker='o', s=66)
# ax.text(1.25, 2.95, 0, 'A1',color='k',fontsize=15)
# ax.text(1.85, 2.95, 0, 'A2',color='k',fontsize=15)
# ax.scatter([-1.6, 0.93],[5.42, 5.04],[0.8, 0.8], c='r', marker='*', s=66)
# zdirs = ('T1', 'T2')
# xs = (-1.6, 0.93)
# ys = (5.42, 5.04)
# zs = (0.8, 0.8)
#
# for zdir, x, y, z in zip(zdirs, xs, ys, zs):
#     label = '%s' % zdir
#     ax.text(x+0.05, y+0.05, z, label,color='red',fontsize=15)


# # 2agents5tasks
# ax.scatter([1.2, 1.8],[3.0, 3.0],[0.0, 0.0], c='k', marker='o', s=66)
# ax.text(1.25, 2.95, 0, 'A1',color='k',fontsize=15)
# ax.text(1.85, 2.95, 0, 'A2',color='k',fontsize=15)
# ax.scatter([-0.77, -1.3, -0.18, -0.13, -0.3],[3.65, 5.35, 6.45, 1.77, 4],[0.8, 0.8, 0.8, 0.8, 0.8], c='r', marker='*', s=66)
# zdirs = ('T1', 'T2', 'T3', 'T4', 'T5')
# xs = (-0.77, -1.3, -0.18, -0.13, -0.3)
# ys = (3.65, 5.35, 6.45, 1.77, 4)
# zs = (0.8, 0.8, 0.8, 0.8, 0.8)
#
# for zdir, x, y, z in zip(zdirs, xs, ys, zs):
#     label = '%s' % zdir
#     ax.text(x+0.05, y+0.05, z, label,color='red',fontsize=15)

#3agent6task
ax.scatter([-0.13, -1.3, -0.18],[1.77, 5.35, 6.45],[0.0, 0.0, 0.0], c='k', marker='o', s=66)
ax.text(-0.38, 1.62, 0, 'A1',color='k',fontsize=15)
ax.text(-1.50, 5.25, 0, 'A2',color='k',fontsize=15)
ax.text(-0.38, 6.35, 0, 'A3',color='k',fontsize=15)
ax.scatter([-0.85, 0.93, -0.3, -0.77, 0.7, 1.24],[5, 5.04, 4, 3.65, 3.5, 3],[0.8, 0.8, 0.8, 0.8, 0.8, 0.8], c='r', marker='*', s=66)
zdirs = ('T1', 'T2', 'T3', 'T4', 'T5', 'T6')
xs = (-0.85, 0.93, -0.3, -0.77, 0.7, 1.24)
ys = (5, 5.04, 4, 3.65, 3.5, 3)
zs = (0.8, 0.8, 0.8, 0.8, 0.8, 0.8)

for zdir, x, y, z in zip(zdirs, xs, ys, zs):
    label = '%s' % zdir
    ax.text(x+0.05, y+0.05, z, label,color='red',fontsize=15)




ax.plot(np.array(traj1)[:,0],np.array(traj1)[:,1],np.array(traj1)[:,2], label='agent 1', color='purple')
ax.plot(np.array(traj2)[:,0],np.array(traj2)[:,1],np.array(traj2)[:,2], label='agent 2')
ax.plot(np.array(traj3)[:,0],np.array(traj3)[:,1],np.array(traj3)[:,2], label='agent 3')
# ax.scatter(traj_gt1[:,0],traj_gt1[:,1],traj_gt1[:,2])
# ax.scatter(traj_gt2[:,0],traj_gt2[:,1],traj_gt2[:,2])
#ax.plot(traj_gt3[:,0],traj_gt3[:,1],traj_gt3[:,2])
ax.legend(loc="upper right", prop={'size': 20})
# plt.gca().set_box_aspect((18, 18, 3.5))
ax.set_xlabel('X axis',fontsize=15)
ax.set_ylabel('Y axis',fontsize=15)
ax.set_zlabel('Z axis',fontsize=15)
plt.show()




# main program
# find sum to first 1 million numbers
sum_x = 0
for i in range(1000000):
    sum_x += i

# wait for 3 seconds
#time.sleep(3)
#print('Sum of first 1 million numbers is:', sum_x)

# get the end time
et =  time.process_time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')