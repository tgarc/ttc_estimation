ttc_estimation
====================
focus of expansion estimation + time to contact estimation algorithm.


Notes on Robot Operating System (ROS) 
-------------------------------------
Basic building

1. Make sure paths ROS_PACKAGE_PATH is set
2. Create project folder (e.g., ~/catkin_ws)
3. Place source of dependent projects in catkin_ws/src
4. cd catkin_ws -> catkin_make

Running tum_ardrone
1. source /opt/ros/hydro/setup.bash
2. source <tum-ardrone-workspace>/install/setup.bash

Setting up virtualbox ssh server
1. sudo apt-get install open-ssh
2. Go to virtualbox main preferences
    i. Click network
    ii. Click Host-only networks tab
    iii. Add an adapter
3. Go to machine settings
    i. Click network
    ii. Click Adapter 2 tab
    iii. Set "Attached to" to Host-Only Adapter
    iv. Restart machine
4. Edit /etc/network/interfaces and add something similar to the following
> auto eth1
iface eth1 inet static
address 192.168.56.2
netmask 255.255.255.0
network 192.168.56.0
broadcast 192.168.56.255
5. Open /etc/hosts _inside the host machine_ and enter the line ><IP>    <hostname>

Previous works
--------------

Camus: Iterative search for "mean" location of flow (center of OF mass)

Match filter: matched focus of expansion filter for angular component
        Pros
        + Doesn't depend on magnitude!
        + weighting by participating components improves robustness
        + a radially increasing weighting may improve even further
        + invariant to rotations

        Issues
        + large search space (but can limit to small search space once found)
        + depends heavily on textured environment
