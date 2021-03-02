# action-conditional-rkn
Pytorch code for CoRL 2020 paper [Action-Conditional Recurrent Kalman Networks For Forward and Inverse Dynamics Learning](https://arxiv.org/abs/2010.10201)

Dependencies
--------------
* torch==1.3.1
* python 3.*

How to Train
-------------

**Forward Dynamics** - With ```action-conditional-rkn``` as the working directory execute the python script
```python experiments/PamForward/pneumaticArm.py```

**Inverse Dynamics** - With ```action-conditional-rkn``` as the working directory execute the python script
```python experiments/Franka/frankaArm.py```

Datasets
------------
Forward dynamics learning uses subset of muscular robot table tennis data from [here](https://musculartt.embodied.ml/).
Inverse dynamics learning uses inhouse data collected from Franka Emika Panda


