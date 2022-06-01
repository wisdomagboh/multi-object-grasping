# Multi-Object Grasping in the Plane

We consider the problem where multiple rigid convex polygonal objects rest in randomly placed positions and orientations on a planar surface visible from an overhead camera. The objective is to efficiently grasp and transport all objects into a bin. Specifically, we explore multi-object push-grasps where multiple objects are pushed together before the grasp can occur. We provide necessary conditions for multi-object push-grasps and apply these to filter inadmissible grasps in a novel multi-object grasp planner. We find that our planner is **19** times faster than a Mujoco simulator baseline. We also propose a picking algorithm that uses both single- and multi-object grasps to pick objects. In physical grasping experiments, compared to a single-object picking baseline, we find that the multi-object grasping system achieves **13.6\%** higher grasp success and is **59.9\%** faster.

Here, we provide the source code for our implementation.

More information can be found in our paper [ISRR 2022 (under review)](https://arxiv.org/abs/1903.08470)

<img src="mog_1.gif" scale="1.0"/>

## Getting Started

	1. Create and activate a virtual environment (Code was tested with Ubuntu 16.04 and python3.5)
		$  virtualenv -p /usr/bin/python3.5 venv ; source venv/bin/activate

	2. Install Physics Simulator Mujoco and dm_control in virtual env
		Follow instructions from Deepmind's dm_control project https://github.com/deepmind/dm_control.

	3. Install other required python packages
		$  pip install -r requirements.txt

	4. Clone this repo/ Download and extract zip file.
		$ git clone https://github.com/wisdomagboh/multi-object-grasping.git

	5. Run setup.py to place custom domains into 'suite'
		$  python3.5 setup/setup.py


## Running experiments

	1. Simulation experiments
    $ bash run_exps.sh

	2. Generate simulation results
 		$ python sim_plots_summary.py

  3. Physical picking experiments - launch two ros nodes.
    $ python3.5 rw_planner.py
    $ python3.5 arm_motion_generator.py 'grasp_type (MOG or SOG)' 'scene_number'

  4. Generate physical experimental results
    $ python rw_table_data.py


## Website

[ISRR 2022 (Under review)](sites.google.com/view/multi-object-grasping)


## Have a question?
For all queries please contact Wisdom Agboh (wisdomagboh@gmail.com)

## License
This project is licensed under the MIT License - see the
[LICENSE.md](LICENSE.md) file for details.
