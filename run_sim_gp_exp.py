import numpy as np
import subprocess
import argparse

from resources import *

parser = argparse.ArgumentParser(description='Run-Sim-Experiments')
parser.add_argument('--num_objs', action="store", default=2)
args = parser.parse_args()
num_objs = int(args.num_objs)

data_path = './mog_data/meta/'

def convert_names_to_str(obj_names, obj_pts, obj_widths):

    obj_names_str = ""
    count_objs = 0
    for obj_n in obj_names:
        obj_names_str = obj_names_str+obj_n
        if count_objs < len(obj_names)-1:
            obj_names_str = obj_names_str+','
        count_objs += 1

    obj_pts_str = ""
    count_objs = 0
    for obj_p in obj_pts:
        obj_pts_str = obj_pts_str+str(obj_p)
        if count_objs < len(obj_pts)-1:
            obj_pts_str = obj_pts_str+','
        count_objs += 1

    obj_widths_str = ""
    count_objs = 0
    for obj_w in obj_widths:
        obj_widths_str = obj_widths_str+str(obj_w)
        if count_objs < len(obj_widths)-1:
            obj_widths_str = obj_widths_str+','
        count_objs += 1

    return obj_names_str, obj_pts_str, obj_widths_str

def two_objects():
    # Use only t1_objs
    list_of_obj_indexes = []
    obj_comb = 2
    all_obj_name_list = []
    for objs_comb_ind in range(num_x_obj_combinations):
        rand_indexes = np.random.permutation(len(t1_obj_list))[0:obj_comb].copy()
        obj_names = []
        obj_pts = []
        obj_widths = []
        for obj_ind_dum in range(obj_comb):
            obj_ind = rand_indexes[obj_ind_dum]
            obj_names.append(t1_obj_list[obj_ind])
            obj_pts.append(int(t1_obj_sizes[obj_ind]))
            obj_widths.append(t1_obj_widths[obj_ind])

        list_of_obj_indexes.append(rand_indexes)
        all_obj_name_list.append(obj_names)
        obj_names_str, obj_pts_str, obj_widths_str = convert_names_to_str(obj_names, obj_pts, obj_widths)
        subprocess.call(['python3', 'mog_h.py', '--obj_list', obj_names_str,
                        '--obj_pts', obj_pts_str, '--obj_widths', obj_widths_str])

    return list_of_obj_indexes, all_obj_name_list

def other_combinations(num_objs):

    all_lists = []
    all_obj_name_list = []
    # One object from t1
    for objs_comb_ind in range(num_x_obj_combinations):
        t1_list = []
        t2_list = []
        t3_list = []

        comb_mode = int(np.random.permutation([1,2,3])[0])

        if num_objs == 3:
            # Pick one each from t1, t2, and t3
            t1_objs = [np.random.permutation(len(t1_obj_list))[0]]
            t2_objs = [np.random.permutation(len(t2_obj_list))[0]]
            t3_objs = [np.random.permutation(len(t3_obj_list))[0]]
            t1_list.append(t1_objs)
            t2_list.append(t2_objs)
            t3_list.append(t3_objs)

        if num_objs == 4:
            t1_objs = [np.random.permutation(len(t1_obj_list))[0]]
            t2_objs = np.random.permutation(len(t2_obj_list))[0:2]
            t3_objs = [np.random.permutation(len(t3_obj_list))[0]]
            t1_list.append(t1_objs)
            t2_list.append(t2_objs)
            t3_list.append(t3_objs)

        if num_objs == 5:
            t1_objs = [np.random.permutation(len(t1_obj_list))[0]]
            t2_objs = np.random.permutation(len(t2_obj_list))[0:2]
            t3_objs = [np.random.permutation(len(t3_obj_list))[0]]
            t1_list.append(t1_objs)
            t2_list.append(t2_objs)
            t3_list.append(t3_objs)

        if num_objs == 6:
            t1_objs = []
            t2_objs = np.random.permutation(len(t2_obj_list))[0:4]
            t3_objs = np.random.permutation(len(t3_obj_list))[0:2]
            t1_list.append(t1_objs)
            t2_list.append(t2_objs)
            t3_list.append(t3_objs)

        if num_objs == 7:
            t1_objs = []
            t2_objs = np.random.permutation(len(t2_obj_list))[0:num_objs]
            t3_objs = []
            t1_list.append(t1_objs)
            t2_list.append(t2_objs)
            t3_list.append(t3_objs)

        obj_names = []
        obj_pts = []
        obj_widths = []

        print ('Mode is', comb_mode)

        if comb_mode == 1:
            for obj_index in list(t1_objs):
                obj_names.append(t1_obj_list[obj_index])
                obj_pts.append(int(t1_obj_sizes[obj_index]))
                obj_widths.append(t1_obj_widths[obj_index])

            for obj_index in list(t2_objs):
                obj_names.append(t2_obj_list[obj_index])
                obj_pts.append(int(t2_obj_sizes[obj_index]))
                obj_widths.append(t2_obj_widths[obj_index])

            for obj_index in list(t3_objs):
                obj_names.append(t3_obj_list[obj_index])
                obj_pts.append(int(t3_obj_sizes[obj_index]))
                obj_widths.append(t3_obj_widths[obj_index])
        elif comb_mode == 2:

            for obj_index in list(t2_objs):
                obj_names.append(t2_obj_list[obj_index])
                obj_pts.append(int(t2_obj_sizes[obj_index]))
                obj_widths.append(t2_obj_widths[obj_index])

            for obj_index in list(t1_objs):
                obj_names.append(t1_obj_list[obj_index])
                obj_pts.append(int(t1_obj_sizes[obj_index]))
                obj_widths.append(t1_obj_widths[obj_index])

            for obj_index in list(t3_objs):
                obj_names.append(t3_obj_list[obj_index])
                obj_pts.append(int(t3_obj_sizes[obj_index]))
                obj_widths.append(t3_obj_widths[obj_index])
        else:
            for obj_index in list(t3_objs):
                obj_names.append(t3_obj_list[obj_index])
                obj_pts.append(int(t3_obj_sizes[obj_index]))
                obj_widths.append(t3_obj_widths[obj_index])

            for obj_index in list(t1_objs):
                obj_names.append(t1_obj_list[obj_index])
                obj_pts.append(int(t1_obj_sizes[obj_index]))
                obj_widths.append(t1_obj_widths[obj_index])

            for obj_index in list(t2_objs):
                obj_names.append(t2_obj_list[obj_index])
                obj_pts.append(int(t2_obj_sizes[obj_index]))
                obj_widths.append(t2_obj_widths[obj_index])


        all_lists.append([t1_list, t2_list, t3_list])

        all_obj_name_list.append(obj_names)

        obj_names_str, obj_pts_str, obj_widths_str = convert_names_to_str(obj_names, obj_pts, obj_widths)
        subprocess.call(['python3', 'mog_h.py', '--obj_list', obj_names_str,
                        '--obj_pts', obj_pts_str, '--obj_widths', obj_widths_str])

    return all_lists, all_obj_name_list

def main():
    ''' Input is num_objs. The number of objects for which we will run sim
        grasp planning experiments. We randomly choose num_objs objects out of
        classes of t1_objs, t2_objs, and t3_objs.
        We do this num_x_obj_combination=20 times. Thereafter we call the
        mog_h.py code which creates 10 randomly generated scenes for each of the
        20 object combinations. Thus, we have 200 scenes per num_objs where we
        run simulation grasp planning experiments.
    '''

    if num_objs == 2:
        obj_ind_lists, all_obj_name_list = two_objects()
        np.save(data_path+'obj_ind_lists_{}'.format(num_objs), obj_ind_lists)
        np.save(data_path+'all_name_list_{}'.format(num_objs), all_obj_name_list)

    else:
        all_list, all_obj_name_list = other_combinations(num_objs)
        np.save(data_path+'all_list_{}'.format(num_objs), all_list)
        np.save(data_path+'all_name_list_{}'.format(num_objs), all_obj_name_list)

if __name__ == '__main__':
    main()
