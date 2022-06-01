import numpy as np

rw_data_path = './rw_data/'
max_data_ind = 50
total_num_objects = 33

objects_left_mog_list = np.load(rw_data_path+'objects_left_mog_list.npy')
objects_left_sog_list = np.load(rw_data_path+'objects_left_sog_list.npy')
failed_attempts_mog_list = np.load(rw_data_path+'failed_attempts_mog_list.npy')
failed_attempts_sog_list = np.load(rw_data_path+'failed_attempts_sog_list.npy')

def robot_times (scene_number, grasp_type):
    total_scene_exec_time = []
    total_scene_plan_time = []

    for attempt_number in range(max_data_ind):
        try:
            exec_time = np.load(rw_data_path+'scene_{}/exec_time_{}_{}.npy'.format(scene_number, grasp_type, attempt_number))
            total_scene_exec_time.append(float(exec_time))
            plan_time = np.load(rw_data_path+'scene_{}/grasp_plan_time_{}_{}.npy'.format(scene_number, grasp_type, attempt_number))
            total_scene_plan_time.append(float(plan_time))
        except:
            break

    return total_scene_exec_time, total_scene_plan_time, attempt_number

def main():

    num_scenes = 20

    all_mog_plan_time = []
    all_mog_e_time = []
    all_sog_plan_time = []
    all_sog_e_time = []
    all_mog_attempt_number = []
    all_sog_attempt_number = []

    all_mog_total_time = []
    all_sog_total_time = []

    all_mog_success = []
    all_sog_success = []

    all_mog_pph = []
    all_sog_pph = []
    all_mog_percent_cleared = []
    all_sog_percent_cleared = []

    all_mog_failed = []
    all_sog_failed = []

    for scene_number in range(num_scenes):
        grasp_type = 'MOG'
        e_times, p_times, attempt_number = robot_times(scene_number, grasp_type)

        if attempt_number != 0:

            all_mog_plan_time.append(np.sum(p_times))
            all_mog_e_time.append(np.sum(e_times))
            all_mog_attempt_number.append(attempt_number)
            all_mog_failed.append(failed_attempts_mog_list[scene_number])
            mog_pick_time = np.sum(p_times) + np.sum(e_times)
            all_mog_total_time.append(mog_pick_time)

            mog_objects_picked = total_num_objects - objects_left_mog_list[scene_number]
            mog_pps = mog_objects_picked/float(mog_pick_time)
            mog_pph = mog_pps*3600
            all_mog_pph.append(mog_pph)

            mog_success = ((attempt_number - failed_attempts_mog_list[scene_number])/float(attempt_number))*100
            all_mog_success.append(mog_success)

            mog_percent_cleared = (mog_objects_picked/float(total_num_objects))*100
            all_mog_percent_cleared.append(mog_percent_cleared)

            # print ('MOG success', mog_success)
            # print ('MOG cleared', mog_percent_cleared)
            # print ('MOG pph', mog_pph)

        grasp_type = 'SOG'
        sog_e_times, sog_p_times, sog_attempt_number = robot_times(scene_number, grasp_type)

        if sog_attempt_number != 0:
            all_sog_plan_time.append(np.sum(sog_p_times))
            all_sog_e_time.append(np.sum(sog_e_times))
            sog_pick_time = np.sum(sog_p_times) + np.sum(sog_e_times)
            all_sog_total_time.append(sog_pick_time)
            all_sog_attempt_number.append(sog_attempt_number)
            all_sog_failed.append(failed_attempts_sog_list[scene_number])

            sog_objects_picked = total_num_objects - objects_left_sog_list[scene_number]
            sog_pps = sog_objects_picked/float(sog_pick_time)
            sog_pph = sog_pps*3600
            all_sog_pph.append(sog_pph)

            sog_success = ((sog_attempt_number - failed_attempts_sog_list[scene_number])/float(sog_attempt_number))*100
            all_sog_success.append(sog_success)

            sog_percent_cleared = (sog_objects_picked/float(total_num_objects))*100
            all_sog_percent_cleared.append(sog_percent_cleared)


    print ('SOG Planning time {} +- {}'.format(np.mean(all_sog_plan_time), np.std(all_sog_plan_time)))
    print ('SOG Execution time {} +- {}'.format(np.mean(all_sog_e_time), np.std(all_sog_e_time)))
    print ('SOG Total time {} +- {}'.format(np.mean(all_sog_total_time), np.std(all_sog_total_time)))
    print ('SOG average success {} +- {}'.format(np.mean(all_sog_success), np.std(all_sog_success)))
    print ('SOG average cleared {} +- {}'.format(np.mean(all_sog_percent_cleared), np.std(all_sog_percent_cleared)))
    print ('SOG average pph {} +- {}'.format(np.mean(all_sog_pph), np.std(all_sog_pph)))

    print ('\n')
    print ('MOG Planning time {} +- {}'.format(np.mean(all_mog_plan_time), np.std(all_mog_plan_time)))
    print ('MOG Execution time {} +- {}'.format(np.mean(all_mog_e_time), np.std(all_mog_e_time)))
    print ('MOG Total time {} +- {}'.format(np.mean(all_mog_total_time), np.std(all_mog_total_time)))
    print ('MOG average success {} +- {}'.format(np.mean(all_mog_success), np.std(all_mog_success)))
    print ('MOG average cleared {} +- {}'.format(np.mean(all_mog_percent_cleared), np.std(all_mog_percent_cleared)))
    print ('MOG average pph {} +- {}'.format(np.mean(all_mog_pph), np.std(all_mog_pph)))

if __name__ == '__main__':
    main()
