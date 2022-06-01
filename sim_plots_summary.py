import numpy as np
import matplotlib.pyplot as plt

font_size = 13
line_width = 5
line_style = '-.'
plt.rcParams.update({'font.size': font_size})
plt.rcParams.update({'figure.figsize': [6.4, 4.8]})
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.tick_params(right=False)
plt.tick_params(top=False)

all_objs = [2,3,4,5,6,7]

def load_data():
    all_desired_times = np.zeros((len(all_objs),4))
    all_desired_time_errs = np.zeros((len(all_objs),4))
    all_desired_samples = np.zeros((len(all_objs),4))
    all_desired_sample_errs = np.zeros((len(all_objs),4))
    all_desired_conds = np.zeros((len(all_objs),4))
    all_desired_conds_errs = np.zeros((len(all_objs),4))

    for objs in range(len(all_objs)):
        all_desired_times[objs] = np.load('./sim_summary_data/desired_times_both_{}.npy'.format(all_objs[objs]))
        all_desired_time_errs[objs] = np.load('./sim_summary_data/desired_time_errs_both_{}.npy'.format(all_objs[objs]))
        all_desired_conds[objs] = np.load('./sim_summary_data/desired_conds_{}.npy'.format(all_objs[objs]))
        all_desired_conds_errs[objs] = np.load('./sim_summary_data/desired_conds_errs_{}.npy'.format(all_objs[objs]))
        all_desired_samples[objs] = np.load('./sim_summary_data/desired_samples_grasp_{}.npy'.format(all_objs[objs]))
        all_desired_sample_errs[objs] = np.load('./sim_summary_data/desired_sample_errs_grasp_{}.npy'.format(all_objs[objs]))

    comb_des_time = np.mean(all_desired_times, axis=0)
    comb_des_time_errs = np.mean(all_desired_time_errs, axis=0)

    comb_des_conds   = np.mean(all_desired_conds, axis=0)
    comb_des_conds_errs = np.mean(all_desired_conds_errs, axis=0)

    comb_des_samples   = np.mean(all_desired_samples, axis=0)
    comb_des_sample_errs = np.mean(all_desired_sample_errs, axis=0)
    return comb_des_time, comb_des_time_errs, comb_des_conds, comb_des_conds_errs, comb_des_samples, comb_des_sample_errs, all_desired_conds, all_desired_conds_errs

comb_des_time, comb_des_time_err, comb_des_conds, comb_des_conds_errs, sam, sam_err, all_desired_conds, all_desired_conds_errs = load_data()

x_axis = np.array([1,2,3,4])
plt.bar(x_axis+0.25, comb_des_time, yerr=comb_des_time_err, width=.5, color='#1f77b4', ecolor='k')
plt.ylabel('Grasp planning time (s)')
plt.xticks(x_axis+0.5, ['Rand-Phys', 'Rank-Phys', 'Rand-Fil-Phys', 'GP(Ours)'] )
plt.savefig('./sim_summary_plots/grasp_planning_time.png')
plt.cla()

p_width = 0.1
des_num = 2
plt.bar(x_axis[0:des_num]-3*p_width+p_width/2., all_desired_conds[0][0:des_num], yerr=all_desired_conds_errs[0][0:des_num], width=p_width, color= "maroon", ecolor='k')
plt.bar(x_axis[0:des_num]-2*p_width+p_width/2., all_desired_conds[1][0:des_num], yerr=all_desired_conds_errs[1][0:des_num], width=p_width, color="yellow", ecolor='k')
plt.bar(x_axis[0:des_num]-p_width+p_width/2., all_desired_conds[2][0:des_num], yerr=all_desired_conds_errs[2][0:des_num], width=p_width, color="cyan", ecolor='k')
plt.bar(x_axis[0:des_num]+p_width-p_width/2., all_desired_conds[3][0:des_num], yerr=all_desired_conds_errs[3][0:des_num], width=p_width, color="indigo", ecolor='k')
plt.bar(x_axis[0:des_num]+2*p_width-p_width/2., all_desired_conds[4][0:des_num], yerr=all_desired_conds_errs[4][0:des_num], width=p_width, color="orange", ecolor='k')
plt.bar(x_axis[0:des_num]+3*p_width-p_width/2., all_desired_conds[5][0:des_num], yerr=all_desired_conds_errs[5][0:des_num], width=p_width, color="seagreen", ecolor='k')
plt.ylabel('Filtered negatives (%)')
plt.xticks([1, 2], ['Int. Area', 'Diameter'] )
plt.ylim([0, 100])
plt.legend(['2 objs.', '3 objs.', '4 objs.', '5 objs.', '6 objs.', '7 objs.'])
plt.savefig('./sim_summary_plots/grasp_failure_conds.png')
plt.cla()

plt.bar(x_axis+0.25, sam, yerr=sam_err, width=.5, color='#1f77b4', ecolor='k')
plt.ylabel('Tested grasp candidates')
plt.xticks(x_axis+0.5, ['Rand-Phys', 'Rank-Phys', 'Rand-Fil-Phys', 'GP(Ours)'] )
plt.savefig('./sim_summary_plots/grasp_planning_sample.png')
plt.cla()
