from openfoam_results_to_array import getTemp as foam_2_array
from openfoam_results_to_array import plot as temp_plot
import matplotlib.pyplot as plt
home_0 = '/home/nathan/Desktop/Group_presentations/Update_12_12/refine_0/'
home_1 = '/home/nathan/Desktop/Group_presentations/Update_12_12/refine_1/'
home_2 = '/home/nathan/Desktop/Group_presentations/Update_12_12/refine_2/'
home_base = '/home/nathan/Desktop/Group_presentations/Update_12_12/base_case/SIM_0/'
home_fancy = '/home/nathan/Desktop/Group_presentations/Update_12_12/base_case/SIM_fancy_1/'
home = '/home/nathan/Desktop/Group_presentations/'

# Base_case
avg_0_0, temp_0_0 = foam_2_array(home_fancy)
temp_plot(avg_0_0, temp_0_0, "fancy")

# Refine mesh level 0
# avg_0_0, temp_0_0 = foam_2_array(home_0 + 'SIM_0/')
# temp_plot(avg_0_0, temp_0_0, "0_0")

# avg_1_0, temp_1_0 = foam_2_array(home_0 + 'SIM_1/')
# temp_plot(avg_1_0, temp_1_0, "1_0")

# avg_1_0, temp_1_0 = foam_2_array(home_0 + 'SIM_2/')
# temp_plot(avg_1_0, temp_1_0, "2_0")

# avg_1_0, temp_1_0 = foam_2_array(home_0 + 'SIM_3/')
# temp_plot(avg_1_0, temp_1_0, "3_0")

# # Refine mesh level 1
# avg_0_0, temp_0_0 = foam_2_array(home_1 + 'SIM_0/')
# temp_plot(avg_0_0, temp_0_0, "0_1")

# avg_1_0, temp_1_0 = foam_2_array(home_1 + 'SIM_1/')
# temp_plot(avg_1_0, temp_1_0, "1_1")

# avg_1_0, temp_1_0 = foam_2_array(home_1 + 'SIM_2/')
# temp_plot(avg_1_0, temp_1_0, "2_1")

# avg_1_0, temp_1_0 = foam_2_array(home_1 + 'SIM_3/')
# temp_plot(avg_1_0, temp_1_0, "3_1")

# Refine mesh level 2
# avg_0_0, temp_0_0 = foam_2_array(home_2 + 'SIM_0/')
# temp_plot(avg_0_0, temp_0_0, "0_2")

# avg_1_0, temp_1_0 = foam_2_array(home_2 + 'SIM_1/')
# temp_plot(avg_1_0, temp_1_0, "1_2")

# avg_1_0, temp_1_0 = foam_2_array(home_2 + 'SIM_2/')
# temp_plot(avg_1_0, temp_1_0, "2_2")

# avg_1_0, temp_1_0 = foam_2_array(home_2 + 'SIM_3/')
# temp_plot(avg_1_0, temp_1_0, "3_2")

# Sim 0
avg_0_0, temp_0_0 = foam_2_array(home_0 + 'SIM_0/')
# avg_0_1, temp_0_1 = foam_2_array(home_1 + 'SIM_0/')
# avg_0_2, temp_0_2 = foam_2_array(home_2 + 'SIM_0/')

# error_0_0 = 100 * abs((temp_0_0 - temp_0_2) / temp_0_2)
# error_0_1 = 100 * abs((temp_0_1 - temp_0_2) / temp_0_2)

# fig = plt.figure(figsize=(6, 3.2))
# ax = fig.add_subplot(111)
# title = "4884 vs 78144 cells"
# ax.set_title(title, loc='center', wrap=True)
# plt.imshow(error_0_0)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# plt.tight_layout()
# ax.set_aspect('equal')
# cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
# cax.get_xaxis().set_visible(False)
# cax.get_yaxis().set_visible(False)
# cax.patch.set_alpha(0)
# cax.set_frame_on(False)
# cb = plt.colorbar(orientation='vertical')
# cb.ax.set_title("% Error ")
# plt.savefig(home + "0_0_error.png", dpi=600)

# fig = plt.figure(figsize=(6, 3.2))
# ax = fig.add_subplot(111)
# title = "19536 vs 78144 cells"
# ax.set_title(title, loc='center', wrap=True)
# plt.imshow(error_0_1)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# plt.tight_layout()
# ax.set_aspect('equal')
# cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
# cax.get_xaxis().set_visible(False)
# cax.get_yaxis().set_visible(False)
# cax.patch.set_alpha(0)
# cax.set_frame_on(False)
# cb = plt.colorbar(orientation='vertical')
# cb.ax.set_title("% Error ")
# plt.savefig(home + "0_1_error.png", dpi=600)

# # Sim 1
# avg_0_0, temp_0_0 = foam_2_array(home_0 + 'SIM_1/')
# avg_0_1, temp_0_1 = foam_2_array(home_1 + 'SIM_1/')
# avg_0_2, temp_0_2 = foam_2_array(home_2 + 'SIM_1/')

# error_0_0 = 100 * abs((temp_0_0 - temp_0_2) / temp_0_2)
# error_0_1 = 100 * abs((temp_0_1 - temp_0_2) / temp_0_2)

# fig = plt.figure(figsize=(6, 3.2))
# ax = fig.add_subplot(111)
# title = "4884 vs 78144 cells"
# ax.set_title(title, loc='center', wrap=True)
# plt.imshow(error_0_0)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# plt.tight_layout()
# ax.set_aspect('equal')
# cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
# cax.get_xaxis().set_visible(False)
# cax.get_yaxis().set_visible(False)
# cax.patch.set_alpha(0)
# cax.set_frame_on(False)
# cb = plt.colorbar(orientation='vertical')
# cb.ax.set_title("% Error ")
# plt.savefig(home + "1_0_error.png", dpi=600)

# fig = plt.figure(figsize=(6, 3.2))
# ax = fig.add_subplot(111)
# title = "19536 vs 78144 cells"
# ax.set_title(title, loc='center', wrap=True)
# plt.imshow(error_0_1)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# plt.tight_layout()
# ax.set_aspect('equal')
# cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
# cax.get_xaxis().set_visible(False)
# cax.get_yaxis().set_visible(False)
# cax.patch.set_alpha(0)
# cax.set_frame_on(False)
# cb = plt.colorbar(orientation='vertical')
# cb.ax.set_title("% Error ")
# plt.savefig(home + "1_1_error.png", dpi=600)

# # Sim 2
# avg_0_0, temp_0_0 = foam_2_array(home_0 + 'SIM_2/')
# avg_0_1, temp_0_1 = foam_2_array(home_1 + 'SIM_2/')
# avg_0_2, temp_0_2 = foam_2_array(home_2 + 'SIM_2/')

# error_0_0 = 100 * abs((temp_0_0 - temp_0_2) / temp_0_2)
# error_0_1 = 100 * abs((temp_0_1 - temp_0_2) / temp_0_2)

# fig = plt.figure(figsize=(6, 3.2))
# ax = fig.add_subplot(111)
# title = "4884 vs 78144 cells"
# ax.set_title(title, loc='center', wrap=True)
# plt.imshow(error_0_0)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# plt.tight_layout()
# ax.set_aspect('equal')
# cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
# cax.get_xaxis().set_visible(False)
# cax.get_yaxis().set_visible(False)
# cax.patch.set_alpha(0)
# cax.set_frame_on(False)
# cb = plt.colorbar(orientation='vertical')
# cb.ax.set_title("% Error ")
# plt.savefig(home + "2_0_error.png", dpi=600)

# fig = plt.figure(figsize=(6, 3.2))
# ax = fig.add_subplot(111)
# title = "19536 vs 78144 cells"
# ax.set_title(title, loc='center', wrap=True)
# plt.imshow(error_0_1)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# plt.tight_layout()
# ax.set_aspect('equal')
# cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
# cax.get_xaxis().set_visible(False)
# cax.get_yaxis().set_visible(False)
# cax.patch.set_alpha(0)
# cax.set_frame_on(False)
# cb = plt.colorbar(orientation='vertical')
# cb.ax.set_title("% Error ")
# plt.savefig(home + "2_1_error.png", dpi=600)

# # Sim 3
# avg_0_0, temp_0_0 = foam_2_array(home_0 + 'SIM_3/')
# avg_0_1, temp_0_1 = foam_2_array(home_1 + 'SIM_3/')
# avg_0_2, temp_0_2 = foam_2_array(home_2 + 'SIM_3/')

# error_0_0 = 100 * abs((temp_0_0 - temp_0_2) / temp_0_2)
# error_0_1 = 100 * abs((temp_0_1 - temp_0_2) / temp_0_2)

# fig = plt.figure(figsize=(6, 3.2))
# ax = fig.add_subplot(111)
# title = "4884 vs 78144 cells"
# ax.set_title(title, loc='center', wrap=True)
# plt.imshow(error_0_0)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# plt.tight_layout()
# ax.set_aspect('equal')
# cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
# cax.get_xaxis().set_visible(False)
# cax.get_yaxis().set_visible(False)
# cax.patch.set_alpha(0)
# cax.set_frame_on(False)
# cb = plt.colorbar(orientation='vertical')
# cb.ax.set_title("% Error ")
# plt.savefig(home + "3_0_error.png", dpi=600)

# fig = plt.figure(figsize=(6, 3.2))
# ax = fig.add_subplot(111)
# title = "19536 vs 78144 cells"
# ax.set_title(title, loc='center', wrap=True)
# plt.imshow(error_0_1)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# plt.tight_layout()
# ax.set_aspect('equal')
# cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
# cax.get_xaxis().set_visible(False)
# cax.get_yaxis().set_visible(False)
# cax.patch.set_alpha(0)
# cax.set_frame_on(False)
# cb = plt.colorbar(orientation='vertical')
# cb.ax.set_title("% Error ")
# plt.savefig(home + "3_1_error.png", dpi=600)