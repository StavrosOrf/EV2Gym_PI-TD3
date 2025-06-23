import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from res_utils import dataset_info, parse_string_to_list

fig, axs = plt.subplots(3, 1, figsize=(14, 18))
all_data = pd.read_csv("./results_analysis/results.csv")
# print(all_data.keys())
# print unique values of the group column
print(all_data["group"].unique())
print(all_data["algorithm"].unique())
print(all_data["K"].unique())
print(all_data["seed"].unique())

# plot the data
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'

# print(axs)
for index in range(1):

    for alg_index, group in enumerate(['v2g', 'v2g_profitmax','grid']):

        # print(
        #     f'Index: {index}, Alg_index: {alg_index}, K: {K}, Algorithm: {algorithm}')

        # data = all_data[(all_data["K"] == K)]
        data = all_data[(all_data["group"] == group)]

        # data = data[data["dataset"].isin(datasets_list)]
        # dataset_info(data)
        # For every row in the data create a new dataframe with epoch as the index and the reward as the value, keep also, the seed, algorithm and dataset

        new_df = pd.DataFrame()
        for i, row in data.iterrows():
            rewards = parse_string_to_list(row["mean_rewards"])

            for j in range(250):
                # if there is no value for the epoch, use the last value

                reward = rewards[j] if j < len(rewards) else rewards[-1]
                entry = {
                    "epoch": j,
                    "reward": reward,
                    "seed": row["seed"],
                    "algorithm": row["algorithm"]+"_" + str(row["K"]),
                    "group": row["group"]
                }
                new_df = pd.concat([new_df, pd.DataFrame([entry])])

        print(f'New df shape: {new_df.shape}')

        new_df["algorithm"] = new_df["algorithm"].replace("TD3_1", "TD3")
        new_df["algorithm"] = new_df["algorithm"].replace("SAC_1", "SAC")



        print(f' Data ready to plot')

        hue_order = ['TD3',
                     'SAC',
                     'MB-TD3_1',
                     'MB-TD3_2',
                     'MB-TD3_5',
                     'MB-TD3_10',
                     'MB-TD3_20',
                     'MB-TD3_25',
                     'MB-TD3_40',
                     'MB-TD3_50',
                     'MB-TD3_60',
                     'MB-TD3_70',
                     'MB-TD3_80',
                     ]

        color_pallete = [
            'red',
            'orange',
        ]
        color_pallete += sns.color_palette("viridis", len(hue_order)-2)

        sns.lineplot(data=new_df,
                     x="epoch",
                     y="reward",
                     hue="algorithm",
                     hue_order=hue_order,
                     ax=axs[alg_index],
                     palette=color_pallete,
                     )

        axs[alg_index].set_xlabel("Epoch", fontsize=16)
        axs[alg_index].set_ylabel("Reward [-]", fontsize=16)
        axs[alg_index].set_title(f"Objective: {group}", fontsize=17)

        if alg_index == 2:

            plt.legend(loc='upper center',
                       bbox_to_anchor=(0.5, -0.22),
                       ncol=5,
                       fontsize=15,
                       title="Algorithm",
                       title_fontsize=14)
        else:
            axs[alg_index].get_legend().remove()

        if alg_index == 2:
            # set title
            axs[alg_index].set_title(
                f"c) Min. Grid Voltage Violations + Max. V2G Profits + Min. User Sat.", fontsize=15)
        elif alg_index == 0:
            # set title
            axs[alg_index].set_title(
                f"a) Max. V2G Profits", fontsize=15)
        elif alg_index == 1:
            # set title
            axs[alg_index].set_title(
                f"b) Max. V2G Profits + Min. User Sat.", fontsize=15)

        axs[alg_index].set_xlim(0, 250)

        # axs[alg_index][index].axhline(
        #     y=-2405, color='r', linestyle='--')

        # # show grid lines
        axs[alg_index].grid(True)

        # if index != 0:
        #     axs[alg_index][index].set_ylabel("")
        # else:
        #     axs[alg_index][index].set_ylabel(f"{algo_name}\nReward [-]",
        #                                      fontsize=14)

        # if alg_index != 2:
        #     axs[alg_index][index].set_xlabel("")
        # else:
        #     axs[alg_index][index].set_xlabel("Epoch", fontsize=14)

        # # Set xticks and yticks font size
        axs[alg_index].tick_params(axis='x', labelsize=15)
        # # show ticks on the y-axis onl the first time
        #         
        axs[alg_index].tick_params(axis='y', labelsize=15)
        axs[alg_index].ticklabel_format(
            style='sci', axis='y', scilimits=(0, 0))

        # if alg_index == 1 and index == 1:
        #     plt.tight_layout()
        #     plt.show()
        #     exit()

# Adjust layout
fig.tight_layout()
# plt.subplots_adjust(
#     left=0.07,    # Space from the left of the figure
#     bottom=0.138,   # Space from the bottom of the figure
#     right=0.986,   # Space from the right of the figure
#     top=0.964,     # Space from the top of the figure
#     wspace=0.15,    # Width space between subplots
#     hspace=0.214     # Height space between subplots
# )
plt.savefig(f"results_analysis/figs/performance_all.pdf",
            dpi=60)
plt.savefig(f"results_analysis/figs/performance_all.png",
            dpi=200)
plt.show()
plt.clf()
