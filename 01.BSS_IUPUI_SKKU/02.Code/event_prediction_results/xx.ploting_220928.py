from matplotlib import pyplot as plt

def create_x(t, w, n, d):
    return [t * x + w * n for x in range(d)]


def depic_error_bar(ax, x, y, up, low, colour):
    for set_ in zip(x, y, up, low):
        print(set_)
        ax.errorbar(
            set_[0],
            set_[1],
            yerr=([set_[1] - set_[3]], [set_[2] - set_[1]]),
            color=colour,
            linewidth=0.8,
            capsize=2,
        )


# data
lower = [85.52,85.02,85.38,85.89,85.11]
normal = [98.69,98.77,98.79,98.7,98.89]
upper = [80.16,78.99,80.08,82.14,79.9]

acc = [96.89,96.89,96.97,96.98,97.02]
# LSTM_acc = [85.39, 87.75, 89.86, 92.14, 95.21]

# eMoGAN_acc_upper = [99.35742025, 99.92116468, 99.95171052, 99.76017161, 99.8371746]
# eMoGAN_acc_lower = [98.00391308, 98.06750198, 96.29028948, 99.65982839, 98.7688254]
# MoGAN_acc_upper = [86.25, 89.57, 91.73, 95.07, 96.27]
# MoGAN_acc_lower = [84.58, 88.05, 90.46, 94.52, 95.82]
# LSTM_acc_upper = [86.49, 88.75, 90.72, 93.89, 95.81]
# LSTM_acc_lower = [84.29, 86.74, 89, 90.4, 94.61]

# x axis
amount = ["60", "70", "80","90","100"]

value_a_x = create_x(3, 0.8, 1, 5)
value_b_x = create_x(3, 0.8, 2, 5)
value_c_x = create_x(3, 0.8, 3, 5)
# value_c_x = create_x(3, 0.8, 3, 5)

# bar setting
ax = plt.subplot()
ax.bar(value_a_x, lower, color="#003d65", edgecolor="black", label="Lower Case")
ax.bar(value_b_x, normal, color="#0072BC", edgecolor="black", label="Normal Case")
ax.bar(value_c_x, upper, color="#069AF3", edgecolor="black", label="Upper Case")
ax.plot(value_b_x, acc,"ro-",markerfacecolor='none', markersize=10,label="Overall" )
# ax.bar(value_c_x, eMoGAN_acc, color="#8BC63E", edgecolor="black", label="iGAN-FF")

# depic_error_bar(ax, value_a_x, LSTM_acc, LSTM_acc_upper, LSTM_acc_lower, "black")
# depic_error_bar(ax, value_b_x, MoGAN_acc, MoGAN_acc_upper, MoGAN_acc_lower, "black")
# depic_error_bar(ax, value_c_x, eMoGAN_acc, eMoGAN_acc_upper, eMoGAN_acc_lower, "black")


handles, labels = ax.get_legend_handles_labels()
dict_labels_handles = dict(zip(labels, handles))
labels = ["Lower Case","Normal Case", "Upper Case","Overall"]
handles = [dict_labels_handles[l] for l in labels]
ax.legend(
    handles, labels, fontsize=9, loc="lower right", framealpha=1.0, edgecolor="black",
)
ax.set_facecolor("#E1E9EB")  # background color
ax.set_axisbelow(True)
plt.grid(linestyle="--")
plt.tick_params(direction="in")

middle_x = [(a + b+c) / 3 for (a, b,c) in zip(value_a_x, value_b_x,value_c_x)]
ax.set_xticks(middle_x)
ax.set_xticklabels(amount)
plt.xlabel("Epoch", fontsize=14)  # label of x axis
plt.ylabel("Accuracy (%)", fontsize=14)  # label of y axis
plt.xticks(fontsize=14)
plt.ylim([75, 100])  # range of y axis
plt.yticks([75,80,85,90,95,100], fontsize=14)  # values to show in y axis


for i, v in enumerate(value_a_x):
    plt.text(v,acc[i], acc[i],  # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
                fontsize=11,
                weight='bold',
                color='black',
                horizontalalignment='center',  # horizontalalignment (left, center, right)
                verticalalignment='bottom')  # verticalalignment (top, center, bottom)
# for i, v in enumerate(value_b_x):
#     plt.text(v,dtw[i], dtw[i],  # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
#                 fontsize=11,
#                 weight='bold',
#                 color='black',
#                 horizontalalignment='center',  # horizontalalignment (left, center, right)
#                 verticalalignment='bottom')  # verticalalignment (top, center, bottom)

# for i, v in enumerate(value_b_x):
#     plt.text(v,dtw[i], dtw[i],  # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
#                 fontsize=11,
#                 weight='bold',
#                 color='black',
#                 horizontalalignment='center',  # horizontalalignment (left, center, right)
#                 verticalalignment='bottom')  # verticalalignment (top, center, bottom)


root_path = "./04.event_prediction/"
plot_path = root_path+"event_prediction_results/"
plt.savefig(
    plot_path + "xx_220928.png",
    dpi=500,
    edgecolor="white",
    bbox_inches="tight",
    pad_inches=0.2,
)

