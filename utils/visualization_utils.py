import matplotlib.pyplot as plt

def plot_drug_release(time_steps, predicted_release, actual_release, save_path):
    plt.figure()
    plt.plot(time_steps, predicted_release, label="Predicted")
    plt.plot(time_steps, actual_release, label="Actual")
    plt.xlabel("Time")
    plt.ylabel("Drug Release")
    plt.legend()
    plt.savefig(save_path)
    plt.close()