from matplotlib import pyplot as plt

def visLearningCurve(loss_log, loss_log_regret, method='spo',n_aircraft = 30):
    # create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))

    # draw plot for training loss
    ax1.plot(loss_log, color="c", lw=2)
    ax1.tick_params(axis="both", which="major", labelsize=12)
    ax1.set_xlabel("Iters", fontsize=16)
    ax1.set_ylabel("Loss", fontsize=16)
    ax1.set_title("Learning Curve on Training Set", fontsize=16)

    # draw plot for regret on test
    ax2.plot(loss_log_regret, color="royalblue", ls="--", alpha=0.7, lw=2)
    
    # Dynamically set x-ticks based on the number of epochs
    num_epochs = len(loss_log_regret)
    tick_spacing = max(1, num_epochs // 10)  # Ensure at least 1 tick, but aim for about 10 ticks
    ax2.set_xticks(range(0, num_epochs, tick_spacing))
    
    ax2.tick_params(axis="both", which="major", labelsize=12)
    ax2.set_ylim(0, 0.5)
    ax2.set_xlabel("Epochs", fontsize=16)
    ax2.set_ylabel("Regret", fontsize=16)
    ax2.set_title("Learning Curve on Test Set", fontsize=16)
    plt.savefig(f'{method}_{n_aircraft}.png', dpi=300)
    plt.show()