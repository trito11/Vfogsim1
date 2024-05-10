import matplotlib.pyplot as plt

def plot_QueueingDelay(DataMatrix):
    plt.figure(figsize=(10, 6))
    plt.title('Queueing Delay per User')
    plt.xlabel('User')
    plt.ylabel('Total delay')
    plt.bar(range(len(DataMatrix)), sum(DataMatrix))
    plt.ylim(0, None)  # Set y-axis limit to start from 0
    plt.grid(True)
    plt.show()
