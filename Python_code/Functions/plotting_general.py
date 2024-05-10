import matplotlib.pyplot as plt

def plotting_general(value, x_text, y_text):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(value)), value, align='center')
    plt.xlabel(x_text)
    plt.ylabel(y_text)
    plt.grid(True)
    plt.show()
