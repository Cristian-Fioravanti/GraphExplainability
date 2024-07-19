import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def load_data(file_confusion, start_size=2):
    data = []
    config_size = start_size
    while True:
        file_name = f'{file_confusion}_{config_size}.pkl'
        if not os.path.exists(file_name):
            break
        with open(file_name, 'rb') as file:
            data.append(pickle.load(file))
        config_size += 1
    return data

def plot_data(data):
    misclassified_signal_as_other_signal  = [item['misclassified_signal_as_other_signal'] for item in data]
    correctly_classified_signal = [item['correctly_classified_signal'] for item in data]
    correctly_classified_background = [item['correctly_classified_background'] for item in data]
    misclassified_signal_as_background = [item['misclassified_signal_as_background'] for item in data]
    misclassified_background_as_signal = [item['misclassified_background_as_signal'] for item in data]
    print("Number of correctly classified signal:", correctly_classified_signal)
    print("Number of correctly classified background:", correctly_classified_background)
    print("Number of misclassified signal as background:", misclassified_signal_as_background)
    print("Number of misclassified background as signal:", misclassified_background_as_signal)
    print("Number of misclassified signal as other signal:", misclassified_signal_as_other_signal)
    sizes = range(2, 2 + len(data))
    
    plt.figure(figsize=(10, 6))
    
    
    plt.plot(sizes, correctly_classified_signal, label='Correctly Classified Signal')
    plt.plot(sizes, correctly_classified_background, label='Correctly Classified Background')
    plt.plot(sizes, misclassified_signal_as_background, label='Misclassified Signal as Background')
    plt.plot(sizes, misclassified_background_as_signal, label='Misclassified Background as Signal')
    plt.plot(sizes, misclassified_signal_as_other_signal, label='Misclassified Signal as Other Signal')

    plt.xlabel('Configuration Size')
    plt.ylabel('Count')
    plt.title('Classification Results by Configuration Size')
    plt.legend()
    plt.grid(True)
    # Imposta l'asse x per mostrare solo valori interi
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))    
    plt.show()

if __name__ == "__main__":
    file_confusion = './confusion_data'  # sostituisci con il percorso corretto del tuo file
    data = load_data(file_confusion)
    if data:
        plot_data(data)
    else:
        print("Nessun file trovato.")