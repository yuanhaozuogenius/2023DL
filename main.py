# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from src import tSNE


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    sample_data, df_labels = tSNE.tSNE.load_data()
    tSNE.tSNE.tSNE_method(sample_data, df_labels)
    # tSNE.PCA.PCA_method(sample_data, df_labels)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
