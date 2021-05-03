import os
import pickle


class Saving:
    def __init__(self, base_dir=".", csv=True, img=True, pickle=False):
        self.base_dir = base_dir

        if csv:
            self.csv_dir = self.base_dir + os.path.sep + "csv"
            os.makedirs(self.csv_dir, exist_ok=True)
        if img:
            self.img_dir = self.base_dir + os.path.sep + "img"
            os.makedirs(self.img_dir, exist_ok=True)
        if pickle:
            self.pickle_dir = self.base_dir + os.path.sep + "pickle"
            os.makedirs(self.pickle_dir, exist_ok=True)

    def save_img(self, img, filename):
        img.savefig(
            self.img_dir
            + os.path.sep
            + Saving.sort_file(filename, self.img_dir)
            + filename
            + ".jpg",
        )

    def save_csv(self, data: str, filename: str):
        """Predefines built-in function open() to save .csv files

        :param data: the .csv file to be saved
        :type data: str
        :param filename: the name of the saved file
        :type filename: str

        """

        file = open(
            self.csv_dir
            + os.path.sep
            + Saving.sort_file(filename, self.csv_dir)
            + filename
            + ".csv",
            "w",
        )
        file.write(data)
        file.close

    def save_pickle(self, data: object, filename: str):
        file = open(
            self.pickle_dir
            + os.path.sep
            + Saving.sort_file(filename, self.pickle_dir)
            + filename
            + ".pickle",
            "wb",
            encoding=None,
        )
        pickle.dump(data, file)
        file.close

    @staticmethod
    def sort_file(filename: str, folderpath: str):
        if len(os.listdir(folderpath)) == 0:
            num = "1."
        else:
            for file in os.listdir(folderpath):
                if filename.split(".")[0] != file.split(".")[1]:
                    num = str(len(os.listdir(folderpath)) + 1) + "."
                else:
                    num = file.split(".")[0] + "."
                    break
        return num
