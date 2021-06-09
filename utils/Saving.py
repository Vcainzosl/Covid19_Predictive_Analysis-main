import os
import pickle


class Saving:
    """Class with util fuctions to save files"""

    def __init__(self, base_dir=".", csv=True, img=True, pickle=False):
        """Contructor method of Saving class

        :param base_dir: base directory, defaults to "."
        :type base_dir: str, optional
        :param csv: create csv folder, defaults to True
        :type csv: bool, optional
        :param img: create img folder, defaults to True
        :type img: bool, optional
        :param pickle: create pickle folder, defaults to False
        :type pickle: bool, optional
        """
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
        """Specific method to save img files

        :param img: the img file to save
        :type img: img
        :param filename: name of the file
        :type filename: str
        """
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
        """Method used to save pickle files

        :param data: pickle object
        :type data: object
        :param filename: name of the file
        :type filename: str
        """
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
        """Util static method to sort files in a directory

        :param filename: name of the file
        :type filename: str
        :param folderpath: path of the folder directory
        :type folderpath: str
        :return: number
        :rtype: str
        """
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
