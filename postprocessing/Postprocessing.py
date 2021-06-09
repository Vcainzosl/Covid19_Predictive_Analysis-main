import sys

sys.path.append(".")
from mdutils import MdUtils
from mdutils.fileutils.fileutils import MarkDownFile
from mdutils import Html
from utils.Saving import Saving
import csv
import os


class Postprocessing(Saving, MdUtils):
    """Postprocessing class

    :param Saving: inheritance of Saving class
    :type Saving: class
    :param MdUtils: inheritance of MdUtils class
    :type MdUtils: class
    """

    def __init__(
        self,
        file_name: str,
        title="",
        author="",
        base_dir=".",
        csv=True,
        img=True,
        pickle=False,
        autogenerate="True",
    ):
        """Constructor method of postprocessing class

        :param file_name: name of the file
        :type file_name: str
        :param title: title of the document, defaults to ""
        :type title: str, optional
        :param author: name of the author, defaults to ""
        :type author: str, optional
        :param base_dir: base directory path, defaults to "."
        :type base_dir: str, optional
        :param csv: if True csv folder will be created, defaults to True
        :type csv: bool, optional
        :param img: if True img folder will be created, defaults to True
        :type img: bool, optional
        :param pickle: if True pickle folder will be created, defaults to False
        :type pickle: bool, optional
        :param autogenerate: if True report will be created automatically reading data from directories, defaults to "True"
        :type autogenerate: str, optional
        """
        Saving.__init__(
            self,
            base_dir=base_dir,
            csv=csv,
            img=img,
            pickle=pickle,
        )
        MdUtils.__init__(self, file_name, title=title, author=author)

        if autogenerate:
            self.autogenerate_report()

    def sort_img_dir(self, elem: str):
        """Auxiliar function to define sorting criterion

        :param elem: element of an iterable
        :type elem: str
        :return: splited element, usually index number
        :rtype: int
        """
        return int(elem.split(".")[0])

    def autogenerate_report(self):
        """This function manage data from directories and creates a report automatically"""
        # List of csv files in the directory
        files = os.listdir(self.csv_dir)
        # Sorting files according to index using auxiliary fuction
        files.sort(key=self.sort_img_dir)
        for file in files:
            self.new_line('<div style="page-break-after: always;"></div>')
            self.new_line("  ")
            self.add_csv(self.csv_dir + os.path.sep + file)
        files = os.listdir(self.img_dir)
        files.sort(key=self.sort_img_dir)
        for file in files:
            if "(" not in file:
                self.new_line('<div style="page-break-after: always;"></div>')
                self.new_line("  ")
            self.add_img(self.img_dir + os.path.sep + file)

        self.create_report()

    def add_csv(self, filepath: str, description=""):
        """Converts csv files to md tables and write them on an md file

        :param filepath: path of the csv file to be written
        :type filepath: str
        :param description: some explaining text to the file, defaults to ""
        :type description: str, optional
        """

        # Reading filename from path with conditional style for aesthetics md
        filename = filepath.split("\\")[-1].split(".")[1].split("-")
        if len(filename) > 1:
            filename[1] = "*(" + filename[1] + ")*"
        self.new_line("  ")
        self.new_header(1, (" ").join(filename))
        self.new_paragraph(description)
        # Reading csv file and coverting to md
        file = open(filepath, "r")
        reader = csv.reader(file)
        # Lists for rows and elements
        l = []
        m = []
        e = []
        for row in reader:
            # Size control to md table aesthetics
            if len(row) > 0 and len(row) <= 7:
                l.append(row)
            if len(row) > 7:
                for i in range(int(len(row) / 7) + 1):
                    if len(m) < int(len(row) / 7) + 1:
                        m.append([])
                    m[i].append(row[i * 7 : i * 7 + 7])
        file.close
        if len(m) > 0:
            cont = 0
            for i in m:
                text = [e.extend(j) for j in i]
                # Create md table
                self.new_line()
                self.new_table(
                    columns=len(i[0]), rows=len(i), text=e, text_align="center"
                )
                cont += 1
                if cont < len(m):
                    self.write("<br>")
                e = []
        else:
            text = [e.extend(i) for i in l]
            # Create md table
            self.new_line()
            self.new_table(
                columns=len(l[0]), rows=len(l), text=e, text_align="center"
            )
            self.new_line()

    def add_img(self, filepath: str, description=""):
        """Insert images to md file

        :param filepath: path of the image file
        :type filepath: str
        :param description: Some explaining text of the image file, defaults to ""
        :type description: str, optional
        """
        # Reading filename from path with conditional style for aesthetics md
        filename = filepath.split("\\")[-1].split(".")[1].split("-")
        if len(filename) > 1:
            filename[1] = "*(" + filename[1] + ")*"
        filename = (" ").join(filename)
        path = filepath
        # Control headers if it is an image of multiples
        if len(filepath.split("(")) == 1:
            self.new_header(1, filename)
            self.new_paragraph(description)
            self.new_line(Html.image(path=path))
        else:
            self.new_line(Html.image(path=path))

    def create_report(self, foldername=""):
        """Call this function to create md file with all data added previously

        :param foldername: specific folder to save md report if exists, defaults to ""
        :type foldername: str, optional
        """
        self.new_table_of_contents(table_title="Contents", depth=2)
        self.report = MarkDownFile(self.file_name)
        self.report.rewrite_all_file(
            data=self.title
            + self.table_of_contents
            + self.file_data_text
            + self.reference.get_references_as_markdown()
        )


if __name__ == "__main__":

    c = Postprocessing("report", "Report", "Víctor Caínzos López")
    # c.add_csv("Preprocessed_DataSets\Data_Slided.csv")
    # c.add_csv("Preprocessed_DataSets\Original_Data.csv")

    # c.create_report()
