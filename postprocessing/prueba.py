# Python
#
# This file implements an example.
#
# This file is part of mdutils. https://github.com/didix21/mdutils
#
# MIT License: (C) 2018 Dídac Coll


from mdutils.mdutils import MdUtils
from mdutils import Html
from mdutils.fileutils import MarkDownFile

mdFile = MdUtils(file_name="table")

list_of_strings = ["Items", "Descriptions", "Data"]
for x in range(5):
    list_of_strings.extend(
        ["Item " + str(x), "Description Item " + str(x), str(x)]
    )
mdFile.new_line()
mdFile.new_table(columns=3, rows=6, text=list_of_strings, text_align="center")

# Create a table of contents
md_file = mdFile.create_md_file()

Mem = MarkDownFile("mery.md")

Mem.read_file("memory.md")
