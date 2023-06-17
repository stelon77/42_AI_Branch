class CsvReader():
    def __init__(self, filename=None, sep=',', header=False, skip_top=0,
                 skip_bottom=0):
        if (type(filename) is not str) or \
           (type(sep) is not str) or \
           (type(header) is not bool) or \
           (type(skip_top) is not int) or \
           (type(skip_bottom) is not int) or \
           (skip_top < 0) or \
           (skip_bottom < 0):
            print("bad arguments provided")
            exit(1)
        self.filename = filename
        self.sep = sep
        self.header = header
        self.skip_top = skip_top
        self.skip_bottom = skip_bottom
        self.fullData = []
        self.csvFile = None

    def __enter__(self):
        try:
            self.csvFile = open(self.filename, 'r')
        except(IOError):
            return None
        for index, line in enumerate(self.csvFile):
            occurenceList = line.split(self.sep)
            occurenceList = list(map(str.strip, occurenceList))
            if self.header and index == 0:
                pass
            else:
                for occurence in occurenceList:
                    if len(occurence) == 0:
                        return None
            self.fullData.append(occurenceList)
        size = len(self.fullData[0])
        if all((len(row) == size) for row in self.fullData):
            return self
        else:
            return None

    def __exit__(self, typere, value, traceback):
        if self.csvFile is not None:
            self.csvFile.close()

    def getdata(self):
        """ Retrieves the data/records from skip_top to skip bottom.
        Returns:
        nested list (list(list, list, ...)) representing the data.
        """
        size = len(self.fullData)
        if self.skip_top >= size or self.skip_bottom >= size:
            return []
        if self.header:
            self.skip_top += 1
            if self.skip_top >= size or self.skip_bottom >= (size - 1):
                return []
        return self.fullData[self.skip_top: size - self.skip_bottom]

    def getheader(self):
        """ Retrieves the header from csv file.
        Returns:
        list: representing the data (when self.header is True).
        None: (when self.header is False).
        """
        if self.header:
            return self.fullData[0]
        return None


# if __name__ == "__main__":
#     with CsvReader("bad.csv", header=True, skip_top=2, skip_bottom=2) as f:
#         print(f.getheader())
#         print()
#         print(f.getdata())
# if __name__ == "__main__":
#     with CsvReader('good.csv') as file:
#         data = file.getdata()
#         header = file.getheader()
#         print(data)
#         print("")
#         print(header)
if __name__ == "__main__":
    with CsvReader('bad.csv') as file:
        if file is None:
            print("File is corrupted")
