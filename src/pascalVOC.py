def appendPascalVOC(head, tail):
    output = head
    if "\n" in output:
        lines = tail.split("\n")
        for line in lines:
            output += "\n\t" + line
    return output


class BoundingBox:

    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def get(self, axis, extrema):
        if str(axis) == 'x':
            if str(extrema) == 'min':
                return self.x_min
            else:
                return self.x_max
        elif str(axis) == 'y':
            if str(extrema) == 'min':
                return self.y_min
            else:
                return self.y_max

    # get the bounding box in the pascal voc format
    def toPascalVOCFormat(self):
        output = "<bndbox>"
        output += "\n\t<xmin>" + str(self.x_min) + "</xmin>"
        output += "\n\t<ymin>" + str(self.y_min) + "</ymin>"
        output += "\n\t<xmax>" + str(self.x_max) + "</xmax>"
        output += "\n\t<ymax>" + str(self.y_max) + "</ymax>"
        output += "\n</bndbox>"
        return output

    # output for YOLOv4
    def toTxt(self):
        output = str(self.x_min) + ',' + str(self.y_min) + ',' + str(self.x_max) + ',' + str(self.y_max)
        return output


class Annotated_Object:

    def __init__(self, name, pose, truncated, difficult, bndBox):
        self.name = name
        self.pose = pose
        self.truncated = truncated
        self.difficult = difficult
        self.bndBox = bndBox

    def toPascalVOCFormat(self):
        output = "<object>"
        output += "\n\t<name>" + 'Bee' + "</name>"   #str(self.name)
        output += "\n\t<pose>" + str(self.pose) + "</pose>"
        output += "\n\t<truncated>" + str(self.truncated) + "</truncated>"
        output = appendPascalVOC(output, self.bndBox.toPascalVOCFormat())
        output += "\n</object>"
        return output


class Annotation:

    def __init__(self, folder, filename, path, source, size, segmented, objects):
        self.folder = folder
        self.filename = filename
        self.path = path
        self.source = source
        self.size = size
        self.segmented = segmented
        self.objects = objects

    def toPascalVOCFormat(self):
        output = "<annotation>"
        output += "\n\t<folder>" + self.folder + "</folder>"
        output += "\n\t<filename>" + self.filename + "</filename>"
        output += "\n\t<path>" + self.path + "</path>"
        output += "\n\t<source>"
        output += "\n\t\t<database>" + self.source + "</database>"
        output += "\n\t</source>"
        output += "\n\t<segmented>" + str(self.segmented) + "</segmented>"
        output += "\n\t<size>"
        output += "\n\t\t<width>" + str(self.size[0]) + "</width>"
        output += "\n\t\t<height>" + str(self.size[1]) + "</height>"
        output += "\n\t\t<depth>" + str(self.size[2]) + "</depth>"
        output += "\n\t</size>"
        if self.objects is not None:
            for object in self.objects:
                output = appendPascalVOC(output, object.toPascalVOCFormat())
        output += "\n</annotation>"
        return output
