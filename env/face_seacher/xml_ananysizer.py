__author__ = 'Air'
import  xml.dom.minidom


class xmlPhrazer():
    def __init__(self):
        return

    def phraze_xml(self, _path):

        dom = xml.dom.minidom.parse(_path)
        root = dom.documentElement

        object_traget = root.getElementsByTagName('object')
        if not object_traget:
            raise "no object in the xml"
            return None

        for oo in object_traget:
            name = oo.getElementsByTagName("name")[0].firstChild.data
            if name == 'person':
                bb = oo.getElementsByTagName("bndbox")[0]
                x_min = int(bb.getElementsByTagName("xmin")[0].firstChild.data)
                y_min = int(bb.getElementsByTagName("ymin")[0].firstChild.data)
                x_max = int(bb.getElementsByTagName("xmax")[0].firstChild.data)
                y_max = int(bb.getElementsByTagName("ymax")[0].firstChild.data)
                return (x_min, y_min, x_max, y_max)
        #raise "no person in the xml"
        return None