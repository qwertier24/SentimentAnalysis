try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import torch


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, xml_path):
        xml_tree = ET.ElementTree(file=xml_file)
