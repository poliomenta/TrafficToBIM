from lxml import etree


class NodesXML:
    def __init__(self, xml_file=None):
        self.xml_file = xml_file
        if xml_file:
            # Load and parse XML file
            self.tree = etree.parse(xml_file)
            self.root = self.tree.getroot()
        else:
            # Create an empty nodes element
            self.root = etree.Element("nodes")
            self.tree = etree.ElementTree(self.root)

    def add_node(self, id, x, y, type=None):
        # Create a new node element
        node = etree.SubElement(self.root, "node")
        node.set("id", id)
        node.set("x", str(x))
        node.set("y", str(y))
        if type is not None:
            node.set("type", type)

    def save(self, path):
        # Save the XML to a file
        self.tree.write(path, pretty_print=True)

    def tostring(self):
        # Return a string representation of the XML
        return etree.tostring(self.root, pretty_print=True).decode()

    def load(self, xml_file):
        # Load and parse XML file
        self.tree = etree.parse(xml_file)
        self.root = self.tree.getroot()
        self.xml_file = xml_file

    def get_nodes(self):
        # Get all nodes as a list of dictionaries
        nodes = []
        for node in self.root.findall("node"):
            nodes.append({
                "id": node.get("id"),
                "x": node.get("x"),
                "y": node.get("y"),
                "type": node.get("type")
            })
        return nodes


class EdgesXML:
    def __init__(self, xml_file=None):
        self.xml_file = xml_file
        if xml_file:
            # Load and parse XML file
            self.tree = etree.parse(xml_file)
            self.root = self.tree.getroot()
        else:
            # Create an empty edges element
            self.root = etree.Element("edges")
            self.tree = etree.ElementTree(self.root)

    def add_edge(self, id, from_node, to_node, priority, numLanes, speed):
        # Create a new edge element
        edge = etree.SubElement(self.root, "edge")
        edge.set("id", id)
        edge.set("from", from_node)
        edge.set("to", to_node)
        edge.set("priority", str(priority))
        edge.set("numLanes", str(numLanes))
        edge.set("speed", str(speed))

    def save(self, path):
        # Save the XML to a file
        self.tree.write(path, pretty_print=True)

    def tostring(self):
        # Return a string representation of the XML
        return etree.tostring(self.root, pretty_print=True).decode()

    def load(self, xml_file):
        # Load and parse XML file
        self.tree = etree.parse(xml_file)
        self.root = self.tree.getroot()
        self.xml_file = xml_file

    def get_edges(self):
        # Get all edges as a list of dictionaries
        edges = []
        for edge in self.root.findall("edge"):
            edges.append({
                "id": edge.get("id"),
                "from": edge.get("from"),
                "to": edge.get("to"),
                "priority": edge.get("priority"),
                "numLanes": edge.get("numLanes"),
                "speed": edge.get("speed")
            })
        return edges


class RoutesXML:
    def __init__(self, xml_file=None):
        self.xml_file = xml_file
        if xml_file:
            # Load and parse XML file
            self.tree = etree.parse(xml_file)
            self.root = self.tree.getroot()
        else:
            # Create an empty routes element
            self.root = etree.Element("routes")
            self.root.set("{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation",
                          "http://sumo.dlr.de/xsd/routes_file.xsd")
            self.tree = etree.ElementTree(self.root)

    def add_trip(self, id, depart, from_edge, to_edge):
        # Create a new trip element
        trip = etree.SubElement(self.root, "trip")
        trip.set("id", id)
        trip.set("depart", str(depart))
        trip.set("from", from_edge)
        trip.set("to", to_edge)

    def save(self, path):
        # Save the XML to a file
        self.tree.write(path, pretty_print=True, xml_declaration=True, encoding="UTF-8")

    def tostring(self):
        # Return a string representation of the XML
        return etree.tostring(self.root, pretty_print=True, xml_declaration=True, encoding="UTF-8").decode()

    def load(self, xml_file):
        # Load and parse XML file
        self.tree = etree.parse(xml_file)
        self.root = self.tree.getroot()
        self.xml_file = xml_file

    def get_trips(self):
        # Get all trips as a list of dictionaries
        trips = []
        for trip in self.root.findall("trip"):
            trips.append({
                "id": trip.get("id"),
                "depart": trip.get("depart"),
                "from": trip.get("from"),
                "to": trip.get("to"),
            })
        return trips


def make_nodes_xml():
    # Let's test the NodesXML class
    nodes_xml = NodesXML()
    nodes_xml.add_node("n1", 0, 0, "traffic_light")
    nodes_xml.add_node("n2", 1, 1)
    print(nodes_xml.tostring())
    nodes_xml.save("../data/test_nodes.xml")


def make_edges_xml():
    # Let's test the EdgesXML class
    edges_xml = EdgesXML()
    edges_xml.add_edge("1fi", "1", "m1", 2, 2, 11.11)
    edges_xml.add_edge("1si", "m1", "0", 3, 3, 13.89)
    print(edges_xml.tostring())
    edges_xml.save("../data/test_edges.xml")


def make_routes_xml():
    # Let's test the RoutesXML class
    routes_xml = RoutesXML()
    routes_xml.add_trip("0", 0.00, "edge3973", "edge1111")
    routes_xml.add_trip("1", 1.00, "edge2262", "edge202")
    print(routes_xml.tostring())
    routes_xml.save("../data/test_routes.xml")

# from SUMO documentation
def make_nodes_xml_manual():
    s = """
<nodes> <!-- The opening tag -->

  <node id="0" x="0.0" y="0.0" type="traffic_light"/> <!-- def. of node "0" -->

  <node id="1" x="-500.0" y="0.0" type="priority"/> <!-- def. of node "1" -->
  <node id="2" x="+500.0" y="0.0" type="priority"/> <!-- def. of node "2" -->
  <node id="3" x="0.0" y="-500.0" type="priority"/> <!-- def. of node "3" -->
  <node id="4" x="0.0" y="+500.0" type="priority"/> <!-- def. of node "4" -->

  <node id="m1" x="-250.0" y="0.0" type="priority"/> <!-- def. of node "m1" -->
  <node id="m2" x="+250.0" y="0.0" type="priority"/> <!-- def. of node "m2" -->
  <node id="m3" x="0.0" y="-250.0" type="priority"/> <!-- def. of node "m3" -->
  <node id="m4" x="0.0" y="+250.0" type="priority"/> <!-- def. of node "m4" -->

</nodes> <!-- The closing tag -->
    """

    with open('../data/test_nodes.nod.xml', 'w+') as file:
        file.write(s)


def make_edges_xml_manual():
    s = """
<edges>

  <edge id="1fi" from="1" to="m1" priority="2" numLanes="2" speed="11.11"/>
  <edge id="1si" from="m1" to="0" priority="3" numLanes="3" speed="13.89"/>
  <edge id="1o" from="0" to="1" priority="1" numLanes="1" speed="11.11"/>

  <edge id="2fi" from="2" to="m2" priority="2" numLanes="2" speed="11.11"/>
  <edge id="2si" from="m2" to="0" priority="3" numLanes="3" speed="13.89"/>
  <edge id="2o" from="0" to="2" priority="1" numLanes="1" speed="11.11"/>

  <edge id="3fi" from="3" to="m3" priority="2" numLanes="2" speed="11.11"/>
  <edge id="3si" from="m3" to="0" priority="3" numLanes="3" speed="13.89"/>
  <edge id="3o" from="0" to="3" priority="1" numLanes="1" speed="11.11"/>

  <edge id="4fi" from="4" to="m4" priority="2" numLanes="2" speed="11.11"/>
  <edge id="4si" from="m4" to="0" priority="3" numLanes="3" speed="13.89"/>
  <edge id="4o" from="0" to="4" priority="1" numLanes="1" speed="11.11"/>

</edges>
    """

    with open('../data/test_edges.edg.xml', 'w+') as file:
        file.write(s)


if __name__ == '__main__':
    make_nodes_xml_manual()
    make_edges_xml_manual()
