import time
import uuid
import math
import tqdm
import ifcopenshell
from bim.gis.osm import Nominatim
from bim.sumo.network import get_stockport_center_boundary

streets_gdf = Nominatim().load_streets_gdf()
boundary_polygon = get_stockport_center_boundary()
streets_gdf = streets_gdf[streets_gdf.geometry.within(boundary_polygon)].copy()


def create_polylines():
    polylines = []
    for t in tqdm.tqdm(streets_gdf.itertuples()):
        rectangle = []
        for coord in t.geometry.coords:
            x,y = coord
            rectangle.append(ifc_file.createIfcCartesianPoint((x, y)))

        polyline = ifc_file.createIfcPolyline(rectangle)
        polylines.append(polyline)
    return polylines


def new_guid():
    return ifcopenshell.guid.compress(uuid.uuid1().hex)


file_name = 'road_network.ifc'
timestamp = time.time()
timestring = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(timestamp))
creator = "P. Polina"
organization = "Poliakova Polina"
application, application_version = "IfcOpenShell", "0.5"
project_globalid, project_name = new_guid(), "Road_network"

ifc_file = ifcopenshell.file()

org = ifc_file.createIfcOrganization()
org.Name = organization

app = ifc_file.createIfcApplication()
app.ApplicationDeveloper = org
app.Version = "3.2"
app.ApplicationFullName = "BlenderBIM"

person = ifc_file.createIfcPerson()
person.FamilyName = "P. Poliakova"

person_org = ifc_file.createIfcPersonAndOrganization()
person_org.ThePerson = person
person_org.TheOrganization = org

owner_hist = ifc_file.createIfcOwnerHistory()
owner_hist.OwningUser = person_org
owner_hist.OwningApplication = app
owner_hist.ChangeAction = "NOCHANGE"
owner_hist.CreationDate = int(time.time())

LengthUnit = ifc_file.createIfcSIUnit()
LengthUnit.UnitType = "LENGTHUNIT"
LengthUnit.Prefix = "MILLI"
LengthUnit.Name = "METRE"

AreaUnit = ifc_file.createIfcSIUnit()
AreaUnit.UnitType = "AREAUNIT"
AreaUnit.Name = "SQUARE_METRE"

VolumeUnit = ifc_file.createIfcSIUnit()
VolumeUnit.UnitType = "VOLUMEUNIT"
VolumeUnit.Name = "CUBIC_METRE"

PlaneAngleUnit = ifc_file.createIfcSIUnit()
PlaneAngleUnit.UnitType = "PLANEANGLEUNIT"
PlaneAngleUnit.Name = "RADIAN"

AngleUnit = ifc_file.createIfcMeasureWithUnit()
AngleUnit.UnitComponent = PlaneAngleUnit
AngleUnit.ValueComponent = ifc_file.createIfcPlaneAngleMeasure(math.pi / 180)

DimExp = ifc_file.createIfcDimensionalExponents(0, 0, 0, 0, 0, 0, 0)

ConvertBaseUnit = ifc_file.createIfcConversionBasedUnit()
ConvertBaseUnit.Dimensions = DimExp
ConvertBaseUnit.UnitType = "PLANEANGLEUNIT"
ConvertBaseUnit.Name = "DEGREE"
ConvertBaseUnit.ConversionFactor = AngleUnit

UnitAssignment = ifc_file.createIfcUnitAssignment([LengthUnit, AreaUnit, VolumeUnit, ConvertBaseUnit])

axis_X = ifc_file.createIfcDirection((1., 0., 0.))
axis_Y = ifc_file.createIfcDirection((0., 1., 0.))
axis_Z = ifc_file.createIfcDirection((0., 0., 1.))
Pnt_O = ifc_file.createIfcCartesianPoint((0., 0., 0.))

WorldCoordinateSystem = ifc_file.createIfcAxis2Placement3D()
WorldCoordinateSystem.Location = Pnt_O
WorldCoordinateSystem.Axis = axis_Z
WorldCoordinateSystem.RefDirection = axis_X

context = ifc_file.createIfcGeometricRepresentationContext()
context.ContextType = "Model"
context.CoordinateSpaceDimension = 3
context.Precision = 1.e-05
context.WorldCoordinateSystem = WorldCoordinateSystem

footprint_context = ifc_file.createIfcGeometricRepresentationSubContext()
footprint_context.ContextIdentifier = 'FootPrint'
footprint_context.ContextType = "Model"
footprint_context.ParentContext = context
footprint_context.TargetView = 'MODEL_VIEW'

myProject = ifc_file.createIfcProject(new_guid())
myProject.OwnerHistory = owner_hist
myProject.Name = "Test OSM"
myProject.RepresentationContexts = [context]
myProject.UnitsInContext = UnitAssignment

site_placement = ifc_file.createIfcLocalPlacement()
site_placement.RelativePlacement = WorldCoordinateSystem
mySite = ifc_file.createIfcSite(new_guid())
mySite.OwnerHistory = owner_hist
mySite.Name = "My Site"
mySite.ObjectPlacement = site_placement
mySite.CompositionType = "ELEMENT"

building_placement = ifc_file.createIfcLocalPlacement()
building_placement.PlacementRelTo = site_placement
building_placement.RelativePlacement = WorldCoordinateSystem

myBuilding = ifc_file.createIfcBuilding(new_guid(), owner_hist)
myBuilding.Name = "Test Building"
myBuilding.ObjectPlacement = building_placement
myBuilding.CompositionType = "ELEMENT"

ground_placement = ifc_file.createIfcLocalPlacement()
ground_placement.PlacementRelTo = building_placement
ground_placement.RelativePlacement = WorldCoordinateSystem

ground = ifc_file.createIfcBuildingStorey(new_guid(), owner_hist)
ground.Name = "Ground"
ground.ObjectPlacement = ground_placement
ground.CompositionType = "ELEMENT"
ground.Elevation = 1000

container_project = ifc_file.createIfcRelAggregates(new_guid(), owner_hist)
container_project.Name = "Project Container"
container_project.RelatingObject = myProject
container_project.RelatedObjects = [mySite]

container_site = ifc_file.createIfcRelAggregates(new_guid(), owner_hist)
container_site.Name = "Site Container"
container_site.RelatingObject = mySite
container_site.RelatedObjects = [myBuilding]

container_storey = ifc_file.createIfcRelAggregates(new_guid(), owner_hist)
container_storey.Name = "Building Container"
container_storey.RelatingObject = myBuilding
container_storey.RelatedObjects = [ground]

container_space = ifc_file.createIfcRelAggregates(new_guid(), owner_hist)
container_space.Name = "Space"
container_space.RelatingObject = myBuilding
container_space.RelatedObjects = [ground]


ifc_file.createIfcRelAggregates(new_guid(), RelatingObject=container_project, RelatedObjects=[container_site])
ifc_file.createIfcRelAggregates(new_guid(), RelatingObject=container_site, RelatedObjects=[container_storey])
ifc_file.createIfcRelAggregates(new_guid(), RelatingObject=container_storey, RelatedObjects=[container_space])

polylines = create_polylines()
print(len(polylines))
representation = ifc_file.createIfcShapeRepresentation(ifc_file.by_type("IfcGeometricRepresentationContext")[0],
                                                       "Plan", "GeometricCurveSet", polylines)
product_definition = ifc_file.createIfcProductDefinitionShape(None, None, [representation])
myGrid = ifc_file.createIfcGrid(new_guid(), owner_hist)
myGrid.Representation = product_definition

container_SpatialStructure = ifc_file.createIfcRelContainedInSpatialStructure(new_guid(), owner_hist)
container_SpatialStructure.Name = 'building_storey_container'
container_SpatialStructure.Description = 'building_storey_container'
container_SpatialStructure.RelatingStructure = ground
container_SpatialStructure.RelatedElements = [myGrid]

ifc_file.write("my_test.ifc")

