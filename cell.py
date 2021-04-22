import torch
import math
from shapely.geometry import Polygon
from setup import xy, width, height, device, granularity, device

class Cell:
    """ A class for representing cells, either as a vertex list or as an ellipse with center and a pose matrix"""
    def __init__(self, pose_matrix, position, brightness):
        """ Initializes cell from a position vertex, a pose matrix and a brightness value """
        self.brightness  = torch.FloatTensor([brightness]).to(device)
        self.brightness.requires_grad = True
        self.visible = True
        self.pose_matrix = pose_matrix
        self.position = position
        self.vertices = None
        self.diffuse_reflectance = None
        if position is not None:
            self.b = position.expand(1, -1).transpose(0, 1)

    @classmethod
    def from_vertices(cls, vertices):
        """ Initializes cell from a vertex list """
        cell = Cell(None, None, 0)
        print(vertices.shape)

        cell.visible = True
        cell.vertices = vertices
        return cell


    def render(self, x, stage=1):
        """ renders the ellipse representation onto the xy-grid *x* """
        self.b = self.position.expand(1, -1).transpose(0, 1)
        diff = x-self.b
        transformed = torch.matmul(self.pose_matrix, diff)
        transformed_dist = torch.sum(transformed.pow(2), dim=(-2,)).pow(stage)
        return 255*torch.exp(-transformed_dist) * torch.log(1+torch.exp(self.brightness))
    
    def delete(self):
        """ 'Deletes' a cell by setting its visibility to False """
        self.visible = False

    def create_polygon(self, threshold=112):
        """ Transforms ellipse representation into vertex list representation"""
        # phi is the set of angles to sample
        # one could potentially sample the angles more closely where one is closer to the ellipse's principal axis,
        # so as to ultimately have equidistant points on the ellipse boundary
        phi = torch.arange(0, 2*math.pi, 2*math.pi/granularity, device=device, dtype=torch.float32)
        # r are are the resulting directional vectors of the angles
        r = torch.stack([torch.cos(phi), torch.sin(phi)])
        # multiplying the pose_matrix with r yields the set of distorted vectors
        offset = (math.pow(-math.log(threshold/255), .25) / torch.matmul(self.pose_matrix, r).pow(2).sum(dim=0).pow(.5)) * r
        self.vertices = (self.position - offset.transpose(0, 1)).detach()
        self.vertices.requires_grad = True

    def area(self):
        """ Computes the ellipse's are with respect to its vertex list"""
        area = 0
        for i, (xi1, yi1) in enumerate(self.vertices):
            xi, yi = self.vertices[i-1]
            area -= (xi1 - xi) * (yi1 + yi)
        return area/2

    def area_intersection(self, other):
        """ Computes the area of the intersection of the ellipse with another ellipse *other* with respect to both their vertex lists"""
        return Polygon(self.vertices).intersection(Polygon(other.vertices)).area

    def area_union(self, other):
        """ Computes the area of the union of the ellipse with another ellipse *other* with respect to both their vertex lists"""
        return self.area() + other.area() - self.area_intersection(other)

    def jaccard_index(self, other):
        """ Computes the jaccard index of the ellipse with another ellipse *other* with respect to both their vertex lists"""
        return self.area_intersection(other)/self.area_union(other)

class CellC:

    def __init__(self, collection):
        self.collection = collection

def vertex_lists(cells):
    """ Convenience method to turn a list of cells into a list of their vertex lists if they're visible """
    return [cell.vertices for cell in cells if cell.visible]

def positions(cells):
    """ Convenience method to turn a list of cells into a list of their vertex lists if they're visible """
    return [cell.position for cell in cells if cell.visible]

def pose_matrices(cells):
    """ Convenience method to turn a list of cells into a list of their vertex lists if they're visible """
    return [cell.pose_matrix for cell in cells if cell.visible]

def brightnesses(cells):
    """ Convenience method to turn a list of cells into a list of their vertex lists if they're visible """
    return [cell.brightness for cell in cells if cell.visible]

def redner_vertices(cells):
    """ Convenience method to turn a list of cells into a list of their vertex lists if they're visible """
    return [cell.vertices for cell in cells if cell.visible]

def redner_reflectances(cells):
    """ Convenience method to turn a list of cells into a list of their vertex lists if they're visible """
    return [cell.diffuse_reflectance for cell in cells if cell.visible]

def render_simulation(cells, stage=1, simulated=None):
    """ renders a scene constisting of a list of *cells* onto the *simulated* arrray. If no array is provided, it creates a new one """
    if simulated is None:
        simulated = torch.zeros((width, height), device=device)
    for cell in cells:
        if(cell.visible):
            simulated += cell.render(xy, stage)

    return simulated

def render_vertex_list(cells, value=255, simulated=None):
    """ renders the vertex lists of all the *cells* onto the *simulated* array. Vertices are drawn with the *value* (grayscale) color value """
    if simulated is None:
        simulated = torch.zeros((width, height), device=device)
    else:
        simulated = simulated.clone()
    for cell in cells:
        if(cell.visible):
            for i in range(len(cell.vertices)):
                if (0 <= cell.vertices[i, 1] < width) and (0 <= cell.vertices[i, 0] < height):
                    simulated[int(cell.vertices[i, 1]), int(cell.vertices[i, 0])] = value
    return simulated
