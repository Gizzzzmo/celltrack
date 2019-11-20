import torch
import math
from setup import xy, width, height, device, depth, granularity, pyredner, cell_indices, cam, materials, area_lights, shape_light
class Cell:

    def __init__(self, pose_matrix, position):
        self.visible = True
        self.pose_matrix = pose_matrix
        self.position = position
        self.vertices = None
        self.shape = None

        self.b = position.expand(1, -1).transpose(0, 1)
    @classmethod
    def from_vertices(cls, vertices):
        cell = Cell(None, None)

        self.visible = True
        cell.vertices = vertices
        return cell


    def render(self, x, stage=1):
        diff = x-self.b
        transformed = torch.matmul(self.pose_matrix, diff)
        transformed_dist = torch.sum(transformed.pow(2), dim=(-2,)).pow(stage)

        return 255*torch.exp(-transformed_dist)
    
    def delete(self):
        self.visible = False

    def create_polygon(self, threshold):
        # phi is the set of angles to sample
        # TODO one could potentially sample the angles more closely where one is closer to the main axis,
        # so as to ultimately have equidistant points on the ellipse boundary
        phi = torch.arange(0, 2*math.pi, 2*math.pi/granularity, device=device, dtype=torch.float32)
        # r are are the resulting directional vectors of the angles
        r = torch.stack([torch.cos(phi), torch.sin(phi)])
        # multiplying the pose_matrix with r yields the set of distorted vectors
        offset = (math.pow(-math.log(threshold/255), .25) / torch.matmul(self.pose_matrix, r).pow(2).sum(dim=0).pow(.5)) * r
        z = depth + torch.zeros(granularity, device=device, dtype=torch.float32)
        self.vertices = torch.cat([self.position - offset.transpose(0, 1), torch.stack([z], dim=1)], dim=1).detach()
        self.vertices.requires_grad = True
    
    def create_shape(self, threshold, newpolygon=False):
        if (self.vertices == None) or newpolygon: 
            self.create_polygon(threshold)
        
        self.shape = pyredner.Shape(\
            vertices = self.vertices,
            indices = cell_indices,
            uvs = None,
            normals = None,
            material_id = 0)



def positions(cells):
    return [cell.position for cell in cells if cell.visible]

def pose_matrices(cells):
    return [cell.pose_matrix for cell in cells if cell.visible]

def redner_shapes(cells):
    return [cell.shape for cell in cells if cell.visible]

def render_simulation(cells, stage=1, simulated=torch.zeros((width, height), device=device)):
    for cell in cells:
        if(cell.visible):
            simulated += cell.render(xy, stage)

    return simulated

def render_vertex_list(cells, value=255, simulated=torch.zeros((width, height), device=device)):
    for cell in cells:
        if(cell.visible):
            for i in range(len(cell.vertices)):
                simulated[int(cell.vertices[i, 1]), int(cell.vertices[i, 0])] = value
    return simulated

def redner_simulation(cells):
    shapes = [shape_light] + redner_shapes(cells)
    scene = pyredner.Scene(cam, shapes, materials, area_lights)

    scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1)

    return pyredner.RenderFunction.apply(0, *scene_args)