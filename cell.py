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
    
    def create_shape(self, threshold):
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
    print('??')
    shape_triangle = pyredner.Shape(\
        vertices = torch.tensor([[-200.0, 150.0, 507], [100.0, 100.0, 507], [-100.5, -150.0, 507]],
            device = pyredner.get_device()),
        indices = torch.tensor([[0, 1, 2]], dtype = torch.int32,
            device = pyredner.get_device()),
        uvs = None,
        normals = None,
        material_id = 0)
    print('???')
    shapes = [shape_light, shape_triangle] + redner_shapes(cells)
    print('????')
    scene = pyredner.Scene(cam, shapes, materials, area_lights)
    print('??????')
    scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
    print('??????????')
    render_fn = pyredner.RenderFunction.apply
    return render_fn(0, *scene_args)