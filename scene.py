from setup import granularity, width, height, device
import torch
import math


z = torch.zeros(granularity, device=device, dtype=torch.float32)
# the cell indices array, that determines the triangulation of the cell polygons
cell_indices = torch.tensor([[0, i+2, i+1] for i in range(granularity-2)], dtype = torch.int32,
                device = device)

def create_scene_environment(pyredner, distance_in_widths=20.0, light_intensity=100.0):
    """
        convenience function to quickly create a scene environment
        consists of a camera capturing a 256x256 square in the xy plane and a light source evenly illuminating said square
    """
    cam = pyredner.Camera(position=torch.tensor([width/2, height/2, -distance_in_widths * width]),
                            look_at=torch.tensor([width/2, height/2, 0.0]),
                            up=torch.tensor([0.0, -1.0, 0.0]),
                            fov=torch.tensor([180 * math.atan(1/distance_in_widths) / math.pi]),
                            clip_near=1e-2,
                            resolution=(width, height),
                            fisheye=False)


    shape_light = pyredner.Shape(\
        vertices = torch.tensor([[-width + width/2, -height + height/2, -distance_in_widths * width - 5],
                                [ width + width/2, -height + height/2, -distance_in_widths * width - 5],
                                [-width + width/2,  height + height/2, -distance_in_widths * width - 5],
                                [ width + width/2,  height + height/2, -distance_in_widths * width - 5]], device = pyredner.get_device()),
        indices = torch.tensor([[0, 1, 2],[1, 3, 2]],
            dtype = torch.int32, device = pyredner.get_device()),
        uvs = None,
        normals = None,
        material_id = 0)

    
    light = pyredner.AreaLight(shape_id = 0, 
                            intensity = torch.tensor([light_intensity, light_intensity, light_intensity]))
    
    return cam, shape_light, light

def create_scene(pyredner, cam, shape_light, area_lights, cells):
    """ 
        given a camera, the light sources and their shapes, and a list of cells,
        returns a function that renders the scene based on the vertex and reflectance data stored in the cells
        as well as all the shape objects contained therein
    """
    materials = []
    shapes = [shape_light]
    for i, c in enumerate(cells):
        materials = materials + [pyredner.Material(diffuse_reflectance=c.diffuse_reflectance)]

        vertices3d = torch.cat([c.vertices, torch.stack([z], dim=1)], dim=1)
        shapes = shapes + [pyredner.Shape(\
                vertices = vertices3d,
                indices = cell_indices,
                uvs = None,
                normals = None,
                material_id = i)]

    scene = pyredner.Scene(cam, shapes, materials, area_lights)
    def render(num_samples, max_bounces):
        for shape, c in zip(shapes[1:], cells):
            shape.vertices[:, 0:2] = c.vertices
        scene_args = pyredner.RenderFunction.serialize_scene(\
                    scene = scene,
                    num_samples = num_samples,
                    max_bounces = max_bounces
                )
        return pyredner.RenderFunction.apply(1, *scene_args)
    return shapes, render


def regularizer(cells):
    """ 
        shape regularization of a list of cells. 
        Computes the sum of each visible cell's ratio of area to circumference squared, weighted with the cell's reflectance.
        The more circular a cell, the higher this number is
    """
    sum_of_regularization = torch.cuda.FloatTensor([0], device=device)
    for c in cells:
        if c.visible:
            area = (c.vertices[:-1, 0] * c.vertices[1:, 1] - c.vertices[:-1, 1] * c.vertices[1:, 0]).sum()
            area += c.vertices[-1, 0] * c.vertices[0, 1] - c.vertices[-1, 1] * c.vertices[0, 0]
            circumference = (c.vertices[:-1] - c.vertices[1:]).pow(2).sum(dim=1).sqrt().sum()
            circumference += (c.vertices[-1] - c.vertices[0]).pow(2).sum().sqrt()

            sum_of_regularization += torch.log(c.diffuse_reflectance[0] + 2).data*area/(circumference**2)
    return sum_of_regularization
