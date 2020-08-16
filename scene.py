from setup import granularity, width, height, device
import torch
import math
import numpy as np


z = torch.zeros(granularity, device=device, dtype=torch.float32)
cell_indices = torch.tensor([[0, i+2, i+1] for i in range(granularity-2)], dtype = torch.int32,
                device = device)

def create_scene_environment(pyredner, distance_in_widths=20.0, light_intensity=100.0):
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
    sum_of_regularization = torch.cuda.FloatTensor([0], device=device)
    for c in cells:
        if c.visible:
            area = (c.vertices[:-1, 0] * c.vertices[1:, 1] - c.vertices[:-1, 1] * c.vertices[1:, 0]).sum()
            area += c.vertices[-1, 0] * c.vertices[0, 1] - c.vertices[-1, 1] * c.vertices[0, 0]
            circumference = (c.vertices[:-1] - c.vertices[1:]).pow(2).sum(dim=1).sqrt().sum()
            circumference += (c.vertices[-1] - c.vertices[0]).pow(2).sum().sqrt()

            sum_of_regularization += torch.log(c.diffuse_reflectance[0] + 2).data*area/(circumference**2)
    return sum_of_regularization

def wiggled_gradients(cells, render, target):
    for c in cells:
        c.gradient_sum = 0
    for offset in [-1, 1]:
        for index in [0, 1]:
            for c in cells:
                c.vertices.data[:, index] += offset
            img = render(4, 1).sum(dim=-1)
            diff = (target - img)
            loss = diff.pow(2).sum()

            loss.backward(retain_graph=True)

            for c in cells:
                c.gradient_sum += c.vertices.grad.abs().sum()
                c.vertices.data[:, index] -= offset
    
    return [c.gradient_sum.item() for c in cells]
