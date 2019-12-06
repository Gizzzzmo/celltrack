import pyredner
import torch
import numpy as np
import cell
import math
import imageio
import skimage
import glob
from setup import width, height, granularity, device
import load
import matplotlib.pyplot as plt

cells = load.simulated_ellipses()[0]

targets = []
for path in sorted(glob.glob('../data/stemcells/01/*.tif')):
    raw_image = imageio.imread(path)

    target = torch.from_numpy(skimage.transform.rescale(raw_image,
        (width/raw_image.shape[0], height/raw_image.shape[1]))).float().to(device)

    targets = targets + [target]

pyredner.set_use_gpu(torch.cuda.is_available())
factor = 2**2
distance_in_widths = 10.0

cell_indices = torch.tensor([[0, i+2, i+1] for i in range(granularity-2)], dtype = torch.int32,
                device = device)

materials = []
shapes = []
for i, c in enumerate(cells):
    c.create_polygon(112)
    c.diffuse_reflectance = torch.tensor([0.25, 0.25, 0.25], device=device, requires_grad=True)
    materials = materials + [pyredner.Material(diffuse_reflectance=c.diffuse_reflectance)]

    z = torch.zeros(granularity, device=device, dtype=torch.float32)
    vertices3d = torch.cat([c.vertices, torch.stack([z], dim=1)], dim=1)

    shapes = shapes + [pyredner.Shape(\
            vertices = vertices3d,
            indices = cell_indices,
            uvs = None,
            normals = None,
            material_id = i)]

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

shapes = [shape_light] + shapes

light = pyredner.AreaLight(shape_id = 0, 
                           intensity = torch.tensor([100.0,100.0,100.0])/factor)
area_lights = [light]

scene = pyredner.Scene(cam, shapes, materials, area_lights)

scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1)

render = pyredner.RenderFunction.apply
target = targets[1]

scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
# Render the initial guess

optimizer = torch.optim.Adam(cell.redner_reflectances(cells), lr=1e-2)

for t in range(30):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image
    scene_args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4, # We use less samples in the Adam loop.
        max_bounces = 1)
    # Important to use a different seed every iteration, otherwise the result
    # would be biased.
    img = render(t+1, *scene_args).sum(dim=-1)
    diff = (target - img)
    if (t % 30 == 29):
        plt.figure(1)
        plt.imshow(img.cpu().detach().numpy())
        plt.figure(2)
        plt.imshow(diff.abs().cpu().detach())
        plt.show()
    # Compute the loss function. Here it is L2.
    loss = diff.pow(2).sum()

    # Backpropagate the gradients.
    loss.backward()

    # Take a gradient descent step.
    optimizer.step()

optimizer = torch.optim.Adam(cell.redner_vertices(cells), lr=1000)

for t in range(100):
    print('vertex iteration', t)
    optimizer.zero_grad()
    # Forward pass: render the image
    scene_args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4, # We use less samples in the Adam loop.
        max_bounces = 1)
    # Important to use a different seed every iteration, otherwise the result
    # would be biased.
    img = render(t+1, *scene_args).sum(dim=-1)
    diff = (target - img)
    if (t % 20 == 19):
        plt.figure(1)
        plt.imshow(img.cpu().detach().numpy())
        plt.figure(2)
        plt.imshow(diff.abs().cpu().detach())
        plt.show()
    # Compute the loss function. Here it is L2.
    loss = diff.pow(2).sum()

    # Backpropagate the gradients.
    loss.backward()

    print(cells[0].vertices.grad)

    optimizer.step()

pyredner.imwrite(img.cpu(), 'results/optimize_single_triangle/final.png')
loss = (img-target).abs()
pyredner.imwrite(loss.cpu(), 'results/optimize_single_triangle/finalloss.png')