import pyredner
import torch
import numpy as np
import matplotlib.pyplot as plt

pyredner.set_use_gpu(torch.cuda.is_available())
width, height = 512, 512
factor = 2**3
cam = pyredner.Camera(position=torch.tensor([0.0, 0.0, -5.0]),
                            look_at=torch.tensor([0.0, 0.0, 0.0]),
                            up=torch.tensor([0.0, 1.0, 0.0]),
                            fov=torch.tensor([45.0]),
                            clip_near=1e-2,
                            resolution=(width, height),
                            fisheye=False)

mat_grey = pyredner.Material(diffuse_reflectance=torch.tensor([0.5, 0.5, 0.5], device=pyredner.get_device()))
materials = [mat_grey]

shape_triangle = pyredner.Shape(\
    vertices = torch.tensor([[-200.0, 150.0, 507], [100.0, 100.0, 507], [-100.5, -150.0, 507]],
        device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2]], dtype = torch.int32,
        device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 0)
shape_light = pyredner.Shape(\
    vertices = torch.tensor([[-width/factor, -height/factor, -7.0],
                            [ width/factor, -height/factor, -7.0],
                            [-width/factor,  height/factor, -7.0],
                            [ width/factor,  height/factor, -7.0]], device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2],[1, 3, 2]],
        dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 0)

shapes = [shape_triangle, shape_light]

light = pyredner.AreaLight(shape_id = 1, 
                           intensity = torch.tensor([20.0,20.0,20.0]))
area_lights = [light]

scene = pyredner.Scene(cam, shapes, materials, area_lights)

scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1)

plt.plot(np.arange(1, 9))
plt.show()
render = pyredner.RenderFunction.apply

img = render(0, *scene_args) 
plt.imshow(img.cpu().detach().numpy())
plt.show()

pyredner.imwrite(img.cpu(), 'results/optimize_single_triangle/target.exr')
pyredner.imwrite(img.cpu(), 'results/optimize_single_triangle/target.png')

target = pyredner.imread('results/optimize_single_triangle/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda()

shape_triangle.vertices = torch.tensor(\
    [[-2.0,1.5,0.3], [0.9,1.2,-0.3], [-0.4,-1.4,0.2]],
    device = pyredner.get_device(),
    requires_grad = True) # Set requires_grad to True since we want to optimize this

scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
# Render the initial guess
img = render(1, *scene_args)
# Save the image
pyredner.imwrite(img.cpu(), 'results/optimize_single_triangle/init.png')

optimizer = torch.optim.Adam([shape_triangle.vertices], lr=5e-2)

for t in range(80):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image
    scene_args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4, # We use less samples in the Adam loop.
        max_bounces = 1)
    # Important to use a different seed every iteration, otherwise the result
    # would be biased.
    img = render(t+1, *scene_args)
    if(t%10 == 0):
        plt.imshow(img.cpu().detach().numpy())
        plt.show()
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()

    # Backpropagate the gradients.
    loss.backward()

    # Take a gradient descent step.
    optimizer.step()

pyredner.imwrite(img.cpu(), 'results/optimize_single_triangle/final.png')
loss = (img-target).abs()
pyredner.imwrite(loss.cpu(), 'results/optimize_single_triangle/finalloss.png')