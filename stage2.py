import pyredner
import torch
import os
import numpy as np
import cell
import math
import imageio
import skimage
import glob
from setup import width, height, granularity, device, density_diagram
from collections import deque
import load
from matplotlib import pyplot as plt, animation

simulated_path = None#'256x256_0.71_4_5e-05'
simulations = None
if simulated_path:
    simulations = load.simulated_ellipses(simulated_path)
plotit = True
saveanimation = True


targets = deque([])
for path in sorted(glob.glob('../data/stemcells/01/*.tif')):
    raw_image = imageio.imread(path)

    target = torch.from_numpy(skimage.transform.rescale(raw_image,
        (width/raw_image.shape[0], height/raw_image.shape[1]))).float().to(device)

    targets.append(target)

if simulations:
    print(len(simulations), len(targets))
    popping_front = True
    while len(simulations) < len(targets):
        if popping_front:
            targets.popleft()
        else:
            targets.pop()
        popping_front = not popping_front
    print(len(simulations), len(targets))


pyredner.set_use_gpu(torch.cuda.is_available())
distance_in_widths = 20.0

cell_indices = torch.tensor([[0, i+2, i+1] for i in range(granularity-2)], dtype = torch.int32,
                device = device)

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

z = torch.zeros(granularity, device=device, dtype=torch.float32)

for j, target in enumerate(targets):
    materials = []
    shapes = []
    cells = []
    if simulations:
        cells = simulations[j]
        for i, c in enumerate(cells):
            c.create_polygon(112)
            c.diffuse_reflectance = torch.tensor([0.25, 0.25, 0.25], device=device, requires_grad=True)
            materials = materials + [pyredner.Material(diffuse_reflectance=c.diffuse_reflectance)]

            vertices3d = torch.cat([c.vertices, torch.stack([z], dim=1)], dim=1)
            shapes = shapes + [pyredner.Shape(\
                    vertices = vertices3d,
                    indices = cell_indices,
                    uvs = None,
                    normals = None,
                    material_id = i)]
    else:
        id = 0
        for i in range(width//9, 8*width//9 + 1, width//9):
            for k in range(height//9, 8*height//9 + 1, width//9):
                pos = torch.FloatTensor([i, k]).to(device)
                M = 2.7*torch.FloatTensor([[3.6e-02, 1.3e-05], [1.3e-05, 3.6e-02]]).to(device)
                c = cell.Cell(M, pos, 0)

                c.create_polygon(100)
                c.diffuse_reflectance = torch.tensor([0.25, 0.25, 0.25], device=device, requires_grad=True)
                materials = materials + [pyredner.Material(diffuse_reflectance=c.diffuse_reflectance)]
                cells.append(c)

                vertices3d = torch.cat([c.vertices, torch.stack([z], dim=1)], dim=1)
                shapes = shapes + [pyredner.Shape(\
                        vertices = vertices3d,
                        indices = cell_indices,
                        uvs = None,
                        normals = None,
                        material_id = id)]
                id += 1

    shapes = [shape_light] + shapes

    light = pyredner.AreaLight(shape_id = 0, 
                            intensity = torch.tensor([100.0, 100.0, 100.0]))
    area_lights = [light]

    scene = pyredner.Scene(cam, shapes, materials, area_lights)

    render = pyredner.RenderFunction.apply

    # Render the initial guess

    optimizerref = torch.optim.Adam(cell.redner_reflectances(cells), lr=1e-2)

    def wiggled_gradients():
        for c in cells:
            c.gradient_sum = 0
        for offset in [-1, 1]:
            for index in [0, 1]:
                for shape, c in zip(shapes[1:-1], cells):
                    if c.visible:
                        c.vertices.data[:, index] += offset
                        shape.vertices[:, 0:2] = c.vertices
                scene_args = pyredner.RenderFunction.serialize_scene(\
                    scene = scene,
                    num_samples = 4,
                    max_bounces = 1
                )
                img = render(1, *scene_args).sum(dim=-1)
                diff = (target - img)
                loss = diff.pow(2).sum()

                loss.backward(retain_graph=True)

                for c in cells:
                    if c.visible:
                        c.gradient_sum += c.vertices.grad.abs().sum()
                        c.vertices.data[:, index] -= offset
        
        return [c.gradient_sum.item() for c in cells if c.visible]

    def show_vertices():
        vlistcolored = cell.render_vertex_list(cells, torch.max(target), target)
        plt.imshow(vlistcolored.detach().cpu())
        plt.show()

    if(plotit):
        show_vertices()

    wiggled = wiggled_gradients()
    x, y = density_diagram(wiggled)
    plt.plot(x, y)
    plt.plot(np.array(wiggled), np.zeros((len(wiggled))), 'ro')
    plt.show()
    # Optimize material properties of cells 
    for t in range(30):
        print('iteration:', t)
        optimizerref.zero_grad()
        # Forward pass: render the image
        scene_args = pyredner.RenderFunction.serialize_scene(\
            scene = scene,
            num_samples = 4,
            max_bounces = 1)
        # Important to use a different seed every iteration, otherwise the result
        # would be biased.
        img = render(t, *scene_args).sum(dim=-1)
        diff = (target - img)
        if (t % 30 == 29) and plotit:
            plt.figure(1)
            plt.imshow(img.cpu().detach().numpy())
            plt.figure(2)
            plt.imshow(diff.abs().cpu().detach())
            plt.show()
        # Compute the loss function. Here it is L2.
        loss = diff.pow(2).sum()

        # Backpropagate the gradients.
        loss.backward(retain_graph=True)

        # Take a gradient descent step.
        optimizerref.step()

    optimizerref = torch.optim.Adam(cell.redner_reflectances(cells), lr=1e-2)
    optimizer = torch.optim.Adam(cell.redner_vertices(cells), lr=2e-1)
    vertex_sequence = []
    simulation_sequence = []

    # Optimize the shape of the cells
    for t in range(100):
        print('vertex iteration', t)
        optimizer.zero_grad()
        optimizerref.zero_grad()
        # Forward pass: render the image
        scene_args = pyredner.RenderFunction.serialize_scene(\
            scene = scene,
            num_samples = 4,
            max_bounces = 1
        )
        # Important to use a different seed every iteration, otherwise the result
        # would be biased.
        for shape, c in zip(shapes[1:-1], cells):
            shape.vertices[:, 0:2] = c.vertices
        img = render(t+1, *scene_args).sum(dim=-1)
        diff = (target - img)
        if (t % 1 == 0):
            vlistcolored = cell.render_vertex_list(cells, torch.max(target), target).detach().cpu()
            cpuimg = img.detach().cpu()
            vertex_sequence = vertex_sequence + [vlistcolored.numpy()]
            simulation_sequence = simulation_sequence + [cpuimg.numpy()]
            if(t % 20 == 19) and plotit:
                plt.figure(1)
                plt.imshow(cpuimg)
                plt.figure(2)
                plt.imshow(diff.abs().cpu().detach())
                plt.figure(3)
                plt.imshow(vlistcolored)
                plt.show()
        # Compute the loss function. Here it is L2.
        loss = diff.pow(2).sum()

        # Backpropagate the gradients.
        loss.backward(retain_graph=True)
        before = cells[0].vertices.clone()
        
        optimizer.step()
        optimizerref.step()
        

    wiggled = wiggled_gradients()
    x, y = density_diagram(wiggled)
    plt.plot(x, y)
    plt.plot(np.array(wiggled), np.zeros((len(wiggled))), 'ro')
    plt.show()

    if not simulated_path:
        simulated_path = 'other_initilization'
    target_dir = '../data/stemcells/simulated/' + simulated_path + '/optimized_vertex_lists/'

    if not os.path.exists(target_dir):
            os.makedirs(target_dir)

    vertex_lists = torch.stack(cell.vertex_lists(cells))
    torch.save(vertex_lists, target_dir+f'{j+1:03}'+'.pt')

    if saveanimation:
        fig = plt.figure()
        im = plt.imshow(np.zeros((width, height)), vmin=0, vmax=255)
        def animate(i):
            factor = 255.0/np.max(vertex_sequence[i])
            im.set_array(factor* vertex_sequence[i])
            return [im]
        anim = animation.FuncAnimation(fig, animate, init_func=lambda: [im], frames=len(vertex_sequence), interval=1, blit=True)
        anim.save('vertex_animation.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
        plt.show()

        fig = plt.figure()
        im = plt.imshow(np.zeros((width, height)), vmin=0, vmax=255)
        def animate(i):
            factor = 255.0/np.max(simulation_sequence[i])
            im.set_array(factor* simulation_sequence[i])
            return [im]
        anim = animation.FuncAnimation(fig, animate, init_func=lambda: [im], frames=len(vertex_sequence), interval=1, blit=True)
        anim.save('simulated_cells.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
        plt.show()