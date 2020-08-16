import pyredner
import torch
import os
import numpy as np
import cell
import csv
import math
import imageio
import skimage.transform
import glob
import gudhi as gd
from setup import width, height, granularity, device, density_diagram, xy
from collections import deque
import load
from matplotlib import pyplot as plt, animation
from sklearn.cluster import KMeans
import time
from scene import create_scene_environment, create_scene, wiggled_gradients, regularizer


def plot_delete_info(wiggled, reflectances):
    x, y = density_diagram(wiggled)
    plt.figure('wiggled gradients')
    plt.plot(x, y)
    plt.plot(wiggled, np.zeros_like(wiggled), 'ro')
    plt.figure('reflectances')
    x, y, = density_diagram(reflectances)
    plt.plot(x, y)
    plt.plot(reflectances, np.zeros_like(reflectances), 'ro')

def find_threshold(values):
    kmeans = KMeans(n_clusters=2).fit(np.stack([values], axis=-1))
    class_0 = [value for value, label in zip(values, kmeans.labels_) if label == 0]
    class_1 = [value for value, label in zip(values, kmeans.labels_) if label == 1]
    return min(class_0) if (min(class_0) > max(class_1)) else min(class_1)

def run_stage(alpha, reflectance_learn_rate, vertex_learn_rate, circles):
    model_init_string = '_'+str(circles)+'x'+str(circles)+'circles_' + str(granularity)
    model_parameter_string = '_' + str(vertex_learn_rate) +'_' + str(reflectance_learn_rate) + '_' + str(alpha)
    simulated_path = None#'256x256_0.71_4_5e-05'
    simulations = None
    if simulated_path:
        simulations = load.simulated_ellipses(simulated_path)


    targets = deque([])
    for path in sorted(glob.glob('../data/stemcells/02/*.tif')):
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
    cam, shape_light, light = create_scene_environment(pyredner)
    area_lights = [light]

    starttime = time.time()
    avg_final_rend_loss = 0
    avg_final_reg_loss = 0

    for j, target in enumerate(targets):
        print(circles, alpha, j)
        cells = []
        if simulations:
            cells = simulations[j]
            for i, c in enumerate(cells):
                c.create_polygon(112)
                c.diffuse_reflectance = torch.tensor([0.25, 0.25, 0.25], device=device, requires_grad=True)
        else:
            for i in range(width//(circles + 1), circles*width//(circles + 1) + 1, width//(circles + 1)):
                for k in range(height//(circles + 1), circles*height//(circles + 1) + 1, width//(circles + 1)):
                    pos = torch.FloatTensor([i, k]).to(device)
                    M = 2.7*torch.FloatTensor([[3.6e-02, 1.3e-05], [1.3e-05, 3.6e-02]]).to(device)
                    c = cell.Cell(M, pos, 0)

                    c.create_polygon(100)
                    c.diffuse_reflectance = torch.tensor([0.25, 0.25, 0.25], device=device, requires_grad=True)
                    cells.append(c)
        
        shapes, render = create_scene(pyredner, cam, shape_light, area_lights, cells)

        optimizerref = torch.optim.Adam(cell.redner_reflectances(cells), lr=1e-2)
        wiggled = np.array(wiggled_gradients([c for c in cells if c.visible], render, target))
        reflectances = np.array([c.diffuse_reflectance.sum().item() for c in cells if c.visible])
        #print('wiggled log: ', find_threshold(np.log(wiggled)))
        #print('wiggled: ', find_threshold(wiggled))
        if plotit:
            plot_delete_info(wiggled, reflectances)
            plt.show()
        # Optimize material properties of cells
        for t in range(30):
            print('iteration:', t)
            optimizerref.zero_grad()
            # Forward pass: render the image
            img = render(4, 1).sum(dim=-1)
            diff = (target - img)
            if (t % 30 == 29) and plotit:
                plt.figure(1)
                plt.imshow(img.cpu().detach().numpy(), cmap='gray')
                plt.figure(2)
                plt.imshow(diff.abs().cpu().detach(), cmap='gray')
                plt.show()
            # Compute the loss function. Here it is L2.
            simple_loss = diff.pow(2).sum()

            loss = simple_loss
            # Backpropagate the gradients.
            loss.backward(retain_graph=True)

            # Take a gradient descent step.
            optimizerref.step()

        for c in cells:
            print(c.vertices.grad.sum().abs()/granularity)
            if(c.vertices.grad.sum().abs()/granularity < 3e-2):
                c.visible = False
        
        shapes, render = create_scene(pyredner, cam, shape_light, area_lights, [c for c in cells if c.visible])

        optimizerref = torch.optim.Adam(cell.redner_reflectances(cells), lr=reflectance_learn_rate)
        optimizer = torch.optim.Adam(cell.redner_vertices(cells), lr=vertex_learn_rate)
        vertex_sequence = []
        simulation_sequence = []

        # Optimize the shape of the cells

        best_rend_loss = float('inf')
        stopping_counter = 0

        for t in range(120):
            print('vertex iteration', t)
            optimizer.zero_grad()
            optimizerref.zero_grad()
            # Forward pass: render the image
            img = render(4, 1).sum(dim=-1)
            diff = (target - img)

            if saveanimation:
                vlistcolored = cell.render_vertex_list(cells, torch.max(target), target).detach().cpu()
                cpuimg = img.detach().cpu()
                vertex_sequence = vertex_sequence + [vlistcolored.numpy()]
                simulation_sequence = simulation_sequence + [cpuimg.numpy()]

            if(t % 40 == 39) and plotit:
                vlistcolored = cell.render_vertex_list(cells, torch.max(target), target).detach().cpu()
                cpuimg = img.detach().cpu()
                wiggled = np.array(wiggled_gradients([c for c in cells if c.visible], render, target))
                reflectances = np.array([c.diffuse_reflectance.sum().item() for c in cells if c.visible])
                #print('wiggled log: ', np.exp(find_threshold(np.log(wiggled + 1)))-1)
                #print('wiggled: ', find_threshold(wiggled))
                #print('reflectances: ', find_threshold(reflectances))
                #print('reflectances log: ', np.exp(find_threshold(np.log(reflectances + 1)))-1)
                #print('correlation:', np.corrcoef(wiggled, np.exp(reflectances)))
                plt.figure('rendered')
                plt.imshow(cpuimg, cmap='gray')
                plt.figure('difference')
                plt.imshow(diff.abs().cpu().detach(), cmap='gray')
                plt.figure('vertices')
                plt.imshow(vlistcolored, cmap='gray')
                plt.show()

            reg = regularizer([c for c in cells if c.visible])
            print("regularization: ", reg)
            simple_loss = diff.pow(2).sum()
            print("simple_loss: ", simple_loss)
            loss = simple_loss #- alpha * reg
            if simple_loss < best_rend_loss - 0.5:
                stopping_counter = 0
                best_rend_loss = simple_loss.item()
            else:
                stopping_counter += 1
                if stopping_counter == 15:
                    print('stopping early')
                    break
            loss.backward(retain_graph=True)
            
            optimizer.step()
            optimizerref.step()

        avg_final_rend_loss += simple_loss.item()
        avg_final_reg_loss += reg.item()

        if not simulated_path:
            simulated_path = 'other_initilization' + model_init_string 
        target_dir = '../data/stemcells/simulated/' + simulated_path + '/optimized_vertex_lists' + model_parameter_string +  '/'

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        vertex_lists = torch.stack(cell.vertex_lists(cells))
        reflectances = torch.stack(cell.redner_reflectances(cells))
        torch.save(vertex_lists, target_dir+f'{j+1:03}'+'vert2.pt')
        torch.save(reflectances, target_dir+f'{j+1:03}'+'refl2.pt')

        if saveanimation:
            fig = plt.figure()
            im = plt.imshow(np.zeros((width, height)), vmin=0, vmax=255, cmap='gray')
            def animate(i):
                factor = 255.0/np.max(vertex_sequence[i])
                im.set_array(factor* vertex_sequence[i])
                return [im]
            anim = animation.FuncAnimation(fig, animate, init_func=lambda: [im], frames=len(vertex_sequence), interval=1, blit=True)
            anim.save('vertex_animation.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
            plt.show()

            fig = plt.figure()
            im = plt.imshow(np.zeros((width, height)), vmin=0, vmax=255, cmap='gray')
            def animate(i):
                factor = 255.0/np.max(simulation_sequence[i])
                im.set_array(factor* simulation_sequence[i])
                return [im]
            anim = animation.FuncAnimation(fig, animate, init_func=lambda: [im], frames=len(vertex_sequence), interval=1, blit=True)
            anim.save('simulated_cells.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
            plt.show()
    
    timediff = time.time() - starttime
    avg_time = timediff / len(targets)
    avg_final_reg_loss /= len(targets)
    avg_final_rend_loss /= len(targets)

    print('--------------------------\n')
    print(' ', alpha, circles)
    print(' ', avg_time, avg_final_reg_loss, avg_final_rend_loss)
    print('\n--------------------------\n')

    with open(target_dir + 'info2.csv', 'w') as info:
        wr = csv.writer(info, delimiter=',')
        wr.writerow([str(avg_time), str(avg_final_rend_loss), str(avg_final_reg_loss)])



reflectance_learn_rate = 1e-2
vertex_learn_rate = 1e-0
plotit = True
saveanimation = True

for circles in range(11, 13):
    for alpha in [0]:
        run_stage(alpha, reflectance_learn_rate, vertex_learn_rate, circles)