import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# width and height of the images, will be scaled automatically
width = 512*2
height = 512*2

# the z coordinate of the plane in which the simulated cells will be placed with pyredner
depth = 507

# the number of vertices used to simuale one cell
granularity = 100

pyredner = None
cell_indices, cam, materials, area_lights, shape_light = None, None, None, None, None
try:
    import pyredner

    pyredner.set_use_gpu(torch.cuda.is_available())

    cam = pyredner.Camera(position=torch.tensor([0.0, 0.0, -5.0]),
                            look_at=torch.tensor([0.0, 0.0, 0.0]),
                            up=torch.tensor([0.0, 1.0, 0.0]),
                            fov=torch.tensor([45.0]),
                            clip_near=1e-2,
                            resolution=(width, height),
                            fisheye=False)
    mat_grey = pyredner.Material(diffuse_reflectance=torch.tensor([0.5, 0.5, 0.5], device=pyredner.get_device()))
    materials = [mat_grey]

    shape_light = pyredner.Shape(\
    vertices = torch.tensor([[-width, -height, -7.0],
                            [ width, -height, -7.0],
                            [-width,  height, -7.0],
                            [ width,  height, -7.0]], device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2],[1, 3, 2]],
        dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 0)

    light = pyredner.AreaLight(shape_id = 0, 
                            intensity = torch.tensor([20.0,20.0,20.0]))
    area_lights = [light]

    cell_indices = torch.tensor([[0, i+1, i+2] for i in range(granularity-2)], dtype = torch.int32,
                device = pyredner.get_device())
    shape_triangle = pyredner.Shape(\
        vertices = torch.tensor([[-200.0, 150.0, 507], [100.0, 100.0, 507], [-100.5, -150.0, 507]],
            device = pyredner.get_device()),
        indices = torch.tensor([[0, 1, 2]], dtype = torch.int32,
            device = pyredner.get_device()),
        uvs = None,
        normals = None,
        material_id = 0)
    shapes = [shape_light, shape_triangle]
    scene = pyredner.Scene(cam, shapes, materials, area_lights)
    scene_args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 16,
        max_bounces = 1)
    render_fn = pyredner.RenderFunction.apply
    img = render_fn(0, *scene_args)
except:
    print('pyredner not initialized')

# creating a coordinate grid to apply functions to
xx = torch.arange(0, width, 1, device=device, dtype=torch.float32)
yy = torch.arange(0, height, 1, device=device, dtype=torch.float32)
xxx = xx.expand((height, -1))
yyy = yy.expand((width, -1))
yyy = yyy.transpose(0, 1)

xy = torch.stack([xxx, yyy], dim=1)