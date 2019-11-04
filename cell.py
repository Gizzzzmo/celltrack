import torch
from setup import xy, width, height, device
class Cell:

    def __init__(self, pose_matrix, position):
        self.visible = True
        self.pose_matrix = pose_matrix
        self.position = position

        self.b = position.expand(1, -1).transpose(0, 1)
    
    def render(self, x, stage=1):
        diff = x-self.b
        transformed = torch.matmul(self.pose_matrix, diff)
        transformed_dist = torch.sum(transformed.pow(2), dim=(-2,)).pow(stage)

        return 255*torch.exp(-transformed_dist)
    
    def delete(self):
        self.visible = False


def positions(cells):
    return [cell.position for cell in cells if cell.visible]

def pose_matrices(cells):
    return [cell.pose_matrix for cell in cells if cell.visible]

def render_simulation(cells, stage=1):
    simulated = torch.zeros((width, height), device=device)
    for cell in cells:
        if(cell.visible):
            simulated += cell.render(xy, stage)

    return simulated