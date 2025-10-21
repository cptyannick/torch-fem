import torch

from torchfem import Planar, Solid
from torchfem.materials import IsotropicHencky3D


def test_checkpointing():
    torch.set_default_dtype(torch.float64)

    # Material
    material = IsotropicHencky3D(E=100.0, nu=0.3)

    # Mesh
    nodes = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    elements = torch.tensor([[0, 1, 2, 3]])
    box = Solid(nodes, elements, material)

    # Constrain face at yz-plane
    box.constraints[0, :] = True
    box.constraints[2, :] = True
    box.constraints[3, :] = True

    # Apply force on the free node
    box.forces[1, 0] = 1.0

    # Solve with and without checkpointing
    u_base, _, _, _, _ = box.solve(nlgeom=True)
    u_check, _, _, _, _ = box.solve(nlgeom=True, n_checkpointed=2)

    # Check that the results are the same
    assert torch.allclose(u_base, u_check)
