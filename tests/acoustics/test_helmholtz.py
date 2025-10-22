import torch
import pytest
from torchfem.acoustics.helmholtz import Helmholtz
from torchfem.mesh import rect_quad, cube_hexa
from scipy.linalg import eig


@pytest.fixture
def mesh_2d():
    nodes, elements = rect_quad(10, 10)
    nodes = nodes.to(torch.complex128)
    return nodes, elements


@pytest.fixture
def mesh_3d():
    nodes, elements = cube_hexa(10, 10, 10)
    nodes = nodes.to(torch.complex128)
    return nodes, elements


def test_helmholtz_plane_wave_2d(mesh_2d):
    nodes, elements = mesh_2d
    k = 1.0
    problem = Helmholtz(nodes, elements, k)
    p_analytical = torch.exp(-1j * k * nodes[:, 0].real)
    problem.constraints[nodes[:, 0] == 0.0, 0] = True
    problem.constraints[nodes[:, 0] == 1.0, 0] = True
    problem.displacements[nodes[:, 0] == 0.0, 0] = p_analytical[nodes[:, 0] == 0.0]
    problem.displacements[nodes[:, 0] == 1.0, 0] = p_analytical[nodes[:, 0] == 1.0]
    p_fem = problem.solve()
    assert torch.allclose(p_fem.ravel(), p_analytical, atol=1e-1)


def test_helmholtz_reflection_2d(mesh_2d):
    nodes, elements = mesh_2d
    k = 1.0
    problem = Helmholtz(nodes, elements, k)
    p_analytical = torch.cos(k * nodes[:, 0].real)
    problem.constraints[nodes[:, 0] == 0.0, 0] = True
    problem.displacements[nodes[:, 0] == 0.0, 0] = p_analytical[
        nodes[:, 0] == 0.0
    ].to(torch.complex128)
    # Get boundary elements and faces for Neumann condition
    boundary_faces = []
    for i, el in enumerate(elements):
        for j, face in enumerate(problem.etype.faces.cpu().numpy()):
            if torch.all(nodes[el[face], 0].real == 1.0):
                boundary_faces.append((i, j))
    g = -k * torch.sin(torch.tensor(k * 1.0))
    values = torch.full((len(boundary_faces),), g, dtype=torch.complex128)
    neumann_bc = {"faces": torch.tensor(boundary_faces), "values": values}
    p_fem = problem.solve(neumann_bc=neumann_bc)
    assert torch.allclose(p_fem.ravel(), p_analytical.to(torch.complex128), atol=1e-1)


def test_helmholtz_reflection_3d(mesh_3d):
    nodes, elements = mesh_3d
    k = 1.0
    problem = Helmholtz(nodes, elements, k)
    p_analytical = torch.cos(k * nodes[:, 0].real)
    problem.constraints[nodes[:, 0] == 0.0, 0] = True
    problem.displacements[nodes[:, 0] == 0.0, 0] = p_analytical[
        nodes[:, 0] == 0.0
    ].to(torch.complex128)
    # Get boundary elements and faces for Neumann condition
    boundary_faces = []
    for i, el in enumerate(elements):
        for j, face in enumerate(problem.etype.faces.cpu().numpy()):
            if torch.all(nodes[el[face], 0].real == 1.0):
                boundary_faces.append((i, j))
    g = -k * torch.sin(torch.tensor(k * 1.0))
    values = torch.full((len(boundary_faces),), g, dtype=torch.complex128)
    neumann_bc = {"faces": torch.tensor(boundary_faces), "values": values}
    p_fem = problem.solve(neumann_bc=neumann_bc)
    assert torch.allclose(p_fem.ravel(), p_analytical.to(torch.complex128), atol=1e-1)


def test_resonant_cavity_driven():
    nodes, elements = cube_hexa(5, 6, 7)
    nodes = nodes.to(torch.complex128)
    k = torch.pi
    problem = Helmholtz(nodes, elements, k)
    problem.constraints[nodes[:, 0] == 0.0, 0] = True
    problem.constraints[nodes[:, 1] == 0.0, 0] = True
    problem.constraints[nodes[:, 2] == 0.0, 0] = True
    problem.constraints[nodes[:, 0] == 1.0, 0] = True
    problem.constraints[nodes[:, 1] == 1.0, 0] = True
    problem.constraints[nodes[:, 2] == 1.0, 0] = True
    problem.displacements[:, 0] = 0.0
    # Add a point source at the center
    center_node = torch.argmin(torch.linalg.norm(nodes.real - 0.5, dim=1))
    problem.forces[center_node, 0] = 1.0
    p_fem = problem.solve()
    # Check that the solution is non-trivial
    assert torch.linalg.norm(p_fem) > 1e-6


def test_resonant_cavity_eigenmodes():
    nodes, elements = rect_quad(20, 20)
    k = torch.pi
    problem = Helmholtz(nodes, elements, k)
    problem.constraints[nodes[:, 0] == 0.0, 0] = True
    problem.constraints[nodes[:, 0] == 1.0, 0] = True
    problem.constraints[nodes[:, 1] == 0.0, 0] = True
    problem.constraints[nodes[:, 1] == 1.0, 0] = True
    problem.displacements[:, 0] = 0.0
    K = problem.stiffness_matrix()
    M = problem.mass_matrix()
    K = problem.assemble_matrix(K)
    M = problem.assemble_matrix(M)
    # Apply BCs
    con = torch.nonzero(problem.constraints[:, 0].ravel()).ravel()
    free = torch.nonzero(~problem.constraints[:, 0].ravel()).ravel()
    K_free = K[free, :][:, free]
    M_free = M[free, :][:, free]

    K_free = K_free.detach().cpu().numpy()
    M_free = M_free.detach().cpu().numpy()
    eigvals, _ = eig(K_free, M_free)
    k_fem = torch.sqrt(torch.from_numpy(eigvals.real))
    k_analytical = torch.pi * torch.sqrt(
        torch.tensor([1**2 + 1**2, 1**2 + 2**2, 2**2 + 1**2, 2**2 + 2**2], dtype=torch.float64)
    )
    # Get the 4 smallest non-zero eigenvalues
    k_fem = k_fem[k_fem > 1e-6]
    assert torch.allclose(torch.sort(k_fem)[0][:4], torch.sort(k_analytical)[0], atol=1e-1)
