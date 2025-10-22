import torch
from torch import Tensor
from ..base import FEM
from ..elements import Hexa1, Hexa2, Tetra1, Tetra2, Tria1, Quad1, Bar1
from ..sparse import sparse_solve


class Helmholtz(FEM):
    def __init__(
        self,
        nodes: Tensor,
        elements: Tensor,
        k: float,
    ):
        """Initialize the Helmholtz FEM problem."""
        # Dummy material for base class
        from .materials import AcousticMaterial

        material = AcousticMaterial(rho=1.0, c=1.0)
        super().__init__(nodes, elements, material)
        self.k = k

        # Use complex tensors for acoustics
        self._displacements = self._displacements.to(torch.complex128)
        self._forces = self._forces.to(torch.complex128)

        # Set element type depending on number of nodes per element
        if self.n_dim == 2:
            if len(elements[0]) == 3:
                self.etype = Tria1()
                self.ftype = Bar1()
            elif len(elements[0]) == 4:
                self.etype = Quad1()
                self.ftype = Bar1()
            else:
                raise ValueError("Element type not supported for 2D.")
        else:
            if len(elements[0]) == 4:
                self.etype = Tetra1()
                self.ftype = Tria1()
            elif len(elements[0]) == 8:
                self.etype = Hexa1()
                self.ftype = Quad1()
            elif len(elements[0]) == 10:
                self.etype = Tetra2()
                self.ftype = Tria1()
            elif len(elements[0]) == 20:
                self.etype = Hexa2()
                self.ftype = Quad1()
            else:
                raise ValueError("Element type not supported for 3D.")

        if not isinstance(self.etype.faces, torch.Tensor):
            self.etype.faces = torch.tensor(
                self.etype.faces, device=self.nodes.device
            )

        # Set element type specific sizes
        self.n_stress = 1
        self.n_int = len(self.etype.iweights())
        self.n_dofs = self.n_nod

    def __repr__(self) -> str:
        etype = self.etype.__class__.__name__
        return f"<torch-fem Helmholtz ({self.n_nod} nodes, {self.n_elem} {etype} elements)>"

    def eval_shape_functions(
        self, xi: Tensor, u: Tensor | float = 0.0
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Gradient operator at integration points xi."""
        nodes = self.nodes.real + u
        nodes = nodes[self.elements, :]
        N = self.etype.N(xi).to(nodes.dtype)
        b = self.etype.B(xi).to(nodes.dtype)
        J = torch.einsum("jk,mkl->mjl", b, nodes)
        detJ = torch.linalg.det(J)
        if torch.any(detJ <= 0.0):
            raise Exception("Negative Jacobian. Check element numbering.")
        B = torch.einsum("jkl,lm->jkm", torch.linalg.inv(J), b)
        return N, B, detJ

    def stiffness_matrix(self):
        """Compute element stiffness matrix K."""
        K = torch.zeros(
            (self.n_elem, self.etype.nodes, self.etype.nodes),
            dtype=torch.complex128,
            device=self.nodes.device,
        )
        for w, xi in zip(self.etype.iweights(), self.etype.ipoints()):
            _, B, detJ = self.eval_shape_functions(xi)
            K_e = torch.einsum("...ji,...jk->...ik", B, B)
            K += w * K_e * detJ.unsqueeze(-1).unsqueeze(-1)
        return K

    def mass_matrix(self):
        """Compute element mass matrix M."""
        M = torch.zeros(
            (self.n_elem, self.etype.nodes, self.etype.nodes),
            dtype=torch.complex128,
            device=self.nodes.device,
        )
        for w, xi in zip(self.etype.iweights(), self.etype.ipoints()):
            N, _, detJ = self.eval_shape_functions(xi)
            M_e = torch.einsum("...i,...j->...ij", N, N)
            M += w * M_e * detJ.unsqueeze(-1).unsqueeze(-1)
        return M

    def robin_matrix(self, boundary_faces):
        """Compute element Robin boundary matrix."""
        R = torch.zeros(
            (self.n_dofs, self.n_dofs),
            dtype=torch.complex128,
            device=self.nodes.device,
        )
        boundary_elements = self.elements[boundary_faces[:, 0]]
        face_local_indices = self.etype.faces[boundary_faces[:, 1]]
        face_nodes = torch.gather(boundary_elements, 1, face_local_indices)
        face_coords = self.nodes[face_nodes]
        if self.n_dim == 2:
            edge_vecs = face_coords[:, 1, :] - face_coords[:, 0, :]
            edge_lengths = torch.linalg.norm(edge_vecs.real, dim=1)
            M_template = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=edge_lengths.dtype)
            M_faces = (edge_lengths / 6.0).unsqueeze(-1).unsqueeze(-1) * M_template.unsqueeze(0)
        else:  # 3D case
            n_nodes_per_face = face_nodes.shape[1]
            M_faces = torch.zeros(
                (face_nodes.shape[0], n_nodes_per_face, n_nodes_per_face),
                dtype=torch.complex128,
                device=self.nodes.device,
            )
            for w, xi in zip(self.ftype.iweights(), self.ftype.ipoints()):
                N = self.ftype.N(xi).to(face_coords.dtype)
                b = self.ftype.B(xi).to(face_coords.dtype)
                J = torch.einsum("ki,mij->mjk", b, face_coords)
                dS = torch.linalg.norm(
                    torch.linalg.cross(J[..., 0], J[..., 1]), dim=-1
                )
                M_faces += w * torch.einsum("i,j->ij", N, N) * dS.unsqueeze(
                    -1
                ).unsqueeze(-1)

        for i, face in enumerate(face_nodes):
            R[face[:, None], face] += M_faces[i]
        return R

    def system_matrix(self, boundary_faces=None):
        """Assemble the system matrix A = K - k^2 * M."""
        K = self.stiffness_matrix()
        M = self.mass_matrix()
        A_e = K - self.k**2 * M
        A = self.assemble_matrix(A_e)
        if boundary_faces is not None:
            R = self.robin_matrix(boundary_faces)
            A += 1j * self.k * R
        return A

    def assemble_matrix(self, A_e):
        """Assemble the global system matrix from element matrices."""
        A = torch.zeros(
            (self.n_dofs, self.n_dofs),
            dtype=torch.complex128,
            device=self.nodes.device,
        )
        for i, el in enumerate(self.elements):
            A[el[:, None], el] += A_e[i]
        return A

    def assemble_force_neumann(self, boundary_faces, values):
        """Assemble the Neumann boundary force vector."""
        f_N = torch.zeros(
            self.n_dofs, dtype=torch.complex128, device=self.nodes.device
        )
        boundary_elements = self.elements[boundary_faces[:, 0]]
        face_local_indices = self.etype.faces[boundary_faces[:, 1]]
        face_nodes = torch.gather(boundary_elements, 1, face_local_indices)
        if self.n_dim == 2:
            edge_coords = self.nodes[face_nodes]
            edge_vecs = edge_coords[:, 1, :] - edge_coords[:, 0, :]
            edge_lengths = torch.linalg.norm(edge_vecs.real, dim=1)
            weights = torch.tensor([0.5, 0.5], dtype=values.dtype, device=values.device)
            load_edges = (edge_lengths * values).unsqueeze(-1) * weights.unsqueeze(0)
            for i, edge in enumerate(face_nodes):
                f_N[edge] += load_edges[i]
        else:  # 3D case
            face_coords = self.nodes[face_nodes].real
            n_nodes_per_face = face_nodes.shape[1]
            f_faces = torch.zeros(
                (face_nodes.shape[0], n_nodes_per_face),
                dtype=torch.complex128,
                device=self.nodes.device,
            )
            for w, xi in zip(self.ftype.iweights(), self.ftype.ipoints()):
                N = self.ftype.N(xi).to(face_coords.dtype)
                b = self.ftype.B(xi).to(face_coords.dtype)
                J = torch.einsum("ki,mij->mjk", b, face_coords)
                dS = torch.linalg.norm(
                    torch.linalg.cross(J[..., 0], J[..., 1]), dim=-1
                )
                f_faces += (
                    w * N.unsqueeze(0) * values.unsqueeze(-1) * dS.unsqueeze(-1)
                )
            for i, face in enumerate(face_nodes):
                f_N[face] += f_faces[i]
        return f_N

    def solve(self, neumann_bc=None, robin_bc=None):
        # 1. Get element-wise system matrices and assemble global system matrix
        A = self.system_matrix(robin_bc)

        # 2. Get constrained DOFs
        con = torch.nonzero(self.constraints[:, 0].ravel()).ravel()

        # 3. Get right-hand side
        f = self.forces[:, 0].ravel().to(torch.complex128)
        if neumann_bc is not None:
            f += self.assemble_force_neumann(
                neumann_bc["faces"], neumann_bc["values"]
            )

        # 4. Apply Dirichlet boundary conditions
        if len(con) > 0:
            f -= A[:, con] @ self.displacements[:, 0].ravel()[con]
            A[con, :] = 0.0
            A[:, con] = 0.0
            A[con, con] = 1.0
            f[con] = self.displacements[:, 0].ravel()[con]

        # 5. Solve for unknown pressures
        p = torch.linalg.solve(A, f)
        return p.reshape(-1, 1)

    def compute_f(self, detJ: Tensor, B: Tensor, S: Tensor):
        pass

    def compute_k(self, detJ: Tensor, BCB: Tensor) -> Tensor:
        pass

    def plot(self, u: float | Tensor = 0.0, **kwargs):
        pass
