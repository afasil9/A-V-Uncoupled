#%%
print("Fully Coupled")
from mpi4py import MPI
import numpy
import ufl
from petsc4py import PETSc
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem import functionspace, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block, apply_lifting, set_bc
from ufl import SpatialCoordinate, sin, pi, grad, div, variable, diff, dx, cos, curl
from dolfinx.fem import Function, Expression, dirichletbc, form
import numpy as np
from basix.ufl import element
from ufl.core.expr import Expr
from scipy.linalg import norm
from dolfinx.io import VTXWriter

def L2_norm(v: Expr):
    """Computes the L2-norm of v"""
    return np.sqrt(MPI.COMM_WORLD.allreduce(
        assemble_scalar(form(ufl.inner(v, v) * ufl.dx)), op=MPI.SUM))


ti = 0.0  # Start time
T = 0.1  # End time
num_steps = 200  # Number of time steps
d_t = (T - ti) / num_steps  # Time step size

n = 4
degree = 1

domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, mesh.CellType.hexahedron)
t = variable(fem.Constant(domain, ti))
dt = fem.Constant(domain, d_t)

nedelec_elem = element("N1curl", domain.basix_cell(), degree)
V = functionspace(domain, nedelec_elem)
lagrange_element = element("Lagrange", domain.basix_cell(), degree)
V1 = functionspace(domain, lagrange_element)

nu = fem.Constant(domain, default_scalar_type(1))
sigma = fem.Constant(domain, default_scalar_type(1))

x = SpatialCoordinate(domain)

def exact(x, t):
    return ufl.as_vector((
        x[1]**2 + x[0]* t, 
        x[2]**2 + x[1]* t, 
        x[0]**2 + x[2]* t))

def exact1(x, t):
    return (x[0]**2) + (x[1]**2) + (x[2]**2)

uex = exact(x,t)
uex1 = exact1(x,t)

f0 = ufl.as_vector((
    -2 + 3*x[0],
    -2 + 3*x[1],
    -2 + 3*x[2])
)

f1 = fem.Constant(domain, -9.0)

def boundary_marker(x):
    """Marker function for the boundary of a unit cube"""
    # Collect boundaries perpendicular to each coordinate axis
    boundaries = [
        np.logical_or(np.isclose(x[i], 0.0), np.isclose(x[i], 1.0))
        for i in range(3)]
    return np.logical_or(np.logical_or(boundaries[0],
                                        boundaries[1]),
                            boundaries[2])

gdim = domain.geometry.dim
facet_dim = gdim - 1

facets = mesh.locate_entities_boundary(domain, dim=facet_dim,
                                        marker= boundary_marker)

bdofs0 = fem.locate_dofs_topological(V, entity_dim=facet_dim, entities=facets)
u_bc_expr_V = Expression(uex, V.element.interpolation_points())
u_bc_V = Function(V)
u_bc_V.interpolate(u_bc_expr_V)
bc_ex = dirichletbc(u_bc_V, bdofs0)

bdofs1 = fem.locate_dofs_topological(V1, entity_dim=facet_dim, entities=facets)
u_bc_expr_V1 = Expression(uex1, V1.element.interpolation_points())
u_bc_V1 = Function(V1)
u_bc_V1.interpolate(u_bc_expr_V1)
bc_ex1 = dirichletbc(u_bc_V1, bdofs1)

bc = [bc_ex, bc_ex1]

u_n = fem.Function(V)
uex_expr = Expression(uex, V.element.interpolation_points())
u_n.interpolate(uex_expr)

u_n1 = fem.Function(V1)
uex_expr1 = Expression(uex1, V1.element.interpolation_points())
u_n1.interpolate(uex_expr1)

u0 = ufl.TrialFunction(V)
v0 = ufl.TestFunction(V)

u1 = ufl.TrialFunction(V1)
v1 = ufl.TestFunction(V1)

a00 = dt*nu*ufl.inner(curl(u0), curl(v0)) * dx + sigma*ufl.inner(u0, v0) * dx
L0 = dt* ufl.inner(f0, v0) * dx + sigma*ufl.inner(u_n, v0) * dx 

a01 = dt * sigma*ufl.inner(grad(u1), v0) * dx
a10 = sigma*ufl.inner(grad(v1), u0) *dx

a11 = dt * ufl.inner(sigma*ufl.grad(u1), ufl.grad(v1)) * ufl.dx
L1 = dt * f1 * v1 * ufl.dx + sigma*ufl.inner(grad(v1),u_n) *dx

a = form([[a00, a01], [a10, a11]])

A_mat = assemble_matrix_block(a, bcs = bc)
A_mat.assemble()

L = form([L0, L1])

b = assemble_vector_block(L, a, bcs = bc)
print("A_mat norm is", A_mat.norm())
print("b norm is", b.norm())

offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A_mat)
ksp.setType("preonly")

pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")

opts = PETSc.Options()
opts["mat_mumps_icntl_14"] = 80
opts["mat_mumps_icntl_24"] = 1
opts["mat_mumps_icntl_25"] = 0
opts["ksp_error_if_not_converged"] = 1
ksp.setFromOptions()

ksp.setUp()

uh = Function(V)
uh1 = Function(V1)

sol = A_mat.createVecRight()

ksp.solve(b, sol)
uh.x.array[:] = sol.array_r[:offset]
uh1.x.array[:] = sol.array_r[offset:]

u_n.x.array[:] = uh.x.array
u_n1.x.array[:] = uh1.x.array

u1_file = VTXWriter(domain.comm, "u1.bp", u_n1, "BP4")
u1_file.write(t.expression().value)

#%%
for n in range(num_steps):
    t.expression().value += d_t
    
    # print("Norm of u_n1",L2_norm(u_n1))

    u_bc_V.interpolate(u_bc_expr_V)
    u_bc_V1.interpolate(u_bc_expr_V1)

    b = assemble_vector_block(L, a, bcs=bc)

    # print("Norm of b", b.norm())

    sol = A_mat.createVecRight()

    # print("Norm of sol", sol.norm())

    ksp.solve(b, sol)

    # print("Norm of sol after solve", sol.norm())

    uh.x.array[:] = sol.array_r[:offset]
    uh1.x.array[:] = sol.array_r[offset:]

    # print("Norm of uh", L2_norm(uh))
    # print("Norm of uh1", L2_norm(uh1))

    u_n.x.array[:] = uh.x.array
    u_n1.x.array[:] = uh1.x.array

    u1_file.write(t.expression().value)

print("u Error", L2_norm(curl(u_n - uex)))
print("u1 Error",L2_norm(u_n1 - uex1))
