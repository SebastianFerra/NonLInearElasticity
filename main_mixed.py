 

from ngsolve import *
from netgen.geom2d import SplineGeometry

from netgen.occ import *
import netgen.meshing as ngm
# from ngsPETSc import NonLinearSolver
# from mpi4py.MPI import COMM_WORLD
import problems
import numpy as np
import params
import pickle
from time import time

## get problem parameters and geometry
problem = problems.problem1

phi0 = problem[0]['phi0']
chi = problem[0]['chi']
G = problem[0]['G']
geom = problem[1]
BC = problem[2]
h = 1
ord = 1
N = params.N
KBTV = params.KBTV
form = "EDP" # EDP //functional

## Generate mesh and geometry ### add parallel stuff
def mesher(geom, h):
    geo = OCCGeometry(geom)
    mesh = Mesh(geo.GenerateMesh(maxh=h))
    print(mesh.GetBoundaries())
    return mesh
mesh = mesher(geom, h)


def F(u):
    return Id(3) + Grad(u)


def Norm(vec):
    return InnerProduct(vec, vec)**0.5

def div_custom(A):
    return CF((div(A[0]), div(A[1]), div(A[2])))

def Gel_energy_EDP(F): ## |F|^2 + H => gamma F:Gradv + H'*J'
    # ddet(A(t))/dt = det(A(t))*trace(A^-1(t)*Grad (v))
    gamma = G/KBTV
    J = Det(F)
    phi = phi0/J
    dv = Grad(v)
    invF = Inv(F)
    H_prime = -phi/N + log(1-phi) + phi + chi*phi**2
    edp = gamma * InnerProduct(F,dv) + H_prime * J * Trace(invF*dv)
    return edp

def Gel_energy_mixed(u,v,Ptup,Ttup,Paux,Taux,lam,mu): ## |F|^2 + H => gamma F:Gradv + H'*J'
    F = Id(3) + Grad(u)
    gamma = G/KBTV
    J = Det(F)
    phi = phi0/J
    invF = Inv(F)
    H_prime = -phi/N + log(1-phi) + phi + chi*phi**2
    # P = F+H_prime*J*F^{-T}
    tens_eq = InnerProduct( gamma * F + H_prime * J * invF.trans - Paux , Taux)
                       
    div_eq = InnerProduct(div_custom(Ptup),v) 
    # agregar int((u1,u2,0) * tau.n) = 0 (this only on BC z = 0)
    return tens_eq + div_eq + lam * u + mu * v 


## Generate spaces and forms
fesU = VectorH1(mesh, order=ord)
fesP1 = HDiv(mesh, order=ord+1, dirichlet = "top|right|back|front|left|bottom")
fesP = FESpace([fesP1, fesP1, fesP1])
R = NumberSpace(mesh)
fes = fesU * fesP * R * R * R
u,P1,P2,P3,mu1,mu2,mu3 = fes.TrialFunction()
P = (P1,P2,P3)
v,T1,T2,T3,lam1,lam2,lam3 = fes.TestFunction()
T = (T1,T2,T3)
BF = BilinearForm(fes)

lam = CF((lam1,lam2,lam3))
mu = CF((mu1,mu2,mu3))

## Assemble forms
def Assemble_Bilinear_Form(BF, u,v=None,P=None,T=None,lam=None,mu=None, form = "Mixed"):
    if form == "EDP":
        BF += Gel_energy_EDP(F).Compile() * dx
        return BF
    elif form == "Mixed":
        Paux = CoefficientFunction((P[0][0],P[0][1],P[0][2],P[1][0],P[1][1],P[1][2],P[2][0],P[2][1],P[2][2]), dims = (3,3))
        Taux = CoefficientFunction((T[0][0],T[0][1],T[0][2],T[1][0],T[1][1],T[1][2],T[2][0],T[2][1],T[2][2]), dims = (3,3))
        BF += Gel_energy_mixed(u,v,P,T,Paux,Taux,lam,mu).Compile() * dx
        # bottom, left, front
        # bot_u = CoefficientFunction((u[0],u[1],0))
        # left_u = CoefficientFunction((u[0],0,u[2]))
        # front_u = CoefficientFunction((0,u[1],u[2]))
        # ## add multiplication by normal explicitly?
        # BF += Taux*bot_u*ds(definedon = mesh.Boundaries("bottom"))
        # BF += Taux*left_u*ds(definedon = mesh.Boundaries("left"))
        # BF += Taux*front_u*ds(definedon = mesh.Boundaries("front"))
        # add zero mean per component condition
        return BF

BF = Assemble_Bilinear_Form(BF, u,v,P,T,lam,mu)

def Solver_freeswell(BF, gfu, tol=1e-8, maxiter=250, damp = 0.5):
    """
    Solves the problem
    """
    res = gfu.vec.CreateVector()
    w = gfu.vec.CreateVector()
    history = GridFunction(fes, multidim = 0)
    # here we may need to add another loop
   
    for iter in range(maxiter):
        # Prints before the iteration: number of it, residual, energy
        print("Energy: ", BF.Energy(gfu.vec), "Residual: ", sqrt(abs(InnerProduct(res,res))), "Iteration: ", iter)
        BF.Apply(gfu.vec, res)
        BF.AssembleLinearization(gfu.vec)
        inv = BF.mat.Inverse(freedofs = fes.FreeDofs())        
        w.data = damp * inv * res
        gfu.vec.data -= w
        history.AddMultiDimComponent(gfu.vec)
        if sqrt(abs(InnerProduct(w,res))) < tol:
            print("Converged")
            break
    return gfu, history


gfu = GridFunction(fes)
gfu.vec[:] = 0
t1 =  time()
gfu, history = Solver_freeswell(BF, gfu)
print("Time on ngsolve newton:", abs(t1-time()))

#gfu = GridFunction(fes)
# #gfu.vec[:] = 0
# t1 = time() 
# gfu = petsc_solver(fes,BF, gfu)
# print("Time on snes newton:", abs(t1-time()))

# pickle the results, history and mesh for later use
#pickle.dump(history, open(f"Sol_Problem{problem[-1]}/history_{form}.p", "wb"))
pickle.dump(gfu, open(f"Sol_Problem{problem[-1]}/gfu_{form}_mixed.p", "wb"))
pickle.dump(mesh, open(f"Sol_Problem{problem[-1]}/mesh.p", "wb"))
vtk = VTKOutput(ma=mesh, coefs=[gfu], names=["u"], filename=f"freeswekk_EDP_{h}", subdivision=0)
vtk.Do() 

