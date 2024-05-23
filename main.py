from ngsolve import *
from netgen.geom2d import SplineGeometry
from netgen.occ import *

import problems
import numpy as np
import params
import pickle
import alive_progress

class Gel():
    """
    Only properties for a specific gel, not problem specific
    """
    def __init__(self, gel):
        self.phi0 = gel['phi0']
        self.chi = gel['chi']
        self.G = gel['G']
        
        

class Geomerty():
    def __init__(self, geom_path):
        self.geometry = self.get_geom(geom_path)

    def get_geom(self, geom_path):
        return OCCGeometry(geom_path)   

    def mesher(self,h=0.1):
        self.geometry = self.geometry.GenerateMesh(maxh=h)

        return self.geometry
        

class GelSolver():
    """
    Here we define the problem specific properties (constants and such)
    """
    def __init__(self, gel, geom, BC,h=1):
        self.gel = Gel(gel)
        self.geom = Geomerty(geom)
        self.mesh = Mesh(self.geom.mesher(h))
        self.fes = VectorH1(self.mesh, order=2, dirichletx = BC["x"], dirichlety = BC["y"], dirichletz = BC["z"])
        self.u = self.fes.TrialFunction()
        self.F = Id(3) + Grad(self.u)
        self.BF = BilinearForm(self.fes, symmetric = True)
        self.Assembled = False
        self.KBTV = params.KBTV
        
    def Assemble_Bilinear_Form(self, form):
        """
        Remember to set parameters as NGsolve parameter objects
        """
        J = Det(self.F)
        gamma = self.gel.G/self.KBTV
        if form == "EDP":
            """
            Analitically calculated variation of the energy
            """

            pass
        elif form == "Functional":
            """
            Nuerically calculated variation of the energy
            """
            self.BF += Variation (((1/params.N)  * self.gel.phi0 * log(self.gel.phi0/J) + (J - self.gel.phi0)*log(1-self.gel.phi0/J) + self.gel.phi0*self.gel.chi * log(1-self.gel.phi0/J)).Compile()*dx)
            self.BF += Variation ((Trace(self.F.trans * self.F)* gamma * 0.5).Compile()*dx)
        self.Assembled = True
    
    def NewtonSolver(self, MAX_ITS=100, tol=1e-6, damp = 0.1):
        
        if not self.Assembled:
            raise Exception("Bilinear form not assembled")
        self.u = GridFunction(self.fes)
        self.u.vec[:] = 0

        self.res = self.u.vec.CreateVector()
        self.w = self.u.vec.CreateVector()
        self.history = GridFunction(self.fes, multidim = 0)

        # here one should calculate the lambda target and stuff
        lams = np.linspace(0,1,10) # newton dampers
        with alive_progress.alive_bar(len(lams)*MAX_ITS) as bar:
            for lam in lams:
                for it in range(MAX_ITS):
                    self.BF.Apply(self.u.vec, self.res)
                    self.BF.AssembleLinearization(self.u.vec)
                    self.inv = self.BF.mat.Inverse(self.fes.FreeDofs())
                    self.w.data = self.inv * self.res
                    self.u.vec.data -= lam * damp *self.w
                    bar()
                self.history.AddMultiDimComponent(self.u.vec)
                    
        return self.u.vec.data
problem = problems.problem1
problem_n = problem.pop(-1)

solver = GelSolver(*problem)

solver.Assemble_Bilinear_Form("Functional")

u = solver.NewtonSolver(MAX_ITS=50)


solver.mesh.ngmesh.Save(f"Sol_Problem1/mesh.vol")
pickle.dump(solver.mesh, open(f"Sol_Problem{problem_n}/mesh.pkl", "wb"))
pickle.dump(solver.history, open(f"Sol_Problem{problem_n}/history.pkl", "wb"))
pickle.dump(u, open(f"Sol_Problem{problem_n}/u.pkl", "wb"))
