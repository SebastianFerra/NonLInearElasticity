from ngsolve import *
from ngsolve.webgui import Draw
import numpy as np
import scipy.optimize
import csv

class gel_debonded2D:
    def __init__(self, length, thickness, delta, shear_modulus, entropic_unit, phi0, chi, mass_density):
        self.L = length      # measured in mm
        self.d = thickness    # measured in mm
        self.delta = delta   # dimensionless
        self.G = shear_modulus               # measured in MPa
        self.entropic_unit = entropic_unit  # measured in MPa
        self.phi0 = phi0
        self.chi =  chi  
        self.mass_density = mass_density

        self.gamma = self.G/self.entropic_unit # 0.0009516837481698391 self.compute_gamma( lamb =1.4874):
        self.lambda_target = 1.99  # 0.13/136.6 = gammafun(1.99)  up to rounding errors               

        def auxIsotropic(s):
            return s*self.dH(s*s*s) + self.gamma
        self.lambda_iso = (scipy.optimize.fsolve(auxIsotropic, 1.7))[0]

        def auxEnergyDensity(lambda1, lambda2, lambda3):
            gel=self
            phi0=gel.phi0;
            G=gel.G;
            chi=gel.chi;
            nu=gel.entropic_unit

            J= lambda1*lambda2*lambda3;
            phi = phi0/J;
            return 0.5*G*(lambda1**2 + lambda2**2 + lambda3**2) + nu*((J-phi0)*np.log(1-phi) + phi0*chi*(1-phi))

        lambda_iso = self.lambda_iso
        self.reference_energy_density = auxEnergyDensity(lambda_iso, lambda_iso, lambda_iso)
        
        def auxFunctionUniaxial(s):
            gel = self
            return gel.dH(s) + gel.gamma*s
        self.lambdaUniaxial = (scipy.optimize.fsolve(auxFunctionUniaxial, 2))[0]
        lambdaUniaxial = self.lambdaUniaxial
        self.wUniaxial = auxEnergyDensity(1, lambdaUniaxial, 1) - self.reference_energy_density
        
        def auxFunctionEquibiaxial(s):
            return self.dH(s*s) + self.gamma
        self.lambdaEquiBiaxial = (scipy.optimize.fsolve(auxFunctionEquibiaxial, 2))[0]
        lambdaEquiBiaxial = self.lambdaEquiBiaxial
        self.wEquiBiaxial = auxEnergyDensity(lambdaEquiBiaxial, lambdaEquiBiaxial, 1) - self.reference_energy_density
        
        
    def phi(self, J):
        return self.phi0/J

    def H(self, J):
        return (J - self.phi0)*log(1-self.phi(J))  + self.phi0 * self.chi*(1-self.phi(J))

    def dH(self, J):
        return self.phi(J) + np.log(1-self.phi(J)) + self.chi * self.phi(J)**2
    
    # compute gamma using uniaxial approximation
    # def gammafun(self,lamb):
    #    return -self.dH(lamb)/lamb
    
    def Gfun(self, lamb):
        nu = self.entropic_unit
        return (-self.dH(lamb)/lamb)*nu
    
    
    # energy density
    def W(self, F):
        J = Det(F)
        C = F.trans* F
        
        gel=self
        G=gel.G
        nu=gel.entropic_unit
        reference_energy_density =  gel.reference_energy_density       
        
        return 0.5*G*(Trace(C)+1) + nu*gel.H(J) - reference_energy_density

############
### Main ###
############

# Most commonly changed parameters
# L= 90.0
# d = 1.62
# indexes_deltas = [74, 75, 80]
# order = 3    # polynomial degree of finite elements

# Ex. if d=3.00 then the list of deltas where the indexes_deltas 
#   are given the corresponding indexes for the simularion
#   is in the file 'meshes3_00/deltas'.
# For example, the delta with index 93 for d=3.00 is delta=0.913

# It will be assumed that the arguments passed from shell
# are, in that order:
# the length, the thickness, the order for the finite element space, 
# and a list of indexes for delta.
# For example,
#    90 1.62 3 '4, 93'
# means L=90, d=1.62, order=3, and indexes_delta = [4,93]

data = sys.argv

L = float(data[1])
d = float(data[2])
order = int(data[3])

s = data[4]
indexes_deltas = s.split(",")
n = len(indexes_deltas)
for i in range(n):
    indexes_deltas[i] = int(indexes_deltas[i])

print('L={}, d={}, order={}'.format(L, d, order))
print('indexes_deltas={}'.format(indexes_deltas))

# More stable parameters
G = 0.13               # measured in MPa
entropic_unit = 136.6  # k_B*T/V_m measured in MPa
phi0 = 0.2
chi =  0.348           # J. Elast. 2022, Sect. 3.4
rho = 1.23e-6          # measured in kg/mm³
gravity = CoefficientFunction((0,-9.8))
AA = 1e5

# Initialization of csv_writer
csv_preliminary = []
aux_csv = []
aux_csv.append('delta')
aux_csv.append('H1 relative error')
aux_csv += ['Theoretical energy [mJ/mm]']
aux_csv += ['Gravitational energy [mJ/mm]']
aux_csv += ['Energy with free-slip BC [mJ/mm]', 'Energy without free-slip BC [mJ/mm]', 'Relative error [%]']
aux_csv += ['Deformed length with free-slip BC [mm]', 'Deformed length without free-slip BC [mm]', 'Relative error [%]']
aux_csv += ['Thickness at the middle with free-slip BC', 'Thickness at the middle without free-slip BC', 'Relative error [%]']
aux_csv += ['Thickness at the far right with free-slip BC', 'Thickness at the far right without free-slip BC', 'Relative error [%]']
aux_csv += ['Eigenvalue1 at (0,d) with free-slip BC', 'Eigenvalue2 at (0,d) with free-slip BC']
aux_csv += ['Eigenvalue1 at (0,d) without free-slip BC', 'Eigenvalue2 at (0,d) without free-slip BC']
aux_csv += ['Eigenvalue1 at (L/2,d) with free-slip BC', 'Eigenvalue2 at (L/2,d) with free-slip BC']
aux_csv += ['Eigenvalue1 at (L/2,d) without free-slip BC', 'Eigenvalue2 at (L/2,d) without free-slip BC']
aux_csv += ['ell2 relative error']

csv_preliminary.append(aux_csv)

# Read the list of the values of delta corresponding to the 100 meshes
folder_name_suffix = str(int(d)) + '_' + str(int(d%1*100)).zfill(2)
delta_values = np.loadtxt('meshes' + folder_name_suffix + '/deltas')

for index_delta in indexes_deltas:    
    delta = delta_values[index_delta];
    print("Mesh number = {}, delta={:.3f}".format(index_delta,delta))

    mesh_file = 'meshes' + folder_name_suffix + '/mesh{}.vol'.format(index_delta);
    mesh = Mesh(mesh_file)

    GEL = gel_debonded2D(length=L, thickness=d, delta=delta, shear_modulus = G,\
                         entropic_unit=entropic_unit, phi0=phi0, chi=chi, mass_density=rho)

    fes = VectorH1(mesh, order=3)
    
    filename_suffix = "_d={:.2f}_delta={:.3f}".format(d, delta)
    filename='gridfunctions'+folder_name_suffix+\
            '/result_debonded2D'+filename_suffix+\
            "_order={}".format(order)
    
    u_withoutExtraBC = GridFunction(fes)
    u_withoutExtraBC.Load(filename + '_withoutExtraBC' + '.gfu')
    F_withoutExtraBC = Id(2) + Grad(u_withoutExtraBC)
    C_withoutExtraBC = F_withoutExtraBC.trans*F_withoutExtraBC
    deformation_withoutExtraBC = GridFunction(fes)
    deformation_withoutExtraBC.Set((x,y))
    deformation_withoutExtraBC += u_withoutExtraBC
    
    u_withExtraBC = GridFunction(fes)
    u_withExtraBC.Load(filename + ".gfu")
    F_withExtraBC = Id(2) + Grad(u_withExtraBC)
    C_withExtraBC = F_withExtraBC.trans*F_withExtraBC
    deformation_withExtraBC = GridFunction(fes)
    deformation_withExtraBC.Set((x,y))
    deformation_withExtraBC += u_withExtraBC
    
    # Draw(u_withoutExtraBC, deformation=True)
    # Draw(GEL.W(F_withoutExtraBC), mesh, deformation=u_withoutExtraBC, max=0.14)
    # print(Integrate(GEL.W(F_withoutExtraBC), mesh))
    
    # Draw(u_withExtraBC, deformation=True)
    # Draw(GEL.W(F_withExtraBC), mesh, deformation=u_withExtraBC, max=0.14)
    # print(Integrate(GEL.W(F_withExtraBC), mesh))
    
    # Draw(F_withoutExtraBC - F_withExtraBC, mesh)
    
    aux_csv=[]
    aux_csv.append(round(delta,3))
    
    deformation_difference = deformation_withExtraBC - deformation_withoutExtraBC
    L2_distance = sqrt(Integrate(InnerProduct(deformation_difference, deformation_difference), mesh, order=10))
    gradDifference = F_withExtraBC - F_withoutExtraBC
    H1_0_distance = sqrt(Integrate(Trace(gradDifference.trans*gradDifference), mesh, order=10))
    H1_distance = L2_distance + H1_0_distance
    H1_norm_withoutExtraBC = sqrt(Integrate(InnerProduct(deformation_withoutExtraBC, deformation_withoutExtraBC),mesh, order=10)) \
        + sqrt(Integrate(Trace(C_withoutExtraBC),mesh, order=10))
    H1_relative_error = H1_distance/H1_norm_withoutExtraBC
     
    aux_csv.append(round(H1_relative_error*100,4))
    
    ### Energies
    lambdaUniaxial = GEL.lambdaUniaxial
    lambdaEquiBiaxial = GEL.lambdaEquiBiaxial
    wUniaxial = GEL.wUniaxial
    wEquiBiaxial = GEL.wEquiBiaxial
    gravitationalEnergy = + rho*9.8*L*(d**2)/2*(delta*(lambdaUniaxial-1) + (1-delta)*(lambdaEquiBiaxial-1))
    theoreticalEnergy = gravitationalEnergy + (((L*delta)*d*wUniaxial + (L*(1-delta))*d*wEquiBiaxial))
    aux_csv.append(theoreticalEnergy)
    aux_csv.append(gravitationalEnergy)
    
    
    Energy_withExtraBC = Integrate(GEL.W(F_withExtraBC) - rho*InnerProduct(gravity, u_withExtraBC), mesh, order=10)
    aux_csv.append(Energy_withExtraBC)
    Energy_withoutExtraBC = Integrate(GEL.W(F_withoutExtraBC) - rho*InnerProduct(gravity, u_withoutExtraBC), mesh, order=10)
    aux_csv.append(Energy_withoutExtraBC)
    Relative_error = (Energy_withExtraBC - Energy_withoutExtraBC)/Energy_withoutExtraBC
    aux_csv.append(Relative_error*100)
    
    ### Deformed lengths    
    Deformed_length_withExtraBC = (deformation_withExtraBC(mesh(L/2,d/2)))[0] - (deformation_withExtraBC(mesh(-L/2,d/2)))[0]
    aux_csv.append(Deformed_length_withExtraBC)
    Deformed_length_withoutExtraBC = (deformation_withoutExtraBC(mesh(L/2,d/2)))[0] - (deformation_withoutExtraBC(mesh(-L/2,d/2)))[0]
    aux_csv.append(Deformed_length_withoutExtraBC)
    Relative_error = (Deformed_length_withExtraBC - Deformed_length_withoutExtraBC)/Deformed_length_withoutExtraBC
    aux_csv.append(Relative_error*100)
    
    ###    
    trial_points = [(0,d), (L/2, d)]
    deformed_thickness_withExtraBC = np.zeros(2)
    deformed_thickness_withoutExtraBC = np.zeros(2)
    eigenvalue1_withExtraBC = np.zeros(2)
    eigenvalue1_withoutExtraBC = np.zeros(2)
    eigenvalue2_withExtraBC = np.zeros(2)
    eigenvalue2_withoutExtraBC = np.zeros(2)
    
    counter=0;
    for (trial_x, trial_y) in trial_points:
        mesh_pnt=mesh(trial_x,trial_y)
        deformed_thickness_withExtraBC[counter] = (deformation_withExtraBC(mesh_pnt))[1]
        deformed_thickness_withoutExtraBC[counter] = (deformation_withoutExtraBC(mesh_pnt))[1]
        eigenvalues_aux = np.linalg.eigvalsh(np.reshape(C_withExtraBC(mesh_pnt),(2,2)))
        eigenvalue1_withExtraBC[counter]=sqrt(eigenvalues_aux[0])
        eigenvalue2_withExtraBC[counter]=sqrt(eigenvalues_aux[1])
        eigenvalues_aux = np.linalg.eigvalsh(np.reshape(C_withoutExtraBC(mesh_pnt),(2,2)))
        eigenvalue1_withoutExtraBC[counter]=sqrt(eigenvalues_aux[0])
        eigenvalue2_withoutExtraBC[counter]=sqrt(eigenvalues_aux[1])
        counter+=1;
    
    for i in [0,1]:
        value1=deformed_thickness_withExtraBC[i]
        value2=deformed_thickness_withoutExtraBC[i]
        aux_csv += [round(value1,2), round(value2,2), (value1-value2)/value2*100]

    aux_ell2_error = 0;
    aux_norm_noG = 0;
    for i in [0,1]:
        value1g = eigenvalue1_withExtraBC[i]; value2g=  eigenvalue2_withExtraBC[i];
        aux_csv += [value1g, value2g]
        value1noG=eigenvalue1_withoutExtraBC[i]; value2noG = eigenvalue2_withoutExtraBC[i]
        aux_csv += [value1noG, value2noG]
        
        aux_norm_noG += value1noG**2 + value2noG**2
        aux_ell2_error += (value1g - value1noG)**2 + (value2g-value2noG)**2
    
    Relative_error = sqrt(aux_ell2_error)/sqrt(aux_norm_noG)
    aux_csv.append(Relative_error*100)

    csv_preliminary.append(aux_csv)

f = open ("result_comparison2D/result_comparison2D-d={:.2f}.csv".format(d), 'w')
with f:
    writer = csv.writer(f)
    writer.writerows(csv_preliminary)
f.close()
    
