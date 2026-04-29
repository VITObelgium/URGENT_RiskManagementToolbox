import numpy as np


def GetShapeFunctionsHex8(gp):
    ksi, eta, zeta = gp[0], gp[1], gp[2]
    
    # Pre-defined node coordinates as numpy arrays
    ksiJ = np.array([-1, 1, 1, -1, -1, 1, 1, -1])
    etaJ = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    zetaJ = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    
    # Vectorized computation
    ksi_term = 1.0 + ksi * ksiJ
    eta_term = 1.0 + eta * etaJ
    zeta_term = 1.0 + zeta * zetaJ
    
    N = 0.125 * ksi_term * eta_term * zeta_term
    Nksi = 0.125 * ksiJ * eta_term * zeta_term
    Neta = 0.125 * etaJ * ksi_term * zeta_term
    Nzeta = 0.125 * zetaJ * ksi_term * eta_term
    
    return N, Nksi, Neta, Nzeta 
    
# module global variables
GP3 = 3.0**-0.5
GAUSS_HEX8 = np.array([
    [-GP3, -GP3, -GP3],
    [ GP3, -GP3, -GP3],
    [ GP3,  GP3, -GP3],
    [-GP3,  GP3, -GP3],
    [-GP3, -GP3,  GP3],
    [ GP3, -GP3,  GP3],
    [ GP3,  GP3,  GP3],
    [-GP3,  GP3,  GP3],
], dtype=float)

W_HEX8 = np.ones(8, dtype=float)

# Precompute shape functions at all 8 Gauss points
N_G     = np.zeros((8, 8))
Nksi_G  = np.zeros((8, 8))
Neta_G  = np.zeros((8, 8))
Nzeta_G = np.zeros((8, 8))

for igp, gp in enumerate(GAUSS_HEX8):
    N_G[igp], Nksi_G[igp], Neta_G[igp], Nzeta_G[igp] = GetShapeFunctionsHex8(gp)

###############################################################################
# FUNCTIONS
###############################################################################   
def GetElementStiffness(xe,ye,ze,Young,Poisson,ndof):
    '''
    GetElementStiffness performs the stiffness calculations for the solid mechanics stiffness
    matrix, the poro-elastic coupling matrix, and the solids internal force vector (body
    forces due to gravity
    '''
    nlink = len(xe)
    size = nlink*ndof
    ID = np.asarray([1.,1.,1.,0.,0.,0.])
    
    # Initialize element stiffness matrices and force vector
    ke = np.zeros(shape=(size,size))
    
    # Set up stardard linear elasticity stiffness matrix
    CMAT = GetElasticityMatrix(Young,Poisson)
    CID = CMAT@ID
    
    for igp, W in enumerate(W_HEX8):
        N     = N_G[igp]
        NJksi = Nksi_G[igp]
        NJeta = Neta_G[igp]
        NJzeta= Nzeta_G[igp]
        
        NJdxyz,jcob = GetShapeDerivativesGlobal(xe,ye,ze,NJksi,NJeta,NJzeta)
        BM = GetBMatrix(NJdxyz,ndof,nlink)
        factor = W*jcob
        ke  += factor*np.dot(BM.T,np.dot(CMAT,BM))
    return ke
    
def GetElementForces(xe,ye,ze,dP_ele,dT_ele,Young,Poisson,biot,cte,rho,ndof,grav=(0.,0.,9.81)):
    '''
    Gets RHS forcing for each element
    '''
    nlink = len(xe)
    size = nlink*ndof
    ID = np.asarray([1.,1.,1.,0.,0.,0.])
    fpe = np.zeros(size)
    fte = np.zeros(size)
    fbe = np.zeros(size)
    
    # Set up stardard linear elasticity stiffness matrix
    CMAT = GetElasticityMatrix(Young,Poisson)
    CID = CMAT@ID
    
    for igp, W in enumerate(W_HEX8):
        N     = N_G[igp]
        NF = np.zeros((3,size))
        NF[0,0::3] = N
        NF[1,1::3] = N
        NF[2,2::3] = N
        
        NJksi = Nksi_G[igp]
        NJeta = Neta_G[igp]
        NJzeta= Nzeta_G[igp]
        NJdxyz,jcob = GetShapeDerivativesGlobal(xe,ye,ze,NJksi,NJeta,NJzeta)
        BM = GetBMatrix(NJdxyz,ndof,nlink)
        factor = W*jcob
        fpe += factor*(dP_ele*biot*np.dot(BM.T,ID))
        fte += factor*(dT_ele*cte*np.dot(BM.T,CID))
        fbe += factor*rho*(NF.T @ grav)
        
    return fpe,fte,fbe

def GetElasticityMatrix(E,v):
    Et = E/((1+v)*(1-2.*v))
    G = E/(2.*(1+v))
    CMAT = [[Et*(1-v), Et*v,     Et*v,     0., 0., 0.],
            [Et*v,     Et*(1-v), Et*v,     0., 0., 0.],
            [Et*v,     Et*v,     Et*(1-v), 0., 0., 0.],
            [0.,       0.,       0.,       G,  0., 0.],
            [0.,       0.,       0.,       0., G,  0.],
            [0.,       0.,       0.,       0., 0., G]]
    return CMAT
    
def GetShapeDerivativesGlobal(xe,ye,ze,NJksi,NJeta,NJzeta):
    '''
    GetShapeDerivativesGlobal: computes the shape function derivatives in the global
    coordinates. This routine essentially performs a coordinate transformation
    from the local (ksi,eta) coordinate system to the global (x,y) coordinate system
    
    jcob = jacobian of transformation - this is the area of the element associated with
    the current Gauss point.
    
    For a reference, see Chapter 3 of Hughes' text.
    '''
    Xksi = np.dot(NJksi,xe)
    Yksi = np.dot(NJksi,ye)
    Zksi = np.dot(NJksi,ze)
    Xeta = np.dot(NJeta,xe)
    Yeta = np.dot(NJeta,ye)
    Zeta = np.dot(NJeta,ze)
    Xzeta = np.dot(NJzeta,xe)
    Yzeta = np.dot(NJzeta,ye)
    Zzeta = np.dot(NJzeta,ze)
    
    cof11 = Yeta*Zzeta-Zeta*Yzeta
    cof12 = -Xeta*Zzeta+Zeta*Xzeta
    cof13 = Xeta*Yzeta-Yeta*Xzeta
    cof21 = -Yksi*Zzeta+Zksi*Yzeta
    cof22 = Xksi*Zzeta-Zksi*Xzeta
    cof23 = -Xksi*Yzeta+Yksi*Xzeta
    cof31 = Yksi*Zeta-Zksi*Yeta
    cof32 = -Xksi*Zeta+Zksi*Xeta
    cof33 = Xksi*Yeta-Yksi*Xeta
    
    jcob = 1.0*(Xksi*(Yeta*Zzeta-Yzeta*Zeta) - Xeta*(Yksi*Zzeta-Yzeta*Zksi) + Xzeta*(Yksi*Zeta-Yeta*Zksi))
    #Jmat = [[cof11, cof12, cof13], [cof21, cof22, cof23], [cof31, cof32, cof33]]*1/jcob
    
    NJdx = (np.dot(NJksi,cof11)+np.dot(NJeta,cof12)+np.dot(NJzeta,cof13))/jcob
    NJdy = (np.dot(NJksi,cof21)+np.dot(NJeta,cof22)+np.dot(NJzeta,cof23))/jcob
    NJdz = (np.dot(NJksi,cof31)+np.dot(NJeta,cof32)+np.dot(NJzeta,cof33))/jcob
    NJdxyz = [NJdx, NJdy, NJdz]
    
    # Jacobian check
    if jcob<0.0:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('Error: Negative Jacobian! Either your element connectivity definition')
        print('       is bad or an element has inverted!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        stop
    
    return  NJdxyz,jcob   
    
def GetBMatrix(NJdxyz, ndof, nlink):
    """
    Build 3D B-matrix for strain ordering [11, 22, 33, 23, 13, 12].

    NJdxyz: (3, nlink) with rows [dN/dx, dN/dy, dN/dz]
    ndof:   should be 3
    nlink:  number of nodes per element (e.g. 8 for Hex8)
    """
    assert ndof == 3, "This optimized B only supports ndof=3."

    BM = np.zeros((6, ndof * nlink), dtype=float)
    NJdxyz = np.asarray(NJdxyz, dtype=float)
    
    dNx = NJdxyz[0, :]   # shape (nlink,)
    dNy = NJdxyz[1, :]
    dNz = NJdxyz[2, :]

    # Column indices of the x DOFs for each node
    # Node i has DOFs at [3*i, 3*i+1, 3*i+2]
    cols = np.arange(nlink) * ndof

    # Normal strains: ε11, ε22, ε33
    BM[0, cols + 0] = dNx   # ε11 from ux,x
    BM[1, cols + 1] = dNy   # ε22 from uy,y
    BM[2, cols + 2] = dNz   # ε33 from uz,z

    # Shear strains in your order: [23, 13, 12]

    # ε23 = 0.5 (∂uy/∂z + ∂uz/∂y)  → Voigt component 23
    BM[3, cols + 1] = dNz   # uy,z
    BM[3, cols + 2] = dNy   # uz,y

    # ε13 = 0.5 (∂ux/∂z + ∂uz/∂x)  → Voigt component 13
    BM[4, cols + 0] = dNz   # ux,z
    BM[4, cols + 2] = dNx   # uz,x

    # ε12 = 0.5 (∂ux/∂y + ∂uy/∂x)  → Voigt component 12
    BM[5, cols + 0] = dNy   # ux,y
    BM[5, cols + 1] = dNx   # uy,x

    return BM 

    