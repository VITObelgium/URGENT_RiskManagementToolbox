import numpy as np


def get_boundary_nodelists(points):
    max_x = points[:,0].max()
    min_x = points[:,0].min()
    max_y = points[:,1].max()
    min_y = points[:,1].min()
    max_z = points[:,2].max()
    min_z = points[:,2].min()
    
    node_dict = {}
    node_dict['x_max'] = np.where(points[:,0]==max_x)[0]
    node_dict['x_min'] = np.where(points[:,0]==min_x)[0]
    node_dict['y_max'] = np.where(points[:,1]==max_y)[0]
    node_dict['y_min'] = np.where(points[:,1]==min_y)[0]
    node_dict['z_max'] = np.where(points[:,2]==max_z)[0]
    node_dict['z_min'] = np.where(points[:,2]==min_z)[0]
    
    return node_dict
    
def BC_dict(bc_type='dirchlet',variable=0,nodelist=[0],values=[0]):
    out_dict = {}
    out_dict['bc_type'] = bc_type
    out_dict['variable'] = variable
    out_dict['nodelist'] = nodelist
    out_dict['values'] = values
    
    return out_dict
    
def SetDirichletBCs(BC_list,time_step_number,ndof):
    dof_ids = []
    dof_vals = []
    for bc_dict in BC_list:
        if bc_dict['bc_type']=='dirchlet':
            bc_dict['nodelist'] = np.asarray(bc_dict['nodelist'],dtype=int)
            id_list = bc_dict['nodelist']*ndof+bc_dict['variable']
            if len(bc_dict['values'])==1:
                val = bc_dict['values'][0]
            else:
                val = bc_dict['values'][time_step_number]
            for nid in id_list:
                dof_ids.append(nid)
                dof_vals.append(val)

    return dof_ids, dof_vals 
 