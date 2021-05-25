# Connectome manipulation function
#
# Definition of apply(edges_table, nodes, ...):
# - The first three parameters are always: edges_table, nodes, aux_dict
# - aux_dict contains information about data splits; may also be used to pass global information from one split iteration to another
# - Other parameters may be added (optional)
# - Returns a manipulated edged_table

from helper_functions import get_gsyn_sum_per_conn, rescale_gsyn_per_conn
from model_building import model_building
import pickle
import logging
import numpy as np

""" Add certain amount of synapses to existing connections, optionally keeping sum of g_syns per connection constant """
def apply(edges_table, nodes, aux_dict, amount, sel_src=None, sel_dest=None, rescale_gsyn=False, method='duplicate', props_model_file=None):
    
    # Input checks
    logging.log_assert(isinstance(amount, dict) and 'type' in amount.keys() and 'value' in amount.keys(), 'Amount must be specified as dict with "type"/"value" entries!')
    logging.log_assert(amount['type'] in ['pct', 'pct_per_conn', 'rnd_per_conn', 'minval_per_conn'], f'Synapse amount type "{amount["type"]}" not supported (must be "pct", "pct_per_conn", "rnd_per_conn", or "minval_per_conn")!')
    # amount: pct ... Overall increase by percentage of total number of existing synapses
    #         pct_per_conn ... Increase each connection by percentage of existing synapses
    #         rnd_per_conn ... Increase each connection by random number of synapses within given range
    #         minval_per_conn ... Increase until minimum target value of synapses per connection is reached
    logging.log_assert(method in ['duplicate', 'derive'], f'Synapse addition method "{method}" not supported (must be "duplicate" or "derive")!')
    # method: duplicate ... Duplicate existing synapses
    #         derive ... Derive from existing synapses (duplicate and randomize certain properties based on a model)
    #         [NYI] randomize ... Fully randomized parameterization, based on model-based distributions
    #         [NYI] load ... Load from external connectome, e.g. structural connectome
    
    # Load synapse properties model (if required)
    if method == 'derive':
        # example: props_model_file = '../working_dir/model_building/circuit-build-S1_v1/model/ConnPropsPerMType.pickle'
        logging.log_assert(not props_model_file is None, f'Synaptic properties model file required for "{method}" method!')
        logging.log_assert(os.path.exists(props_model_file), 'Synaptic properties model file not found!')
        logging.info(f'Loading synaptic properties model from {props_model_file}')
        with open(props_model_file, 'rb') as f:
            props_model_dict = pickle.load(f)
        props_model = model_building.get_model(props_model_dict['model'], props_model_dict['model_inputs'], props_model_dict['model_params'])
    
    # Determine number of synapses to be added
    gids_src = nodes[0].ids(sel_src)
    gids_dest = nodes[1].ids(sel_dest)
    syn_sel_idx = np.logical_and(np.isin(edges_table['@source_node'], gids_src), np.isin(edges_table['@target_node'], gids_dest))
    num_syn = np.sum(syn_sel_idx)
    
    conns, syn_conn_idx, num_syn_per_conn = np.unique(edges_table[syn_sel_idx][['@source_node', '@target_node']], axis=0, return_inverse=True, return_counts=True)
    
    if amount['type'] == 'pct': # Overall increase by percentage of total number of existing synapses between src and dest
        logging.log_assert(amount['value'] >= 0.0, f'Amount value for type "{amount["type"]}" out of range!')
        num_add = np.round(amount['value'] * num_syn / 100).astype(int)
    elif amount['type'] == 'pct_per_conn': # Increase each connection by percentage of existing synapses
        logging.log_assert(amount['value'] >= 0.0, f'Amount value for type "{amount["type"]}" out of range!')
        num_add = np.round(amount['value'] * num_syn_per_conn / 100).astype(int)
    elif amount['type'] == 'rnd_per_conn': # Increase each connection by random number of synapses within given range
        logging.log_assert(not np.isscalar(amount['value']) and len(amount['value']) == 2, f'Amount value for type "{amount["type"]}" needs to be a range with two elements!')
        logging.log_assert(amount['value'][0] >= 0 and amount['value'][1] >= amount['value'][0], f'Amount value range for type "{amount["type"]}" out of range!')
        num_add = np.random.randint(low=np.round(amount['value'][0]).astype(int), high=np.round(amount['value'][1]).astype(int)+1, size=conns.shape[0])
    elif amount['type'] == 'minval_per_conn': # Increase until minimum target value of synapses per connection is reached
        logging.log_assert(amount['value'] >= 0, f'Amount value for type "{amount["type"]}" out of range!')
        num_add = np.maximum(np.round(amount['value']).astype(int) - num_syn_per_conn, 0)
    else:
        logging.log_assert(False, f'Amount type "{amount["type"]}" not supported!')
    
    logging.info(f'Adding {np.sum(num_add)} synapses to {edges_table.shape[0]} total synapses (sel_src={sel_src}, sel_dest={sel_dest}, amount[type]={amount["type"]}, amount[value]={amount["value"]}, method={method}, rescale_gsyn={rescale_gsyn})')
    
    # Add num_add synapses between src and dest nodes
    if np.sum(num_add) > 0:
        if rescale_gsyn:
            # Determine connection strength (sum of g_syns per connection) BEFORE adding synapses
            gsyn_table = get_gsyn_sum_per_conn(edges_table, gids_src, gids_dest)
        
        if method == 'duplicate' or method == 'derive': # Duplicate or derive from existing synapses
            if np.isscalar(num_add): # Overall number of synapses to add
                sel_dupl = np.random.choice(np.where(syn_sel_idx)[0], num_add) # Random sampling from existing synapses with replacement
            else: # Number of synapses per connection to add [requires syn_conn_idx for mapping between connections and synapses]
                sel_dupl = np.full(np.sum(num_add), -1) # Empty selection
                dupl_idx = np.hstack((0, np.cumsum(num_add))) # Index vector where to save selected indices
                for cidx, num in enumerate(num_add):
                    if num == 0: # Nothing to add for this connection
                        continue
                    conn_sel = np.zeros_like(syn_sel_idx, dtype=bool) # Empty selection mask
                    conn_sel[syn_sel_idx==True] = syn_conn_idx == cidx # Sub-select (mask) synapses belonging to given connection from all selected synapses
                    sel_dupl[dupl_idx[cidx]:dupl_idx[cidx+1]] = np.random.choice(np.where(conn_sel)[0], num) # Random sampling from existing synapses per connection with replacement
            new_edges = edges_table.iloc[sel_dupl].copy()
            
            if method == 'derive': # Derive from existing synapses (duplicate and randomize certain properties based on a model)
                src_mtypes = nodes[0].get(new_edges['@source_node'].to_numpy(), properties='mtype')
                tgt_mtypes = nodes[1].get(new_edges['@target_node'].to_numpy(), properties='mtype')
                
                # Access props. statistics (example): props_model('u_syn', src_mtypes.iloc[0], tgt_mtypes.iloc[0], 'mean')
                model_props = sorted(props_model(None, None, None, None)) # List of properties in the model
                model_props = sorted(np.intersect1d(edges_table.keys(), model_props)) # Select model props which are part of the edges table
                
                # Randomize within connections
                new_conns, new_syn_conn_idx = np.unique(new_edges[['@source_node', '@target_node']], axis=0, return_inverse=True)
                for new_idx in range(new_conns.shape[0]):
                    # Estimate current property means per connection
                    conn_idx = np.where(np.all(conns == new_conns[new_idx], 1))[0][0]
                    prop_means = [np.mean(edges_table.loc[syn_conn_idx == conn_idx, p]) for p in model_props]
                    
                    # Get within-connection statistics from model
                    prop_stds = [props_model(p, src_mtypes[new_syn_conn_idx == new_idx].iloc[0], tgt_mtypes[new_syn_conn_idx == new_idx].iloc[0], 'std-within') for p in model_props]
                    prop_mins = [props_model(p, src_mtypes[new_syn_conn_idx == new_idx].iloc[0], tgt_mtypes[new_syn_conn_idx == new_idx].iloc[0], 'min') for p in model_props]
                    prop_maxs = [props_model(p, src_mtypes[new_syn_conn_idx == new_idx].iloc[0], tgt_mtypes[new_syn_conn_idx == new_idx].iloc[0], 'max') for p in model_props]
                    
                    # Randomize new property values per connection
                    # TODO: Take actual distributions (and data types) into account!
                    new_vals = np.random.randn(np.sum(new_syn_conn_idx == new_idx), len(model_props)) * prop_stds + prop_means
                    new_vals = np.maximum(new_vals, prop_mins)
                    new_vals = np.minimum(new_vals, prop_maxs)
                    new_edges.loc[new_syn_conn_idx == new_idx, model_props] = new_vals
                new_edges = new_edges.astype(edges_table.dtypes)
                
                logging.log_assert(False, f'Method "{method}" not yet implemented!')
        else:
            logging.log_assert(False, f'Method "{method}" not supported!')
        
        # Add new synapses to table, re-sort, and assign new index
        edges_table = edges_table.append(new_edges)
        edges_table.sort_values(['@target_node', '@source_node'], inplace=True)    
        edges_table.reset_index(inplace=True, drop=True) # [No index offset required when merging files in block-based processing]
        
        if rescale_gsyn:
            # Determine connection strength (sum of g_syns per connection) AFTER adding synapses ...
            gsyn_table_manip = get_gsyn_sum_per_conn(edges_table, gids_src, gids_dest)
            
            # ... and rescale g_syn so that the sum of g_syns per connections BEFORE and AFTER the manipulation is kept the same
            rescale_gsyn_per_conn(edges_table, gids_src, gids_dest, gsyn_table, gsyn_table_manip)        
    else:
        logging.warning(f'Nothing to add!')
    
    return edges_table
