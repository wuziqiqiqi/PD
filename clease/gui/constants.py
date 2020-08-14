FOREGROUND_TEXT_COLOR = '#FFFFFF'
INACTIVE_TEXT_COLOR = '#333333'
BACKGROUND_COLOR = '#000000'
ECI_GRAPH_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
MC_MEAN_CURVE_COLOR = '#8da0cb'

# Modes for concentration in MC
SYSTEMS_FROM_DB = 0
CONC_PER_BASIS = 1

# Messages for job exec page
JOB_EXEC_MSG = {
    'db_id_error': 'Could not convert the passed DB ids to int',
    'jobs_finished': 'All jobs are successfully finished'
}

# Main MC PAGE
SCREEN_TRANSLATIONS = {'Main': 'MCMainPage', 'Canonical': 'MC', 'Metadynamics': 'MetaDynPage'}
# Mapping between MC types and MC page
MC_TYPE_TO_PAGE = {'Canonical': 'MC', 'Meta dynamics': 'MetaDynPage'}

# Meta dynamics messages
META_DYN_MSG = {
    'settings_is_none': 'Apply settings prior to running MC',
    'more_than_one_basis': 'Currently the GUI only supports running '
                           'metadynamics when all elements can occupy '
                           'all sites. Sorry.',
    'mc_cell_is_template': 'MC cell is already set. Seems like a MC '
                           'calculation is running...',
    'no_eci': 'Cannot load ECI from',
    'unkown_var': 'Unknown variable:',
    'unknown_ens': 'Unkown ensemble:',
    'var_editor_not_launched': 'Check the variable editor before running',
    'launch_ens_editor': 'Check the ensemble editor before running',
    'abort_mc': 'MC calculation was aborted'
}
