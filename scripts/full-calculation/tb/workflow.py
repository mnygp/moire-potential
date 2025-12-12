import taskblaster as tb

path = '/structures/MoS2-WSe2-MatterSim/1.11_2946/structure_ml.json'

@tb.workflow
class Workflow:
    struct_path = tb.var()

    @tb.task
    def get_atoms_path(self):
        return tb.node('get_root_path', directory='moire-potential',
                       path_str=self.struct_path)

    @tb.task
    def geometric_parameters(self):
        return tb.node('get_geometry', atom_path=self.get_atoms_path)


    @tb.task
    def write_csv_Mo_pos(self):
        return tb.node('write_Mo_pos_to_csv', results_dict=self.geometric_parameters,
                       csv_name='Mo_positions.csv')


    # Fixed cell size and fix TM position
    @tb.dynamical_workflow_generator({'results': '*/*',
                                      'results_dict': '*/return_dict'})
    def fixed_cell_fixed_TM(self):
        return tb.node('generate_wfs_task',
                       input=self.geometric_parameters,
                       fixed_cell=True,
                       fixed_atom=True,
                       structure_path=self.get_atoms_path)

    @tb.task
    def write_csv_fixed_cell_fixed_TM(self):
        return tb.node('write_results_to_csv',
                       results_dict=self.fixed_cell_fixed_TM.results_dict,
                       csv_name='results_fixed_cell_fixed_TM.csv')

    tb.task
    def write_kpt_csv_fixed_cell_fixed_TM(self):
        return tb.node('write_kpts_to_csv',
                       results_dict=self.fixed_cell_fixed_TM.results_dict,
                       csv_name='ktps_fixed_cell_fixed_TM_scissors.csv')
    
    """
    # Fixed cell size and variable TM position
    @tb.dynamical_workflow_generator({'results': '*/*',
                                      'results_dict': '*/return_dict'})
    def fixed_cell_variable_TM(self):
        return tb.node('generate_wfs_task',
                       input=self.geometric_parameters,
                       fixed_cell=True,
                       fixed_atom=False,
                       structure_path=self.get_atoms_path)

    @tb.task
    def write_csv_fixed_cell_variable_TM(self):
        return tb.node('write_results_to_csv',
                       results_dict=self.fixed_cell_variable_TM.results_dict,
                       csv_name='results_fixed_cell_variable_TM.csv')

    tb.task
    def write_kpt_csv_fixed_cell_variable_TM(self):
        return tb.node('write_kpts_to_csv',
                       results_dict=self.fixed_cell_variable_TM.results_dict,
                       csv_name='ktps_fixed_cell_variable_TM.csv')
    """


def workflow(runner):
    runner.run_workflow(Workflow(struct_path=path))  # type:ignore
