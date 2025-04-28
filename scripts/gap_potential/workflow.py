import taskblaster as tb


@tb.workflow
class Workflow:
    folder_name = tb.var()

    @tb.task
    def find_file_task(self):
        return tb.node('get_path', structure_name=self.folder_name)

    @tb.task
    def create_atoms_task(self):
        return tb.node('create_atoms_list', file=self.find_file_task)

    @tb.dynamical_workflow_generator(
            {'results': '*/*',
             'gap_results': '*/calc_gap_task',
             'extra_dist_atoms': '*/calc_extra_gap_task'})
    def generate_wfs(self):
        return tb.node('generate_wfs_task',
                       result_dict=self.create_atoms_task)

    @tb.task
    def write_csv_task(self):
        return tb.node('write_results_to_csv',
                       results_dict=self.generate_wfs.gap_results,
                       csv_name="gap_results.csv")

    @tb.task
    def write_extra_space_csv_task(self):
        return tb.node('write_results_to_csv',
                       results_dict=self.generate_wfs.extra_dist_atoms,
                       csv_name="gap_extra_space_results.csv")


def workflow(runner):
    runner.run_workflow(Workflow(folder_name='1.05_3027'))
