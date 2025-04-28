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

    @tb.dynamical_workflow_generator({'results': '*/*'})
    def generate_wfs(self):
        return tb.node('generate_wfs_task',
                       result_dict=self.create_atoms_task)

    @tb.task
    def write_csv_task(self):
        return tb.node('write_results_to_csv',
                       results_dict=self.generate_wfs.results)


def workflow(runner):
    runner.run_workflow(Workflow(folder_name='1.05_3027'))
