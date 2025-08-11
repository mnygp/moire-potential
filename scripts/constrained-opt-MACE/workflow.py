import taskblaster as tb

path = '/structures/1.11_2946/structure_ml.json'


@tb.workflow
class Workflow:
    struct_path = tb.var()

    @tb.task
    def get_root_task(self):
        return tb.node('get_root_path', directory='moire-potential')

    @tb.task
    def get_shifts_task(self):
        return tb.node('get_shifts', structure_path=self.struct_path,
                       root=self.get_root_task)

    @tb.dynamical_workflow_generator({'results': '*/*',
                                      'gap_results': '*/return_dict_task'})
    def generate_wfs(self):
        return tb.node('generate_wfs_task',
                       input=self.get_shifts_task)  # type:ignore

    @tb.task
    def write_csv_task(self):
        return tb.node('write_results_to_csv',
                       results_dict=self.generate_wfs.gap_results,
                       csv_name='results.csv')


def workflow(runner):
    runner.run_workflow(Workflow(struct_path=path))  # type:ignore
