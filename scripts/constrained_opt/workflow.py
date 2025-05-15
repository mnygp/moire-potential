import taskblaster as tb


@tb.workflow
class Workflow:
    x_points = tb.var()
    y_points = tb.var()

    @tb.task
    def tuple_list_task(self):
        return tb.node('create_tuple_list', x=self.x_points, y=self.y_points)

    @tb.dynamical_workflow_generator({'results': '*/*',
                                      'gap_results': '*/return_dict_task'})
    def generate_wfs(self):
        return tb.node('generate_wfs_task',
                       coords=self.tuple_list_task)

    @tb.task
    def write_csv_task(self):
        return tb.node('write_results_to_csv',
                       result_dict=self.generate_wfs.gap_results)


def workflow(runner):
    runner.run_workflow(Workflow(x_points=15, y_points=15))  # type:ignore
