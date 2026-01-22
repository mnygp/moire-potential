import taskblaster as tb


@tb.workflow
class Workflow:
    @tb.task
    def get_struct_dir(self):
        return tb.node("get_dirs")

    @tb.dynamical_workflow_generator({"results": "*/*"})
    def gen_wfs(self):
        return tb.node("generate_wfs", paths=self.get_struct_dir)


def workflow(runner):
    runner.run_workflow(Workflow())
