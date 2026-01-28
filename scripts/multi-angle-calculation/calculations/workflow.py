import taskblaster as tb


@tb.workflow
class Workflow:
    @tb.task
    def get_struct_dir(self):
        return tb.node("get_dirs")

    @tb.task
    def calc_ref_energy(self):
        return tb.node("strain_ref")

    @tb.dynamical_workflow_generator(
        {
            "results": "*/*",
            "geometric_results": "*/geometry",
            "z_opt_gap": "*/corrected_opt_gap",
            "z_param_gap": "*/corrected_param_gap",
        }
    )
    def gen_wfs(self):
        return tb.node(
            "generate_wfs", paths=self.get_struct_dir, strain_ref_E=self.calc_ref_energy
        )


def workflow(runner):
    runner.run_workflow(Workflow())
