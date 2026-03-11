import numpy as np
import taskblaster as tb


@tb.workflow
class ref_Wf:
    shift_1 = tb.var()
    shift_2 = tb.var()

    @tb.dynamical_workflow_generator({"results": "*/*", "gaps": "*/gap_calc"})
    def generated_wfs(self):
        return tb.node("wfs", inputs={"shift 1": self.shift_1, "shift 2": self.shift_2})

    @tb.task
    def write_csv(self):
        return tb.node(
            "write_results_to_csv",
            results_dict=self.generated_wfs.gaps,
            csv_name="opt_z_gaps_005_variable_screen.csv",
        )


shift_arr = list(np.linspace(0, 1, 20, endpoint=False))


def workflow(runner):
    runner.run_workflow(ref_Wf(shift_1=shift_arr, shift_2=shift_arr))
