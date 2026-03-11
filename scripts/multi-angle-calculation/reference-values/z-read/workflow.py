import numpy as np
import taskblaster as tb


@tb.workflow
class ref_Wf:
    z_values = tb.var()
    shift_1 = tb.var()
    shift_2 = tb.var()

    @tb.dynamical_workflow_generator({"results": "*/*", "gaps": "*/gap_calc"})
    def generated_wfs(self):
        return tb.node(
            "wfs",
            inputs={
                "z": self.z_values,
                "shift 1": self.shift_1,
                "shift 2": self.shift_2,
            },
        )

    @tb.task
    def write_csv(self):
        return tb.node('write_results_to_csv', results_dict=self.generated_wfs.gaps, csv_name='optimized_z_gaps.csv')


z_arr = list(np.linspace(5.9, 7, 15, endpoint=True))
shift_arr = list(np.linspace(0, 1, 15, endpoint=False))


def workflow(runner):
    runner.run_workflow(ref_Wf(z_values=z_arr, shift_1=shift_arr, shift_2=shift_arr))
