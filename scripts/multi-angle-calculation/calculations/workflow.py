import taskblaster as tb


@tb.workflow
class plotting_Workflow:
    atoms_dict = tb.var()
    geometry_dict = tb.var()
    gap_opt_dict = tb.var()
    strain_corr_dict = tb.var()
    z_diff_dict = tb.var()
    fd_opt = tb.var()
    LCAO_dict = tb.var()

    @tb.task
    def z(self):
        return tb.node("plot_z", input=self.geometry_dict, atoms=self.atoms_dict)

    @tb.task
    def strain(self):
        return tb.node("plot_strain", input=self.geometry_dict, atoms=self.atoms_dict)

    @tb.task
    def gap_opt(self):
        return tb.node("plot_gap", input=self.gap_opt_dict, atoms=self.atoms_dict)

    @tb.task
    def wavefunction(self):
        return tb.node("plot_wavefunction", input=self.fd_opt, atoms=self.atoms_dict)

    @tb.task
    def strain_correction(self):
        return tb.node(
            "plot_strain_correction", input=self.strain_corr_dict, atoms=self.atoms_dict
        )

    @tb.task
    def z_diff(self):
        return tb.node("plot_z_diff", input=self.z_diff_dict)

    @tb.task
    def energy(self):
        return tb.node("plot_energy", fd_input=self.fd_opt, gap_input=self.gap_opt_dict)

    @tb.task
    def plot_LCAO(self):
        return tb.node("plot_local_gap", input=self.LCAO_dict, atoms=self.atoms_dict)


@tb.workflow
class Workflow:
    @tb.task
    def get_struct_dir(self):
        return tb.node("get_dirs")

    @tb.dynamical_workflow_generator(
        {
            "results": "*/*",
            "atoms": "*/get_atoms",
            "geometric_results": "*/geometry",
            "strain_correction": "*/strain_correction",
            "z_opt_gap": "*/corrected_opt_gap",
            "z_param_gap": "*/corrected_param_gap",
            "finite_diff_opt": "*/fd_opt",
            "z_difference": "*/compare_z",
            "LCAO": "*/LCAO_projection",
        }
    )
    def gen_wfs(self):
        return tb.node("generate_wfs", paths=self.get_struct_dir)

    @tb.subworkflow
    def plotting(self):
        return plotting_Workflow(
            atoms_dict=self.gen_wfs.atoms,
            geometry_dict=self.gen_wfs.geometric_results,
            strain_corr_dict=self.gen_wfs.strain_correction,
            gap_opt_dict=self.gen_wfs.z_opt_gap,
            z_diff_dict=self.gen_wfs.z_difference,
            fd_opt=self.gen_wfs.finite_diff_opt,
            LCAO_dict=self.gen_wfs.LCAO,
        )


def workflow(runner):
    runner.run_workflow(Workflow())
