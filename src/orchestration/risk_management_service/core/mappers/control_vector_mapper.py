from services.problem_dispatcher_service import ControlVector as PDControlVector
from services.solution_updater_service import ControlVector as SUControlVector


class ControlVectorMapper:
    @staticmethod
    def convert_su_to_pd(
        su_control_vectors: list[SUControlVector] | None,
    ) -> list[PDControlVector] | None:
        """
        Converts a list of ControlVector objects from SolutionUpdaterService's format
        to ProblemDispatcherService's format.

        Args:
            su_control_vectors (list[SUControlVector] | None):
                A list of ControlVector objects from the SolutionUpdaterService, or None.

        Returns:
            list[PDControlVector] | None:
                A list of converted ControlVector objects in ProblemDispatcherService format,
                or None if the input is None.
        """

        if su_control_vectors is None:
            return None
        return [PDControlVector(**vars(cv)) for cv in su_control_vectors]
