from experiment_config import ExperimentConfig
from experiment_runner import ResNetExperiment


class SEExperiment(ResNetExperiment):
    def config(self):
        return ExperimentConfig(run_name="se", attention="se")


if __name__ == "__main__":
    SEExperiment().run()
