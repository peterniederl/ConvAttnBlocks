from experiment_config import ExperimentConfig
from experiment_runner import ResNetExperiment


class BaselineExperiment(ResNetExperiment):
    def config(self):
        return ExperimentConfig(run_name="baseline", attention="none")


if __name__ == "__main__":
    BaselineExperiment().run()
