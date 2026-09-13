from experiment_config import ExperimentConfig
from experiment_runner import ResNetExperiment


class CBAMExperiment(ResNetExperiment):
    def config(self):
        return ExperimentConfig(run_name="cbam", attention="cbam")


if __name__ == "__main__":
    CBAMExperiment().run()
