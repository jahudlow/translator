from pydantic import BaseModel
import os

import modal


# For Modal naming get the filename, which should be
# either cpu.py or gpu.py. This is converted to the Modal
# endpoint name to specify either CPU or GPU.
processor = os.path.basename(__file__).split('.')[0]


# Download the model.
cache_dir = "/vol/cache"
def download_model():
    from comet import download_model, load_from_checkpoint
    model_path = download_model("wmt20-comet-qe-da", saving_directory=cache_dir)
    load_from_checkpoint(model_path)


# Manage suffix on modal endpoint if testing.
suffix = ''
if os.environ.get('MODAL_TEST') == 'TRUE':
    suffix = '_test'


# For Modal naming get the directory name, which should include
# the task name and model name.
taskname = os.getcwd().split('/')[-2]
modelname = os.getcwd().split('/')[-1]


# Define the modal stub.
stub = modal.Stub(
    "qe-comet" + suffix,
    image=modal.Image.from_dockerhub("dwhitena/comet").run_function(
        download_model,
    ),
)


# Define the model.
class Model:
    def __enter__(self):
        from comet import load_from_checkpoint
        self.model = load_from_checkpoint("/vol/cache/wmt20-comet-qe-da/checkpoints/model.ckpt")

    @stub.function(cpu=8, retries=3)
    def predict(self, input: dict) -> str:
        data = [{
            "src": input['source'],
            "mt": input['translation'],
            #"ref": input.reference,
        }]
        _, sys_score = self.model.predict(data, batch_size=8, gpus=0)
        return sys_score


if __name__ == "__main__":
    stub.serve()