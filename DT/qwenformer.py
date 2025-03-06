
from qwen import Qwen2PreTrainedModel
from qwen import Qwen2Config


config = Qwen2Config()
model = Qwen2PreTrainedModel(config)
print(model)
# Output: Qwen2PreTrainedModel(