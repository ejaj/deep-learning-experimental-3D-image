import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

# model = models.resnet18()
# inputs = torch.randn(5, 3, 224, 224)
#
# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#     with record_function("model_inference"):
#         model(inputs)

# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))

# model = models.resnet18().cuda()
# inputs = torch.randn(5, 3, 224, 224).cuda()
#
# with profile(activities=[
#     ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#     with record_function("model_inference"):
#         model(inputs)
#
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


# model = models.resnet18()
# inputs = torch.randn(5, 3, 224, 224)
#
# with profile(activities=[ProfilerActivity.CPU],
#              profile_memory=True, record_shapes=True) as prof:
#     model(inputs)

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
# print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))


model = models.resnet18().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)

prof.export_chrome_trace("trace.json")
