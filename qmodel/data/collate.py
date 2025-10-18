import torch

from line_profiler import profile

@profile
def custom_collate_fn(batch):
    return batch
    data_list, target_list, code_dates = zip(*batch)

    data_tensor = torch.stack(data_list, dim=0)
    target_tensor = torch.stack(target_list, dim=0)
    code_dates_tensor = torch.stack(code_dates, dim=0)

    return data_tensor, target_tensor, code_dates_tensor
