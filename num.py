import torch

# Example tensor
tensor = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]
                    ])

# Sum along dim=0 (rows)
sum_along_rows = torch.sum(tensor, dim=0)
# Output shape: (1, 4)
print(sum_along_rows)
# Sum along dim=1 (columns)
sum_along_columns = torch.sum(tensor, dim=1)
# Output shape: (3, 1)
print(sum_along_columns)