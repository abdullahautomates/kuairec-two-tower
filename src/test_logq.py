"""Hand-checked test of the logQ arithmetic in train.batch_loss.

Run:  uv run python src/test_logq.py

The example, small enough to work out on paper:
  A batch of two rows: two different users, two different videos (IDs 0, 1).
  Every plain score is 0, so the model has no opinion yet.

  Without the correction, each row is a coin flip between two columns:
      loss = ln(2) = 0.693

  Now say video 0 is 4 times as frequent as video 1: q = [0.8, 0.2].
  The correction subtracts ln(q) from each column:
      column 0: 0 - ln(0.8) = 0.223        column 1: 0 - ln(0.2) = 1.609
  The rare video (column 1) now looks like the stronger candidate. Softmax
  turns the two corrected scores into chances: e^0.223 = 1.25, e^1.609 = 5,
  total 6.25.
      row 0 (right answer = video 0): chance 1.25 / 6.25 = 0.2 -> loss -ln(0.2) = 1.609
      row 1 (right answer = video 1): chance 5 / 6.25 = 0.8    -> loss -ln(0.8) = 0.223
      average loss = (1.609 + 0.223) / 2 = 0.916

  Reading it: the row whose right answer is the FREQUENT video has the big
  loss, so training raises that video's plain score. That is the correction
  giving popular videos back the credit that in-batch training takes away.
"""

import math

import torch

from train import batch_loss

user_vecs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
video_vecs = torch.zeros(2, 2)                      # every plain score = 0
user_ids, video_ids = torch.tensor([10, 11]), torch.tensor([0, 1])

plain = batch_loss(user_vecs, video_vecs, user_ids, video_ids).item()
corrected = batch_loss(user_vecs, video_vecs, user_ids, video_ids,
                       log_q=torch.log(torch.tensor([0.8, 0.2]))).item()

assert abs(plain - math.log(2)) < 1e-6, plain
print(f"ok  without correction: {plain:.3f}")
assert abs(corrected - (-math.log(0.2) - math.log(0.8)) / 2) < 1e-6, corrected
print(f"ok  with correction:    {corrected:.3f}")
print("\nall logQ checks passed")
