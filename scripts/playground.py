import time

from tqdm.auto import tqdm

for i in tqdm(range(1000), position=0, leave=True):
    time.sleep(0.05)
