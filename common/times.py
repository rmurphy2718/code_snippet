import time
from collections import defaultdict

import tqdm
from tqdm import tqdm


class Times:
    def __init__(self):
        self.start = time.time()
        self.times = []
        self.events = []
        self.pbar = tqdm(desc="Progress")

    def add(self, event):
        self.pbar.update()
        self.pbar.set_description(event)
        self.times.append(time.time())
        self.events.append(event)

    def add_all(self, that):
        self.times += that.times
        self.events += that.events
        return self

    def compute(self):
        self.pbar.close()
        tmap = defaultdict(float)
        for time, event in zip(self.times, self.events):
            t = time - self.start
            tmap[event] += t
            self.start = time
        return tmap
