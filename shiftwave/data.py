"""Equal-domain window sampling, independently shuffled inside each domain."""
from bisect import bisect_right
import random
from shift500.data import Clips as BaseClips


class Clips(BaseClips):
    cycle = ('gopro', 'dvd', 'bsd')

    def locate(self, index):
        if index < 0 or index >= self.total:
            raise IndexError(index)
        domain = self.cycle[index % 3]
        occurrence = index // 3
        count = self.ends[domain][-1]
        epoch, position = divmod(occurrence, count)
        if self.orders.get(domain, (-1, None))[0] != epoch:
            order = list(range(count))
            random.Random(self.seed + epoch*9176 + sum(map(ord, domain))).shuffle(order)
            self.orders[domain] = (epoch, order)
        window = self.orders[domain][1][position]
        record = bisect_right(self.ends[domain], window)
        start = window - (self.ends[domain][record-1] if record else 0)
        return domain, self.records[domain][record], start
