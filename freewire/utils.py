import time

class Timer:
  def __init__(self):
    self.data = {}
    self.start_time = None
    self.current_name = None
  def start(self, name):
    """Begins timing an activity of specified name.
    """
    assert self.start_time is None, "Can't time when another activity is being timed"
    self.start_time = time.time()
    self.current_name = name
    if name not in self.data.keys():
      self.data.update({name:0})
  def end(self, name):
    assert self.start_time is not None, "Start timing before calling end()"
    assert name == self.current_name, "Called end on name that wasn't started. Current name: {}".format(self.current_name)
    delta = time.time() - self.start_time
    self.data[name] += delta
    self.start_time = None
    self.current_name = None
  def reset(self):
    self.data = {}
    self.start_time = None
    self.current_name = None
  def report(self):
    """Display report sorted by most time consuming activity.
    """
    activities = list(self.data.keys())
    activity_times = [self.data[a] for a in activities]
    sorted_zip = sorted(zip(activities, activity_times), key=lambda x: x[1], reverse=True)
    for activity, time in sorted_zip:
      print(activity, time)

def size(tensor):
  byte = tensor.nelement() * tensor.element_size()
  unit = "bytes"
  if byte > 1_000_000:
    byte = byte / 1_000_000
    unit = "MB"
  elif byte > 1_000:
    byte = byte / 1_000_000
    unit = "KB"
  return str(byte) + " " + unit

if __name__ == "__main__":
  t = Timer()
  for trial in range(0, 10):
    t.start('small_loop')
    j = 0
    for i in range(0, 10_000):
      j += 1
    t.end('small_loop')

    t.start('large_loop')
    j = 0
    for i in range(0, 1_000_000):
      j += 1
    t.end('large_loop')