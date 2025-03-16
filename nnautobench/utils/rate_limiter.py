import time
from collections import deque
from threading import Lock

class RateLimiter:
    def __init__(self, max_requests, window_seconds):
        """Initialize the rate limiter with max requests and time window."""
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_times = deque()
        self.lock = Lock()

    def wait_for_token(self):
        """Wait until a request can be made without exceeding the rate limit."""
        with self.lock:
            current_time = time.time()
            # Remove timestamps outside the time window
            while self.request_times and self.request_times[0] <= current_time - self.window_seconds:
                self.request_times.popleft()
            if len(self.request_times) < self.max_requests:
                self.request_times.append(current_time)
                return
            else:
                # Wait until the oldest request falls outside the window
                time_to_wait = self.request_times[0] + self.window_seconds - current_time
                if time_to_wait > 0:
                    time.sleep(time_to_wait)
                # After waiting, add the new request time
                self.request_times.append(time.time())