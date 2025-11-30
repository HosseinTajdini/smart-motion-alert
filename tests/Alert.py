import time
class Alert:
    def __init__(self,alert_interval):
        self.ALERT_INTERVAL = alert_interval


    def printAlert(self,motions):
        last_alert_time = 0
        if len(motions) > 0:
            now = time.time()
            if now - last_alert_time > self.ALERT_INTERVAL:
                print(f"[ALERT] Motion detected at {time.ctime(now)}")
                last_alert_time = now
