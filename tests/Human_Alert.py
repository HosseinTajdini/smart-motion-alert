import time
class Alert:
    def __init__(self,alert_interval):
        self.ALERT_INTERVAL = alert_interval


    def printAlert(self,detected_object):
        last_alert_time = 0
        if detected_object is not None:
            now = time.time()
            if now - last_alert_time > self.ALERT_INTERVAL:
                print(f"[ALERT] {detected_object['label']} detected at {time.ctime(now)}")
                last_alert_time = now
