"""Stop running intake experiments at the user account-quota threshold."""
import importlib,json,time
import usage
previous=None
while not usage.STOP.exists():
    importlib.reload(usage)
    result=usage.status()
    remaining=result.get('remaining_percent')
    if remaining!=previous:print(json.dumps(result),flush=True);previous=remaining
    if result.get('stop'):break
    time.sleep(10)
