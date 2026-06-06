import subprocess, sys
result = subprocess.run(
    ["netstat", "-ano"],
    capture_output=True, text=True
)
pids = set()
for line in result.stdout.splitlines():
    if ":8080" in line and "LISTENING" in line:
        pid = line.strip().split()[-1]
        pids.add(pid)
print(f"PIDs listening on 8080: {pids}")
for pid in pids:
    subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
    print(f"Killed {pid}")
