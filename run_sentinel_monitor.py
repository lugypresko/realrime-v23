import subprocess
import psutil
import time
import sys
import threading

# Define the command to run the sentinel test
cmd = [sys.executable, "03_sentinel_test.py"]

print("Starting Sentinel Test with Monitoring...")
print("PLEASE SPEAK NOW (Test runs for 20 seconds)")
print("-" * 60)

# Start the process
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')

# Monitoring variables
cpu_usages = []
output_lines = []

def monitor_cpu():
    ps_process = psutil.Process(process.pid)
    while process.poll() is None:
        try:
            cpu = ps_process.cpu_percent(interval=0.5)
            cpu_usages.append(cpu)
        except:
            break

def read_output():
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line, end='')  # Echo to console
            output_lines.append(line)

# Start threads
monitor_thread = threading.Thread(target=monitor_cpu)
output_thread = threading.Thread(target=read_output)

monitor_thread.start()
output_thread.start()

# Let it run for 20 seconds
try:
    time.sleep(20)
except KeyboardInterrupt:
    pass

# Terminate
process.terminate()
monitor_thread.join()
output_thread.join()

print("-" * 60)
print("TEST COMPLETE")
print("-" * 60)

# Analysis
avg_cpu = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
max_cpu = max(cpu_usages) if cpu_usages else 0

print(f"Average CPU Usage: {avg_cpu:.1f}%")
print(f"Max CPU Usage:     {max_cpu:.1f}%")

full_output = "".join(output_lines)
silence_detected = "SILENCE DETECTED" in full_output
buffer_extracted = "Extracted" in full_output and "buffer" in full_output

print(f"Silence Detection: {'✅ YES' if silence_detected else '❌ NO'}")
print(f"Buffer Extraction: {'✅ YES' if buffer_extracted else '❌ NO'}")

if silence_detected and buffer_extracted and avg_cpu < 30:
    print("Result: PASS")
else:
    print("Result: FAIL (Check output above)")

# Save log
with open("sentinel_report.txt", "w", encoding="utf-8") as f:
    f.write(full_output)
    f.write("\n" + "-"*60 + "\n")
    f.write(f"Avg CPU: {avg_cpu:.1f}%\n")
    f.write(f"Max CPU: {max_cpu:.1f}%\n")
