import psutil, time, csv, datetime

log_file = "lsmc_log.csv"
duration = 3600  # 1시간
interval = 10  # 10초마다 기록

with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "cpu_percent", "memory_percent", "net_sent_MB", "net_recv_MB"])

    net1 = psutil.net_io_counters()
    end = time.time() + duration

    while time.time() < end:
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent
        net2 = psutil.net_io_counters()
        sent = (net2.bytes_sent - net1.bytes_sent) / (1024 * 1024)
        recv = (net2.bytes_recv - net1.bytes_recv) / (1024 * 1024)
        net1 = net2
        print(f"CPU: {cpu:.1f}% | MEM: {mem:.1f}% | ↑{sent:.2f}MB ↓{recv:.2f}MB", flush=True)
        writer.writerow([datetime.datetime.now(), cpu, mem, f"{sent:.2f}", f"{recv:.2f}"])
        f.flush()  # 강제 기록

